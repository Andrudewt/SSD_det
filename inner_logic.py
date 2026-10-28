import torch
import torch.nn.functional as F
import math
import config
import cv2


# =============================================================================
# 4. ГЕНЕРАТОР ЯКОРНЫХ БОКСОВ (Default / Prior Boxes)
# =============================================================================

class AnchorBoxGenerator:
    """
    Генератор якорных боксов (default / prior boxes) по спецификации SSD.

    Алгоритм для каждого уровня k с разрешением f x f:
        Для каждой ячейки (i, j):
            1. Малый квадрат:  (cx, cy, s_k, s_k)
            2. Большой квадрат: (cx, cy, sqrt(s_k * s_{k+1}), sqrt(s_k * s_{k+1}))
            3. Для каждого aspect_ratio r:
               (cx, cy, s_k*sqrt(r), s_k/sqrt(r))
               (cx, cy, s_k/sqrt(r), s_k*sqrt(r))

    Все координаты нормализованы в [0, 1].
    Формат выхода: [cx, cy, w, h] (центр + размеры).

    Attributes:
        Всего якорей: 7*14*3 = 294
    """
    def __init__(self, cfg=config):
        self.cfg = cfg
        self.feature_maps = cfg.FEATURE_MAPS
        self.min_sizes = cfg.MIN_SIZES
        self.max_sizes = cfg.MAX_SIZES
        self.aspect_ratios = cfg.ASPECT_RATIOS
        self.image_size = cfg.IMAGE_SIZE

    def generate(self):
        """
        Returns:
            torch.Tensor: [294, 4] — якорные боксы в формате [cx, cy, w, h]
        """
        boxes = []
        # Центр ячейки (i, j) в нормализованных координатах [0, 1] относ размера fm
        for k, f in enumerate(self.feature_maps):
            for iy in range(f[0]):
                for jx in range(f[1]):
                    cy = (iy + 0.5) / f[0]
                    cx = (jx + 0.5) / f[1]

                    # 1. Малый квадратный бокс масштаба s_k
                    s_k = self.min_sizes[k]
                    boxes.append([cx, cy, s_k[0], s_k[1]])

                    # для маркировки большой квадрат не использ
                    # 2. Большой квадратный бокс масштаба sqrt(s_k * s_{k+1})
                    # s_k_prime = math.sqrt(s_k * max_sizes[k])
                    # boxes.append([cx, cy, s_k_prime, s_k_prime])

                    # 3. Прямоугольные боксы для каждого aspect ratio
                    for ar in self.aspect_ratios[k]:
                        # Горизонтально-вытянутый
                        boxes.append([cx, cy, s_k[0] * math.sqrt(ar), s_k[1] / math.sqrt(ar)])
                        # Вертикально-вытянутый
                        boxes.append([cx, cy, s_k[0] / math.sqrt(ar), s_k[1] * math.sqrt(ar)])

        boxes = torch.tensor(boxes, dtype=torch.float32)  # [Nancbox, 4]
        # Защита: ограничиваем координаты и размеры [0, 1] при больших max_sizes может выйти за 1.0
        boxes = torch.clamp(boxes, min=0.0, max=1.0)
        return boxes


# =============================================================================
# 5. СТРАТЕГИЯ СОПОСТАВЛЕНИЯ И КОДИРОВАНИЕ
# =============================================================================

def intersect(box_a, box_b):
    """
    Вычисляет площадь пересечения двух наборов боксов.

    Args:
        box_a: [A, 4] — первый набор в формате [xmin, ymin, xmax, ymax]
        box_b: [B, 4] — второй набор в том же формате
    Returns:
        torch.Tensor: [A, B] — матрица площадей пересечения
    """
    A = box_a.size(0)
    B = box_b.size(0)

    # Находим координаты пересечения
    # expand(A, B, 2) транслирует размерности для попарного сравнения
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),  # [A, 1, 2] → [A, B, 2]
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2)   # [1, B, 2] → [A, B, 2]
    )
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2)
    )

    # Ширина и высота пересечения (обрезаем отрицательные = нет пересечения)
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]  # [A, B]


def IoU(box_a, box_b):
    """
    Вычисляет IoU между двумя наборами боксов.

    IoU = Area(Intersection) / Area(Union)

    Args:
        box_a: [A, 4] в формате [xmin, ymin, xmax, ymax]
        box_b: [B, 4] в том же формате
    Returns:
        torch.Tensor: [A, B] — матрица IoU
    """
    inter = intersect(box_a, box_b)

    # Площади боксов
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)

    union = area_a + area_b - inter
    return inter / union  # [A, B]


def encode(matched, priors, variances=config.VARIANCE):
    """
    Кодирует ground truth боксы относительно якорных боксов (priors).

    Преобразование в пространство смещений:
        g_cx = (gt_cx - prior_cx) / (variance[0] * prior_w)
        g_cy = (gt_cy - prior_cy) / (variance[1] * prior_h)
        g_w  = log(gt_w / prior_w) / variance[1]
        g_h  = log(gt_h / prior_h) / variance[1]

    Это позволяет сети предсказывать малые смещения вместо абсолютных координат.

    Args:
        matched: [num_priors, 4] — GT боксы в формате [xmin, ymin, xmax, ymax]
        priors:  [num_priors, 4] — якорные боксы [cx, cy, w, h]
        variances: [2] — масштабирующие коэффициенты
    Returns:
        torch.Tensor: [num_priors, 4] — закодированные смещения
    """
    # Центры GT боксов
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2.0 - priors[:, :2]
    # Нормализуем на размер якоря с учётом variance
    g_cxcy /= (variances[0] * priors[:, 2:])

    # Размеры GT боксов
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    # Логарифмическое кодирование размеров
    g_wh = torch.log(g_wh) / variances[1]

    return torch.cat([g_cxcy, g_wh], dim=1)


def decode(loc, priors, variances=config.VARIANCE):
    """
    Декодирует предсказанные смещения в абсолютные координаты боксов.

    Обратное преобразование encode():
        cx = prior_cx + loc_cx * variance[0] * prior_w
        cy = prior_cy + loc_cy * variance[0] * prior_h
        w  = prior_w * exp(loc_w * variance[1])
        h  = prior_h * exp(loc_h * variance[1])

    Затем конвертируем [cx, cy, w, h] → [xmin, ymin, xmax, ymax].

    Args:
        loc: [batch, num_priors, 4] или [num_priors, 4] — предсказанные смещения
        priors: [num_priors, 4] или [1, num_priors, 4] — якорные боксы
    Returns:
        torch.Tensor: боксы в формате [xmin, ymin, xmax, ymax]
    """
    if loc.dim() == 3:
        priors = priors.unsqueeze(0)  # [1, num_priors, 4] для broadcasting

    # Восстанавливаем центры и размеры
    boxes = torch.cat([
        priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:],
        priors[..., 2:] * torch.exp(loc[..., 2:] * variances[1])
    ], dim=-1)  # [batch, num_priors, 4] или [num_priors, 4]

    # Конвертация: center-form → point-form
    result = torch.cat([
        boxes[..., :2] - boxes[..., 2:] / 2.0,  # xmin, ymin
        boxes[..., :2] + boxes[..., 2:] / 2.0   # xmax, ymax
    ], dim=-1)

    return result


def match(threshold, truths, priors, labels, loc_t, conf_t, idx):
    """
    Сопоставляет (match) GT боксы с якорными боксами для одного изображения.

    Алгоритм:
        1. Считаем IoU между всеми GT и всеми якорями → [num_gt, num_priors]
        2. Для каждого GT находим лучший якорь (max IoU)
        3. Для каждого якоря находим лучший GT (max IoU)
        4. Гарантируем, что лучший якорь для каждого GT всегда считается positive
           (даже если его IoU < threshold)
        5. Якоря с IoU >= threshold → positive (класс GT)
        6. Якоря с IoU < threshold → background

    Args:
        threshold (float): Порог IoU для positive matching (обычно 0.5)
        truths: [num_objs, 4] — GT боксы в формате [xmin, ymin, xmax, ymax]
        priors: [num_priors, 4] — якорные боксы [cx, cy, w, h]
        labels: [num_objs] — метки классов GT
        loc_t: [batch, num_priors, 4] — тензор для записи закодированных локализаций
        conf_t: [batch, num_priors] — тензор для записи меток классов
        idx (int): Индекс изображения в батче
    """
    # Конвертируем якоря из center-form в point-form для IoU
    overlaps = IoU(truths, point_form(priors))  # [num_gt, num_priors]

    # --- Шаг 1: для каждого GT находим лучший якорь ---
    # best_prior_overlap: [num_gt] — максимальное IoU для каждого GT
    # best_prior_idx: [num_gt] — индекс лучшего якоря для каждого GT
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # --- Шаг 2: для каждого якоря находим лучший GT ---
    # best_truth_overlap: [num_priors] — максимальное IoU для каждого якоря
    # best_truth_idx: [num_priors] — индекс лучшего GT для каждого якоря
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx = best_truth_idx.squeeze(0)
    best_truth_overlap = best_truth_overlap.squeeze(0)

    # --- Шаг 3: force match — гарантируем, что лучший якорь для каждого GT
    # всегда positive, даже если его IoU ниже threshold ---
    best_prior_idx = best_prior_idx.squeeze(1)
    best_prior_overlap = best_prior_overlap.squeeze(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2.0)  # Устанавливаем IoU=2.0

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # --- Шаг 4: формируем целевые значения ---
    # matched: [num_priors, 4] — GT бокс, сопоставленный каждому якорю
    matches = truths[best_truth_idx]

    # Метки классов:
    conf = labels[best_truth_idx]

    # Фон: якоря с IoU < threshold получают класс background
    conf[best_truth_overlap < threshold] = 0

    # Кодируем смещения относительно якорей
    loc = encode(matches, priors)

    # Записываем в батчевые тензоры
    loc_t[idx] = loc
    conf_t[idx] = conf


def point_form(boxes):
    """
    Конвертирует боксы из center-form [cx, cy, w, h] в point-form [xmin, ymin, xmax, ymax].

    Args:
        boxes: [N, 4] в формате [cx, cy, w, h]
    Returns:
        torch.Tensor: [N, 4] в формате [xmin, ymin, xmax, ymax]
    """
    return torch.cat([
        boxes[:, :2] - boxes[:, 2:] / 2.0,  # xmin, ymin
        boxes[:, :2] + boxes[:, 2:] / 2.0   # xmax, ymax
    ], dim=1)


# =============================================================================
# 8. ДЕТЕКЦИЯ (Инференс с NMS (Non-Maximum Suppression))
# =============================================================================

class Detector:
    """
    Пост-обработка предсказаний SSD для получения финальных боксов.

    Алгоритм инференса:
        1. Softmax по классам для получения вероятностей
        2. Декодирование смещений → абсолютные координаты
        3. Для каждого класса:
           - Отбор якорей с confidence > CONF_THRESH
           - NMS (Non-Maximum Suppression) для удаления дубликатов
        4. Сборка результатов по всем классам

    NMS: если два бокса одного класса имеют IoU > NMS_THRESH,
         оставляем только тот, у которого выше confidence.
    """
    def __init__(self, cfg=config):
        self.cfg = cfg
        self.priors = AnchorBoxGenerator(cfg).generate()
        if torch.cuda.is_available():
            self.priors = self.priors.cuda()

    def detect(self, loc_preds, conf_preds, conf_thresh=None, nms_thresh=None):
        """
        Args:
            loc_preds:  [batch, num_ancbox, 4] — предсказанные смещения
            conf_preds: [batch, num_ancbox, num_classes] — логиты классов
            conf_thresh: порог уверенности (если None, используется cfg.CONF_THRESH)
            nms_thresh: порог IoU для NMS (если None, используется cfg.NMS_THRESH)
        Returns:
            list длиной batch: каждый элемент — [N, 6] тензор
                [x1, y1, x2, y2, score, class_id]
                Если детекций нет — [1, 6] из нулей (padding)
        """
        if conf_thresh is None:
            conf_thresh = self.cfg.CONF_THRESH
        if nms_thresh is None:
            nms_thresh = self.cfg.NMS_THRESH

        priors = self.priors.to(loc_preds.device)
        batch_size = loc_preds.size(0)
        num_classes = conf_preds.size(2)

        # Softmax по классам: [batch, 8732, num_classes]
        conf_preds = F.softmax(conf_preds, dim=2)

        # Декодируем смещения в боксы [xmin, ymin, xmax, ymax]
        boxes = decode(loc_preds, priors)
        boxes = torch.clamp(boxes, min=0.0, max=1.0)  # Защита от выхода за [0,1]

        output = []
        for i in range(batch_size):  # Обрабатываем каждое изображение отдельно
            results = []

            # Перебираем все классы
            for cl in range(1, num_classes):
                # Отбираем якоря, уверенность в классе cl > threshold
                c_mask = conf_preds[i, :, cl] > conf_thresh
                scores = conf_preds[i, c_mask, cl]  # [N_candidates]

                if scores.size(0) == 0:
                    continue  # Нет кандидатов для этого класса

                boxes_filtered = boxes[i, c_mask, :]  # [N_candidates, 4]

                # Non-Maximum Suppression через OpenCV
                # Возвращает индексы боксов, которые нужно оставить
                # на вход принимает [xmin, ymin, xmax, ymax]
                ids = cv2.dnn.NMSBoxes(
                    boxes_filtered.cpu().numpy().tolist(),
                    scores.cpu().numpy().tolist(),
                    conf_thresh,
                    nms_thresh
                )
                if len(ids) > 0:
                    ids = ids.flatten()
                    # Собираем результат: [x1, y1, x2, y2, score, class]
                    results.append(torch.cat([
                        boxes_filtered[ids],
                        scores[ids].unsqueeze(1),
                        torch.full((len(ids), 1), cl, dtype=torch.float32, device=scores.device)
                    ], dim=1))

            if len(results) > 0:
                output.append(torch.cat(results, dim=0))  # [N_total, 6]
            else:
                # Padding для сохранения размерности (если нет детекций)
                output.append(torch.zeros((1, 6), device=loc_preds.device))

        return output
