import torch
from inner_logic import Detector
import numpy as np
import config


# =============================================================================
# 10. ОЦЕНКА (mAP) И ВИЗУАЛИЗАЦИЯ
# =============================================================================

# в обучении не учавствует
def compute_iou(box1, box2):
    """
    Вычисляет IoU между двумя боксами в формате [x1, y1, x2, y2].

    Args:
        box1, box2: iterable из 4 float в [0, 1]
    Returns:
        float: IoU в диапазоне [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def evaluate_map(model, dataloader, cfg=config, iou_thresh=0.5):
    """
    Вычисляет mean Average Precision (mAP) по метрике VOC Pascal.

    Алгоритм для одного класса:
        1. Собираем все предсказания по всем изображениям
        2. Сортируем по убыванию confidence
        3. Для каждого предсказания проверяем IoU с GT:
           - Если IoU >= 0.5 и GT ещё не занят → True Positive
           - Иначе → False Positive
        4. Строим PR-кривую и считаем AP методом 11-point interpolation

    Args:
        model: обученная модель SSDMobileNet
        dataloader: DataLoader с валидационными данными
        iou_thresh: порог IoU для счёта детекции правильной (default 0.5)
    Returns:
        float: mAP@iou_thresh (усреднённое по всем классам)
    """
    model.eval()
    detector = Detector(cfg)

    all_detections = []   # Предсказания модели
    all_annotations = []    # Ground truth

    with torch.no_grad():
        for images, targets in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()

            # Предсказания модели
            loc_preds, conf_preds = model(images)
            detections = detector.detect(loc_preds, conf_preds)

            for i, target in enumerate(targets):
                # GT: нормализованные [0,1] боксы, метки 0-based
                gt_boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else target['boxes']
                gt_labels = target['labels'].cpu().numpy() if torch.is_tensor(target['labels']) else target['labels']

                all_annotations.append({
                    'boxes': gt_boxes,
                    'labels': gt_labels
                })

                # Убираем padding (строки из нулей)
                dets = detections[i].cpu().numpy()
                valid_mask = dets.sum(axis=1) != 0
                dets = dets[valid_mask]

                all_detections.append({
                    'boxes': dets[:, :4] if len(dets) > 0 else np.zeros((0, 4)),
                    'scores': dets[:, 4] if len(dets) > 0 else np.zeros(0),
                    'classes': dets[:, 5] if len(dets) > 0 else np.zeros(0)
                })

    # Считаем AP для каждого класса (кроме background)
    aps = []
    for cl in range(1, cfg.NUM_CLASSES):
        # Собираем все предсказания класса cl по всем изображениям
        preds = []
        for img_id in range(len(all_detections)):
            dets = all_detections[img_id]
            mask = dets['classes'] == cl
            if mask.sum() == 0:
                continue
            for idx in np.where(mask)[0]:
                preds.append({
                    'img_id': img_id,
                    'bbox': dets['boxes'][idx],
                    'score': dets['scores'][idx]
                })

        if len(preds) == 0:
            continue  # Нет предсказаний для этого класса

        # Сортируем по убыванию confidence
        preds.sort(key=lambda x: x['score'], reverse=True)

        # Общее число GT объектов класса cl
        npos = sum((ann['labels'] == cl).sum() for ann in all_annotations)
        if npos in (0, None):
            print('Warning! npos is ', npos, type(npos))

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        gt_matched = [set() for _ in range(len(all_annotations))]  # Какие GT уже заняты

        for i, pred in enumerate(preds):
            img_id = pred['img_id']
            ann = all_annotations[img_id]

            # GT объекты нужного класса на этом изображении
            valid_gt = ann['labels'] == (cl)
            if valid_gt.sum() == 0:
                fp[i] = 1  # Предсказание, но нет GT → False Positive
                continue

            gt_boxes_cl = ann['boxes'][valid_gt]
            gt_indices = np.where(valid_gt)[0]

            # IoU с каждым GT
            ious = [compute_iou(pred['bbox'], gt) for gt in gt_boxes_cl]
            max_iou = max(ious)
            max_idx = np.argmax(ious)

            # Если IoU >= threshold и GT ещё не занят → True Positive
            if max_iou >= iou_thresh and gt_indices[max_idx] not in gt_matched[img_id]:
                tp[i] = 1
                gt_matched[img_id].add(gt_indices[max_idx])
            else:
                fp[i] = 1

        # Кумулятивные суммы для PR-кривой
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        rec = tp_cumsum / npos
        prec = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # VOC 11-point interpolation AP
        # AP = среднее максимальной precision при recall >= t для t in [0, 0.1, ..., 1.0]
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11
        aps.append(ap)

    return np.mean(aps) if aps else 0.0
