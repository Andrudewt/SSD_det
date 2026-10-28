import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# =============================================================================
# 6. MULTIBOX LOSS (Smooth L1 + CrossEntropy + Hard Negative Mining)
# =============================================================================

class MultiBoxLoss(nn.Module):
    """
    Комбинированная функция потерь SSD.

    Состоит из двух частей:
        1. Loc Loss: Smooth L1 только для positive якорей (matched с GT)
        2. Conf Loss: CrossEntropy для positive + negative якорей

    Hard Negative Mining:
        Из всех negative якорей выбираем top(neg_pos_ratio * num_pos)
        с наибольшим loss. Это решает проблему дисбаланса классов
        (background якорей ~99% от всех).

    Итоговый loss = (loc_loss + conf_loss) / num_positives
    """
    def __init__(self, cfg=config):
        super().__init__()
        self.num_classes = cfg.NUM_CLASSES
        self.neg_pos_ratio = cfg.NEG_POS_RATIO
        self.variance = cfg.VARIANCE

    def forward(self, loc_preds, conf_preds, loc_targets, conf_targets):
        """
        Args:
            loc_preds:   [batch, num_priors, 4] — предсказанные смещения
            conf_preds:  [batch, num_priors, num_classes] — логиты классов
            loc_targets: [batch, num_priors, 4] — целевые смещения
            conf_targets:[batch, num_priors] — целевые метки
        Returns:
            tuple: (loc_loss, conf_loss) — скаляры
        """
        batch_size = loc_preds.size(0)
        num_priors = loc_preds.size(1)

        # --- Локализационный loss: только positive якоря ---
        # pos: [batch, num_priors] — булев маска positive якорей
        # Positive = всё, кроме background
        pos = conf_targets > 0
        num_pos = pos.sum(dim=1, keepdim=True)  # [batch, 1] — число positive на изображение

        # Smooth L1 (Huber loss) между предсказанными и целевыми смещениями
        # reduction='sum' — суммируем по всем positive якорям
        loc_loss = F.smooth_l1_loss(loc_preds[pos], loc_targets[pos], reduction='sum')

        # --- Hard Negative Mining ---
        # Вычисляем loss для ВСЕХ якорей (для ранжирования negative)
        # [batch*num_priors, num_classes] → [batch*num_priors]
        batch_conf = conf_preds.view(-1, self.num_classes)
        loss_c = F.cross_entropy(batch_conf, conf_targets.view(-1), reduction='none')
        loss_c = loss_c.view(batch_size, -1)  # [batch, num_priors]

        # Positive якоря не участвуют в выборе negative (обнуляем их loss)
        loss_c[pos] = 0

        # Сортируем loss по убыванию и выбираем top negative
        _, loss_idx = loss_c.sort(dim=1, descending=True)   # [batch, num_priors]
        _, idx_rank = loss_idx.sort(dim=1)                 # [batch, num_priors] — ранги

        # Число negative = 3 * число positive (но не больше доступных)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=num_priors - num_pos)

        # Маска выбранных negative якорей
        neg = idx_rank < num_neg.expand_as(idx_rank)  # [batch, num_priors]

        # --- Confidence loss: по positive + отобранным negative ---
        # Расширяем маски до размерности conf_preds: [batch, num_priors, num_classes]
        pos_idx = pos.unsqueeze(2).expand_as(conf_preds)
        neg_idx = neg.unsqueeze(2).expand_as(conf_preds)

        # Отбираем предсказания и метки для positive и negative
        conf_p = conf_preds[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_targets[(pos + neg).gt(0)]

        conf_loss = F.cross_entropy(conf_p, targets_weighted, reduction='sum',)    # label_smoothing=0.1

        # --- Нормализация ---
        N = num_pos.sum().float()  # Общее число positive в батче

        # Защита от деления на 0 (если в батче нет ни одного объекта)
        if N == 0:
            return torch.tensor(0.0, device=loc_preds.device), torch.tensor(0.0, device=loc_preds.device)

        loc_loss /= N
        conf_loss /= N

        # суммируется в train
        return loc_loss, conf_loss
