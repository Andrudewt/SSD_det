import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# =============================================================================
# 2. Backbone MOBILENET V1 (Адаптирован под 1 канал)
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution — ключевой блок MobileNet.

    Разделяет стандартную свёртку на два этапа:
        1. Depthwise: свёртка по каждому каналу отдельно (groups=in_channels)
        2. Pointwise: свёртка 1x1 для смешивания каналов

    Вычислительная сложность: ~1/8 от обычной Conv3x3.

    Args:
        in_channels (int): Число входных каналов
        out_channels (int): Число выходных каналов
        stride (int): Шаг свёртки (1 = сохранить размер, 2 = даунсэмплинг)

    Forward:
        Вход:  [B, in_channels, H, W]
        Выход: [B, out_channels, H/stride, W/stride]
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Depthwise: фильтр на канал, без смешивания между каналами
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride, 1,
            groups=in_channels, bias=False  # bias=False т.к. есть BatchNorm
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise: линейная комбинация каналов (1x1 свёртка)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # ReLU6: ограничивает активации [-6, 6], устойчивее для мобильных сетей
        x = F.relu6(self.bn1(self.depthwise(x)))
        x = F.relu6(self.bn2(self.pointwise(x)))
        return x


class MobileNetBackbone(nn.Module):
    """
    Бэкбон на базе MobileNet V1, возвращающий многоуровневые признаки для SSD.

    Структура потока данных:
        [B, 1, 300, 300]
        → Conv1 (stride=2)      → [B, 32, 150, 150]
        → Layers 0-5              → [B, 512, 19, 19]   (feat_19, уровень 1)
        → Layers 6-12             → [B, 1024, 10, 10]  (feat_10, уровень 2)
        → Extras[0]               → [B, 512, 5, 5]     (feat_5, уровень 3)
        → Extras[1]               → [B, 256, 3, 3]     (feat_3, уровень 4)
        → Extras[2]               → [B, 256, 2, 2]     (feat_2, уровень 5)
        → Extras[3]               → [B, 256, 1, 1]     (feat_1, уровень 6)

    Returns:
        list: [feat_19, feat_10, feat_5, feat_3, feat_2, feat_1]
    """
    def __init__(self, in_channels=1, dropout=0.2):
        super().__init__()

        # Начальная стандартная свёртка (не depthwise) для увеличения числа каналов
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # Основная часть MobileNet: 13 DepthwiseSeparable блоков
        # Каждый stride=2 уменьшает пространственное разрешение в 2 раза
        self.layers = nn.ModuleList([
            DepthwiseSeparableConv(32, 64, 1),     # 0: 150x150
            DepthwiseSeparableConv(64, 128, 2),    # 1: 75x75  (даунсэмпл)
            DepthwiseSeparableConv(128, 128, 1),   # 2: 75x75
            DepthwiseSeparableConv(128, 256, 2),   # 3: 38x38  (даунсэмпл)
            DepthwiseSeparableConv(256, 256, 1),   # 4: 38x38
            DepthwiseSeparableConv(256, 512, 2),   # 5: 19x19  ← feat_19
            DepthwiseSeparableConv(512, 512, 1),   # 6: 19x19
            DepthwiseSeparableConv(512, 512, 1),   # 7: 19x19
            DepthwiseSeparableConv(512, 512, 1),   # 8: 19x19
            DepthwiseSeparableConv(512, 512, 1),   # 9: 19x19
            DepthwiseSeparableConv(512, 512, 1),   # 10: 19x19
            DepthwiseSeparableConv(512, 1024, 2),  # 11: 10x10 ← feat_10
            DepthwiseSeparableConv(1024, 1024, 1), # 12: 10x10
            nn.Dropout2d(dropout)
        ])

        # Дополнительные слои для SSD: уменьшают разрешение и генерируют
        # признаки для крупных объектов (мелкие карты = крупные объекты)
        self.extras = nn.ModuleList([
            # 10x10 → 5x5
            nn.Sequential(
                nn.Conv2d(1024, 256, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ),
            # 5x5 → 3x3
            nn.Sequential(
                nn.Conv2d(512, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ),
            # 3x3 → 2x2
            nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ),
            # 2x2 → 1x1
            nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ),
        ])

    def forward(self, x):
        """
        Args:
            x: [B, 1, 300, 300] — батч грейскейл изображений, нормализованных [0,1]
        Returns:
            list из 6 тензоров: [ [B,512,19,19], [B,1024,10,10], [B,512,5,5],
                                 [B,256,3,3], [B,256,2,2], [B,256,1,1] ]
        """
        x = self.conv1(x)  # [B, 32, 150, 150]

        # Проходим через основные слои, сохраняя признаки на уровнях 19x19 и 10x10
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 5:    # После 6-го слоя: 19x19, 512 каналов
                feat_19 = x
            if i == 12:   # После 13-го слоя: 10x10, 1024 канала
                feat_10 = x

        # Дополнительные уровни пирамиды признаков
        feat_5 = self.extras[0](feat_10)   # [B, 512, 5, 5]
        feat_3 = self.extras[1](feat_5)    # [B, 256, 3, 3]
        feat_2 = self.extras[2](feat_3)    # [B, 256, 2, 2]
        feat_1 = self.extras[3](feat_2)    # [B, 256, 1, 1]

        return [feat_19, feat_10, feat_5, feat_3, feat_2, feat_1]


# =============================================================================
# 3. SSD HEAD (Предсказатели локализации и классификации)
# =============================================================================

class SSDHead(nn.Module):
    """
    Головная сеть SSD для одного уровня пирамиды признаков.

    Для каждой ячейки карты признаков и каждого якорного бокса предсказывает:
        - 4 значения смещений (dx, dy, dw, dh) — локализация
        - num_classes логитов — классификация

    Архитектура: две параллельные свёртки 3x3 с padding=1 (сохраняют размер).

    Args:
        in_channels (int): Число каналов входной карты признаков
        num_anchors (int): Число якорных боксов на одну ячейку (например, 6)
        num_classes (int): Число классов (включая background)

    Forward:
        Вход:  [B, in_channels, H, W]
        Выход: loc  — [B, H*W*num_anchors, 4]
               conf — [B, H*W*num_anchors, num_classes]
    """
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # Локализация: 4 параметра на каждый якорный бокс
        self.loc = nn.Conv2d(
            in_channels, num_anchors * 4,
            kernel_size=3, padding=1
        )
        # Классификация: num_classes логитов на каждый якорный бокс
        self.conf = nn.Conv2d(
            in_channels, num_anchors * num_classes,
            kernel_size=3, padding=1
        )

        # Инициализация весов: нормальное распределение с малым std
        # для стабильного старта обучения (предсказания близки к 0)
        self.loc.weight.data.normal_(0, 0.01)
        self.loc.bias.data.zero_()
        self.conf.weight.data.normal_(0, 0.01)
        self.conf.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)

        # Локализация: [B, num_anchors*4, H, W] → [B, H, W, num_anchors*4]
        loc = self.loc(x).permute(0, 2, 3, 1).contiguous()
        # Классификация: [B, num_anchors*num_classes, H, W] → [B, H, W, num_anchors*num_classes]
        conf = self.conf(x).permute(0, 2, 3, 1).contiguous()

        # Выравниваем в плоский список по всем якорям: [B, H*W*num_anchors, ...]
        loc = loc.view(batch_size, -1, 4)
        conf = conf.view(batch_size, -1, self.num_classes)
        return loc, conf


class SSDMobileNet(nn.Module):
    """
    Полная модель SSD с бэкбоном MobileNet.

    Объединяет бэкбон и 6 головных сетей (по одной на каждый уровень).

    Число якорей на ячейку:
        Для каждого уровня: 2 квадрата + 2 * len(aspect_ratios) прямоугольников
        Например, для aspect_ratios=[2]: 2 + 2*1 = 4 якоря
        Для aspect_ratios=[2,3]: 2 + 2*2 = 6 якорей

    Forward:
        Вход:  [B, 1, 300, 300]
        Выход: loc  — [B, 8732, 4]   (смещения для всех якорей)
               conf — [B, 8732, NUM_CLASSES] (логиты классов)
    """
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = MobileNetBackbone(in_channels=config.NUM_CHANNELS)

        # Вычисляем число якорей на ячейку для каждого уровня
        # len(ar)*2: для каждого соотношения сторон 2 бокса (горизонтальный и вертикальный)
        # +2: маленький квадрат + большой квадрат
        self.num_anchors = [len(ar) * 2 + 2 for ar in config.ASPECT_RATIOS]

        # Число каналов на выходе каждого уровня пирамиды признаков
        in_channels = [512, 1024, 512, 256, 256, 256]

        # Создаём 6 головных сетей (по одной на уровень)
        self.heads = nn.ModuleList([
            SSDHead(in_ch, num_anch, num_classes)
            for in_ch, num_anch in zip(in_channels, self.num_anchors)
        ])

        self._init_weights()

    def _init_weights(self):
        """Kaiming инициализация для Conv и стандартная для BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 1, 300, 300]
        Returns:
            loc:  [B, 8732, 4]
            conf: [B, 8732, num_classes]
        """
        features = self.backbone(x)  # 6 карт признаков разного разрешения

        locs = []
        confs = []
        for feat, head in zip(features, self.heads):
            loc, conf = head(feat)
            locs.append(loc)
            confs.append(conf)

        # Конкатенируем предсказания со всех уровней по оси якорей (dim=1)
        loc = torch.cat(locs, dim=1)    # [B, 8732, 4]
        conf = torch.cat(confs, dim=1)  # [B, 8732, num_classes]
        return loc, conf
