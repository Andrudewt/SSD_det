import torch
import numpy as np
import config
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import albumentations as A # type: ignore


def visualize_detections(image, detections, class_names=config.CLASS_NAMES, save_path=None):
    """
    Визуализирует детекции на грейскейл изображении.

    Args:5
        image: [H, W] numpy array (грейскейл, [0, 1]) или torch.Tensor
        detections: [N, 6] — [x1, y1, x2, y2, score, class_id] в нормализованных [0,1]
        class_names: список имён классов
        save_path: если указан, сохраняет изображение в файл
    """
    if torch.is_tensor(image):
        image = image.squeeze().cpu().numpy()

    h, w = image.shape[:2]  # Для грейскейла: (H, W)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    # Цвета для классов
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names) - 1))  # 35 цветов

    for det in detections:
        if det.sum() == 0:
            continue  # Пропускаем padding (строки из нулей)

        x1, y1, x2, y2, score, cl = det.cpu().numpy()

        # Денормализуем [0,1] → пиксельные координаты
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h

        color = colors[int(cl) % len(colors)]
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5,
            f"{class_names[int(cl)]}: {score:.2f}",
            color=color, fontsize=12, weight='bold'
        )

    ax.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1.e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def load_checkpoin(checkpoint, model, optimizer):
    """checkpoint=torch.load(path)"""
    print('loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


train_transform = A.Compose(
    [
    A.Blur(blur_limit=(3, 3), p=0.2),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.3), contrast_limit=(-0.2, 0.2), p=0.2),
    A.RandomGamma(gamma_limit=(50, 150), p=0.2),
    A.GaussNoise(std_range=(0.08, 0.09), p=0.2),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
    ]
)
