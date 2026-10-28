import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as TF
import os
from PIL import Image
from conv_yolo_label_2_ssd import yolo_to_corners
import config
import numpy as np
from utils import train_transform

# =============================================================================
# 7. ДАТАСЕТ
# =============================================================================
# в train -> on epoch end вставить
# tmp_s.split_shuffle()
# tmp_s_val.split_shuffle()


# настроен на разметку YOLO
class RailMarkDset(Dataset):
    """
    Вход:
        image_folder: список путей к изображениям
        label_folder: список путей к .txt

    Выход __getitem__:
        img:    [1, H, W] — torch.Tensor, нормализованный [0, 1]
        boxes:  [num_objs, 4] — torch.Tensor
        labels: [num_objs] — torch.Tensor классы"""

    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.image_list = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))]
        self.transform = transform


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img = Image.open(os.path.join(self.image_folder, img_path)).convert('L')
        # transform
        if self.transform:
            img = np.asarray(img)
            augmented = self.transform(image=img)
            img = augmented['image']

        # Преобразование в тензор (размерность будет [1, H, W], значения от 0.0 до 1.0)
        img_tensor = TF.to_tensor(img)

        boxes = []
        labels = []
        file_path = os.path.join(self.image_folder, img_path.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(file_path, 'r') as file:
            for line in file:
                # Разбиваем строку по запятым и преобразуем в числа
                parts = list(map(float, line.strip().split(' ')))
                # Извлекаем метку класса
                label = parts[0] + 1  # в разметке 0 это "0", а в детекторе 0 это фон
                labels.append(label)
                x1, y1, x2, y2 = yolo_to_corners(*parts[1:], config.IMAGE_SIZE, config.IMAGE_SIZE)

                # Извлекаем координаты бокса и нормализуем в [0, 1]
                x1 /= config.IMAGE_SIZE
                y1 /= config.IMAGE_SIZE
                x2 /= config.IMAGE_SIZE
                y2 /= config.IMAGE_SIZE

                # x1, y1, x2, y2 = parts[1]/config.IMAGE_SIZE , parts[2]/config.IMAGE_SIZE, parts[3]/config.IMAGE_SIZE, parts[4]/config.IMAGE_SIZE
                boxes.append([x1, y1, x2, y2])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return img_tensor, boxes, labels

def collate_fn(batch):
    """
    Кастомная функция сборки батча.

    Проблема: в батче переменное число объектов на каждом изображении,
    поэтому стандартный collate не работает.

    Args:
        batch: список кортежей (img, boxes, labels)
    Returns:
        images: [B, 1, H, W] — стек изображений
        targets: list из B словарей {'boxes': ..., 'labels': ...}
    """
    images = []
    targets = []
    for img, boxes, labels in batch:
        images.append(img)
        targets.append({'boxes': boxes, 'labels': labels})
    return torch.stack(images, 0), targets


class CVS(Sampler):
    def __init__(self, ds_idx, v_sz, valid=False):
        self.valid = valid
        if not self.valid:
            self.ds_idx = ds_idx
            self.v_sz = v_sz
        self.split_shuffle()

    def __len__(self):
        return len(self.ds_idx) - self.bord

    def split_shuffle(self):
        print('cross-v shuffle')
        if not self.valid:
            self.bord = int(len(self.ds_idx) / self.v_sz)
            np.random.shuffle(self.ds_idx)
            self.res_idx = self.ds_idx[self.bord:]
            val_idx = self.ds_idx[:self.bord]
            np.save(r'./tmp/tmp_val_idx.npy', val_idx)
        else:
            self.res_idx = np.load(r'./tmp/tmp_val_idx.npy')

    def __iter__(self):
        yield from self.res_idx

if __name__ == '__main__':
    import torchvision.transforms.functional as TF

    folder = r'test_val'
    ds = RailMarkDset(image_folder=folder, label_folder=folder, transform=train_transform)
    batch = ds.__getitem__(0)

    # print(batch[0].shape)
    # print(torch.unique(batch[0]))
    # print(*batch)

    img = TF.to_pil_image(batch[0])
    img.show()
