import torch
import torch.optim as optim
from inner_logic import Detector, AnchorBoxGenerator, match
from loss import MultiBoxLoss
from utils import EarlyStopping
import config
from tqdm import tqdm, trange
import csv
import os

# =============================================================================
# 9. ОБУЧАЮЩИЙ ЦИКЛ
# =============================================================================


class SSDTrainer:
    """
    Класс-обёртка для обучения SSD.

    Отвечает за:
        - Подготовку целевых тензоров (matching GT с якорями)
        - Вычисление loss
        - Оптимизацию (SGD с momentum)
    """
    def __init__(self, model, cfg=config):
        self.cfg = cfg
        self.model = model
        self.detector = Detector(cfg)
        self.anchor_gen = AnchorBoxGenerator(cfg)
        self.priors = self.anchor_gen.generate()
        self.criterion = MultiBoxLoss(cfg)
        self.name_ext = config.datetime.now().strftime('%Y%m%d_%H%M')
        self.MODEL_DIR = os.path.join('models', config.MODEL_NAME, f"{config.MODEL_NAME}_{self.name_ext}")
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        # SGD — стандартный оптимизатор для SSD (как в оригинальной статье)
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY
        )

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.priors = self.priors.cuda()

    def prepare_targets(self, targets):
        """
        Подготавливает целевые тензоры для всего батча.

        Для каждого изображения в батче вызывает match(), который:
            - Считает IoU между GT и всеми 8732 якорями
            - Назначает каждому якорю класс (0 = background, >0 = object)
            - Кодирует смещения GT относительно якорей

        Args:
            targets: list из B словарей {'boxes': [N,4], 'labels': [N]}
                     boxes в нормализованных координатах [0,1]
        Returns:
            loc_t:  [B, 8732, 4] — целевые смещения
            conf_t: [B, 8732] — целевые метки классов
        """
        batch_size = len(targets)
        num_priors = self.priors.size(0)

        # Инициализируем нулями (background по умолчанию)
        loc_t = torch.zeros(batch_size, num_priors, 4)
        conf_t = torch.zeros(batch_size, num_priors, dtype=torch.long)

        if torch.cuda.is_available():
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        for idx in range(batch_size):
            truths = targets[idx]['boxes']   # [N, 4] в [0,1]
            labels = targets[idx]['labels']  # [N] (0-based)

            truths = truths.cuda()
            labels = labels.cuda()

            if truths.numel() == 0:
                continue  # Изображение без объектов — всё остаётся background

            # Защита: если truths пришёл как 1D (один бокс), добавляем размерность
            if truths.dim() == 1:
                truths = truths.unsqueeze(0)

            # Сопоставляем GT с якорями (threshold=0.5 — стандарт SSD)
            match(0.5, truths, self.priors, labels, loc_t, conf_t, idx)

        return loc_t, conf_t

    def train_epoch(self, dataloader):
        """
        Одна эпоха обучения.

        Returns:
            tuple: (средний loc_loss, средний conf_loss) за эпоху
        """
        self.model.train()
        total_loc_loss = 0
        total_conf_loss = 0

        for batch_idx, (images, targets) in enumerate(tqdm(dataloader)):
            if torch.cuda.is_available():
                images = images.cuda()

            # Подготовка целей: matching + encoding
            loc_t, conf_t = self.prepare_targets(targets)

            # === ДЕБАГ ===
            # num_pos = (conf_t > 0).sum().item()
            # num_bg  = (conf_t == 0).sum().item()
            # print(f"  Batch {batch_idx}: pos={num_pos}, bg={num_bg}, "
            #       f"max_label={conf_t.max().item()}, min_label={conf_t.min().item()}")

            # # Проверьте масштаб GT боксов
            # if batch_idx == 0 and len(targets[0]['boxes']) > 0:
            #     print(f"  Sample GT boxes: {targets[0]['boxes'][:3]}")  # должны быть ~[0.1-0.9]
            # =============

            # Forward pass
            loc_preds, conf_preds = self.model(images)

            # Вычисление loss
            loc_loss, conf_loss = self.criterion(loc_preds, conf_preds, loc_t, conf_t)
            loss = loc_loss + conf_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loc_loss += loc_loss.item()
            total_conf_loss += conf_loss.item()

        return total_loc_loss / len(dataloader), total_conf_loss / len(dataloader)

    def save_checkpoint(self, state, filename='checkpoint'):
        print('=> save checkpoint')
        torch.save(state, filename)

    def fit(self, dataloader, num_epochs=None):
        """
        Полный цикл обучения.

        Args:
            dataloader: DataLoader с Dataset
            num_epochs: число эпох (если None, используется cfg.NUM_EPOCHS)
        """
        if num_epochs is None:
            num_epochs = self.cfg.NUM_EPOCHS

        log_file = os.path.join(self.MODEL_DIR, "training_log.csv")
        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        for epoch in range(num_epochs):
            # периодическое сохранение
            if epoch > 9 and epoch % 10 == 0:
                checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
                self.save_checkpoint(checkpoint, os.path.join(self.MODEL_DIR, f"{epoch}_{config.MODEL_NAME}_{self.name_ext}.pt.tar"))

            loc_loss, conf_loss = self.train_epoch(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Loc Loss: {loc_loss:.4f} | Conf Loss: {conf_loss:.4f}")
            # запись в файл
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, loc_loss, conf_loss])

            # вызов EarlyStopping
            early_stopping((loc_loss + conf_loss))
            if early_stopping.early_stop:
                print("Early stopping activated.")
                self.save_checkpoint(checkpoint, os.path.join(self.MODEL_DIR, f"best(es)_{config.MODEL_NAME}_{self.name_ext}.pt.tar"))
                break

        checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        # самая последняя эпоха
        self.save_checkpoint(checkpoint, os.path.join(self.MODEL_DIR, f"last_{config.MODEL_NAME}_{self.name_ext}.pt.tar"))


if __name__ == '__main__':
    print()
