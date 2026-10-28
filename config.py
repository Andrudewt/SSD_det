from datetime import datetime
import os

# =============================================================================
# 1. CONFIGURATION for SSD 640x640
# =============================================================================


# model
MODEL_NAME = f"SSDmn_rmark"

# Image
IMAGE_SIZE = 640
NUM_CHANNELS = 1  # Grayscale

# Classes
# класс 0 зарезервирован логикой под background
class_dict = {1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7',
            9: '8', 10: '9', 11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E',
            16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'J', 21: 'K', 22: 'L',
            23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R', 29: 'S',
            30: 'T', 31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y'}

NUM_CLASSES = len(class_dict) + 1  # 36 (35 реальных + background)
CLASS_NAMES = ['background'] + list(class_dict.values())
BG_CLASS_ID = 0

# Feature map sizes for SSD300-like architecture with MobileNet
# размеры карт признаков, с которых снимаются предсказания
FEATURE_MAPS = [40, 20, 10, 5, 3, 2]

# Anchor boxes configuration
MIN_SIZES = [0.10, 0.20, 0.37, 0.54, 0.71, 0.88]
MAX_SIZES = [0.20, 0.37, 0.54, 0.71, 0.88, 1.05]
ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
VARIANCE = [0.1, 0.2]

# Training
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 100
NEG_POS_RATIO = 3  # Hard negative mining ratio

# Inference
CONF_THRESH = 0.01
NMS_THRESH = 0.45   # Non-Maximum Suppression фильтрация боксов по их confidence
TOP_K = 200
