from pathlib import Path

data_dir = './sartorius-cell-instance-segmentation'
BATCH_SIZE = 2
NUM_EPOCHS = 5

TRAIN_CSV = f"{data_dir}/train.csv"
TRAIN_PATH = f"{data_dir}/train"
TEST_PATH = f"{data_dir}/test"
TRAIN_FILES = sorted(list(Path(TRAIN_PATH).rglob('*png')))


ROOT = Path(data_dir)
TRAIN_PATH_ = Path(TRAIN_PATH)

WIDTH = 704
HEIGHT = 520
# Threshold for mask length
TH = 40
BATCH_SIZE = 2
LR = 1e-3
WEIGHT_DECAY = 0.0005


# Normalize to resnet mean and std if True.
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]
IMAGE_RESIZE = (224, 224)