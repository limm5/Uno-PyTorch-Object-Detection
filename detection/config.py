import torch

BATCH_SIZE = 32 
NUM_EPOCHS = 5
NUM_WORKERS = 0

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

RESIZE_TO = 208 # resize the image for training and transforms; default 416

# location to save model and plots
OUT_DIR = 'outputs'

# training images and XML files directory
TRAIN_DIR = '/content/train'
# validation images and XML files directory
VALID_DIR = '/content/valid'

# classes: 0 index is reserved for background
CLASSES = [
    '__background__', '11', '9', '13', '10', '6', '7', '0', '5', '4', '2', '14', 
    '8', '12', '1', '3'
]

FREEZE_BACKBONE = True
BACKBONE_TYPE = "mobilenetv3" # resnet50 / mobilenetv3

NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False