import torch

BATCH_SIZE = 8 
NUM_EPOCHS = 10
NUM_WORKERS = 0

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

RESIZE_TO = 416 # resize the image for training and transforms

# location to save model and plots
OUT_DIR = 'outputs'

# training images and XML files directory
TRAIN_DIR = '/Users/nelsonlim/Development/PyTorch Object Detection/data/Uno Cards.v1-v1.voc/train'
# validation images and XML files directory
VALID_DIR = '/Users/nelsonlim/Development/PyTorch Object Detection/data/Uno Cards.v1-v1.voc/valid'

# classes: 0 index is reserved for background
CLASSES = [
    '__background__', '11', '9', '13', '10', '6', '7', '0', '5', '4', '2', '14', 
    '8', '12', '1', '3'
]

NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False