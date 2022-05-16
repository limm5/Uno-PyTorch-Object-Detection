import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import (
    FREEZE_BACKBONE, BACKBONE_TYPE
)

def create_model(num_classes):
    
    # load Faster RCNN pre-trained model

    if BACKBONE_TYPE == "resnet50":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif BACKBONE_TYPE == "mobilenetv3":        
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    else:
        raise Exception("Invalid BACKBONE_TYPE")
    
    if FREEZE_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False        
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model
