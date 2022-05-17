import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import (
    FREEZE_BACKBONE, BACKBONE_TYPE
)

def create_model(num_classes):
    
    # load Faster RCNN pre-trained model

    if BACKBONE_TYPE == "resnet50":
        # trainable 0 - 5
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=FREEZE_BACKBONE)
    elif BACKBONE_TYPE == "mobilenetv3":        
        # trainable 0 - 6
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, trainable_backbone_layers=FREEZE_BACKBONE)
    else:
        raise Exception("Invalid BACKBONE_TYPE")        
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model


# if __name__ == '__main__':
#     create_model(2)