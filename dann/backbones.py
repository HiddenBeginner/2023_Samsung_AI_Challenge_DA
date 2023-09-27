"""
Segmentation Model Customization Script

This script provides customizable versions of popular segmentation models such as DeepLabV3, FCN, and LRASPP.
Each model allows for:
- Choice of backbone (e.g., 'mobilenet', 'resnet50', 'resnet101' depending on the model).
- Option to initialize with pretrained weights.
- Flexibility in choosing which layers are trainable.

Usage:
1. Import the desired model class: 
   from <script_name> import DeepLabV3, FCN, LRASPP

2. Create a model instance with desired configurations:
   model = DeepLabV3(backbone='resnet50', pretrained=True, mode='all')

3. Use the model for training or inference as usual.

Note: 
Each model returns outputs in the form of an OrderedDict. The keys in the dictionary 
represent different output heads, such as 'out' and 'aux'. Users should primarily 
focus on the 'out' key during inference or evaluation.

"""

import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights, ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
from torchvision.models.segmentation import FCN_ResNet50_Weights, FCN_ResNet101_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights

NUMBER_OF_CLASSES = 13

class DeepLabV3(nn.Module):
    """
    Custom DeepLabV3 model with flexible backbone, pretrained weights, and trainable layers.
    
    The model produces outputs in the form of an OrderedDict with two keys: 'out' and 'aux'. 
    - 'out': Represents the primary segmentation output of the model.
    - 'aux': Represents the auxiliary segmentation output from an intermediate layer, 
             primarily used for regularization during training.
    
    Users should primarily focus on the 'out' key during inference or evaluation.

    Parameters:
    - backbone (str): The backbone model to use ('mobilenet', 'resnet50', 'resnet101').
                      Determines the architecture of the model.
    - pretrained (bool): Whether to initialize the model with pretrained weights.
                        If True, the model will be initialized with weights pre-trained on COCO dataset.
    - mode (str): Training mode ('last', 'classifier', 'all').
                  Determines which layers of the model should be trainable.
        - 'last': Only the last layer is trainable.
        - 'classifier': Only the classifier is trainable.
        - 'all': All layers are trainable.

    Attributes:
    - model (nn.Module): The customized DeepLabV3 model.
    """
    
    def __init__(self, backbone: str = 'mobilenet', 
                       pretrained: bool = True,
                       mode: str = 'last'):
        super(DeepLabV3, self).__init__()

        # Check if the provided backbone is valid
        valid_backbones = ['mobilenet', 'resnet50', 'resnet101']
        if backbone not in valid_backbones:
            raise ValueError(f"'backbone' should be one of {valid_backbones}. Got '{backbone}' instead.")

        # Determine the model function and weights based on the provided backbone
        if backbone == 'mobilenet':
            model_func = deeplabv3_mobilenet_v3_large
            weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
            weights_backbone = MobileNet_V3_Large_Weights.DEFAULT
        elif backbone == 'resnet50':
            model_func = deeplabv3_resnet50
            weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
            weights_backbone = ResNet50_Weights.DEFAULT
        else:
            model_func = deeplabv3_resnet101
            weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
            weights_backbone = ResNet101_Weights.DEFAULT
        
        
        self.model = model_func(weights=weights, weights_backbone=weights_backbone)
        
        # Update the classifier layers for custom number of classes
        self.model.classifier[4] = nn.Conv2d(256, NUMBER_OF_CLASSES, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(10, NUMBER_OF_CLASSES, kernel_size=(1, 1), stride=(1, 1))

        # Set all layers to non-trainable by default
        for param in self.model.parameters():
            param.requires_grad = False

        # Update layers to be trainable based on the provided mode
        if mode == 'all':
            for param in self.model.parameters():
                param.requires_grad = True
        elif mode == 'classifier':
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            for param in self.model.aux_classifier.parameters():
                param.requires_grad = True
        elif mode == 'last':
            for param in self.model.classifier[4].parameters():
                param.requires_grad = True
            for param in self.model.aux_classifier[4].parameters():
                param.requires_grad = True

    def freeze_bn(self):
        """Freezes the Batch Normalization layers."""
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    
    def forward(self, x):
        """
        Forward pass of the DeepLabV3 model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - OrderedDict: Contains two keys - 'out' for the primary output and 'aux' for the auxiliary output.
        """
        return self.model(x)


class FCN(nn.Module):
    """
    Custom FCN(Fully Convolutional Networks) model with flexible backbone, pretrained weights, and trainable layers.
    
    The model produces outputs in the form of an OrderedDict with two keys: 'out' and 'aux'. 
    - 'out': Represents the primary segmentation output of the model.
    - 'aux': Represents the auxiliary segmentation output from an intermediate layer, 
             primarily used for regularization during training.
    
    Users should primarily focus on the 'out' key during inference or evaluation.

    Parameters:
    - backbone (str): The backbone model to use ('resnet50', 'resnet101').
                      Determines the architecture of the model.
    - pretrained (bool): Whether to initialize the model with pretrained weights.
                        If True, the model will be initialized with weights pre-trained on COCO dataset.
    - mode (str): Training mode ('last', 'classifier', 'all').
                  Determines which layers of the model should be trainable.
        - 'last': Only the last layer is trainable.
        - 'classifier': Only the classifier is trainable.
        - 'all': All layers are trainable.

    Attributes:
    - model (nn.Module): The customized DeepLabV3 model.
    """
    
    def __init__(self, backbone: str = 'resnet101', 
                       pretrained: bool = True,
                       mode: str = 'last'):
        super(FCN, self).__init__()

        # Check if the provided backbone is valid
        valid_backbones = ['resnet50', 'resnet101']
        if backbone not in valid_backbones:
            raise ValueError(f"'backbone' should be one of {valid_backbones}. Got '{backbone}' instead.")

        # Determine the model function and weights based on the provided backbone
        if backbone == 'resnet50':
            model_func = fcn_resnet50
            weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
            weights_backbone = ResNet50_Weights.DEFAULT
        else:
            model_func = fcn_resnet101
            weights = FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
            weights_backbone = ResNet101_Weights.DEFAULT
        
        
        self.model = model_func(weights=weights, weights_backbone=weights_backbone)
        
        # Update the classifier layers for custom number of classes
        self.model.classifier[4] = nn.Conv2d(512, NUMBER_OF_CLASSES, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(256, NUMBER_OF_CLASSES, kernel_size=(1, 1), stride=(1, 1))

        # Set all layers to non-trainable by default
        for param in self.model.parameters():
            param.requires_grad = False

        # Update layers to be trainable based on the provided mode
        if mode == 'all':
            for param in self.model.parameters():
                param.requires_grad = True
        elif mode == 'classifier':
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            for param in self.model.aux_classifier.parameters():
                param.requires_grad = True
        elif mode == 'last':
            for param in self.model.classifier[4].parameters():
                param.requires_grad = True
            for param in self.model.aux_classifier[4].parameters():
                param.requires_grad = True

    def freeze_bn(self):
        """Freezes the Batch Normalization layers."""
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    
    def forward(self, x):
        """
        Forward pass of the DeepLabV3 model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - OrderedDict: Contains two keys - 'out' for the primary output and 'aux' for the auxiliary output.
        """
        return self.model(x)
    
    
class LRASPP(nn.Module):
    """
    Custom LRASPP (Lite R-ASPP) model with flexible backbone, pretrained weights, and trainable layers.
    
    The model produces outputs in the form of an OrderedDict with a key: 'out'. 
    - 'out': Represents the primary segmentation output of the model.
    
    Users should primarily focus on the 'out' key during inference or evaluation.

    Parameters:
    - backbone (str): The backbone model to use ('resnet50', 'resnet101').
                      Determines the architecture of the model.
    - pretrained (bool): Whether to initialize the model with pretrained weights.
                        If True, the model will be initialized with weights pre-trained on COCO dataset.
    - mode (str): Training mode ('last', 'classifier', 'all').
                  Determines which layers of the model should be trainable.
        - 'last': Only the last layer is trainable.
        - 'classifier': Only the classifier is trainable.
        - 'all': All layers are trainable.

    Attributes:
    - model (nn.Module): The customized DeepLabV3 model.
    """
    
    def __init__(self, backbone: str = 'mobilenet', 
                       pretrained: bool = True,
                       mode: str = 'last'):
        super(LRASPP, self).__init__()

        # Check if the provided backbone is valid
        valid_backbones = ['mobilenet']
        if backbone not in valid_backbones:
            raise ValueError(f"'backbone' should be one of {valid_backbones}. Got '{backbone}' instead.")

        # Determine the model function and weights based on the provided backbone
        if backbone == 'mobilenet':
            model_func = lraspp_mobilenet_v3_large
            weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
            weights_backbone = MobileNet_V3_Large_Weights.DEFAULT
        
        self.model = model_func(weights=weights, weights_backbone=weights_backbone)
        
        # Update the classifier layers for custom number of classes
        self.model.classifier.low_classifier = nn.Conv2d(40, NUMBER_OF_CLASSES, kernel_size=(1, 1), stride=(1, 1))
        self.model.classifier.high_classifier = nn.Conv2d(128, NUMBER_OF_CLASSES, kernel_size=(1, 1), stride=(1, 1))

        # Set all layers to non-trainable by default
        for param in self.model.parameters():
            param.requires_grad = False

        # Update layers to be trainable based on the provided mode
        if mode == 'all':
            for param in self.model.parameters():
                param.requires_grad = True
        elif mode == 'classifier':
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            for param in self.model.aux_classifier.parameters():
                param.requires_grad = True
        elif mode == 'last':
            for param in self.model.classifier.low_classifier.parameters():
                param.requires_grad = True
            for param in self.model.classifier.high_classifier.parameters():
                param.requires_grad = True

    def freeze_bn(self):
        """Freezes the Batch Normalization layers."""
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        """
        Forward pass of the DeepLabV3 model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - OrderedDict: Contains two keys - 'out' for the primary output.
        """
        return self.model(x)
    
    