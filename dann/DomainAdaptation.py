"""
This script provides the functionality to create a Domain-Adversarial Neural Network (DANN) 
for the purpose of domain adaptation in semantic segmentation tasks. It allows the user to 
utilize different feature backbones and classifiers, including pre-trained models, and provides 
default implementations for both semantic and domain classification.
"""
from typing import Optional, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

#from backbones import DeepLabV3, FCN, LRASPP


def get_backbone(model: nn.Module) -> nn.Module:
    """
    Extract the backbone from the provided model.
    
    Args:
        model (torch.nn.Module): The model from which to extract the backbone.
        
    Returns:
        torch.nn.Module: The backbone of the provided model.
    """
    backbone = model.model.backbone
    return backbone


def get_classifier(model: nn.Module, aux: bool = False) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
    """
    Extract the semantic classifier from the provided model.
    
    Args:
        model (torch.nn.Module): The model from which to extract the classifier.
        aux (bool): Whether to also return the auxiliary classifier. Default is False.
        
    Returns:
        torch.nn.Module or (torch.nn.Module, torch.nn.Module): 
        The semantic classifier of the provided model or a tuple containing 
        the semantic classifier and the auxiliary classifier if aux is True.
    """
    semantic_classifier = model.model.classifier
    if aux:
        aux_classifier = model.model.aux_classifier
        return semantic_classifier, aux_classifier
    else:
        return semantic_classifier


class DefaultConvClassifier(nn.Module):
    """
    A default fully convolutional classifier for semantic segmentation.
    """
    def __init__(self, in_channels: int, num_classes: int):
        super(DefaultConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1)  # Final prediction per pixel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class DefaultDomainClassifier(nn.Module):
    """
    A default classifier for domain adaptation.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, use_dropout: bool = True, dropout_rate: float = 0.5):
        super(DefaultDomainClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the domain classifier.
        """
        return self.classifier(x)
    

class FullyConvolutionalDiscriminator(nn.Module):
    """
    Domain discriminator
    
        - References
            [1] DANNet: A One-Stage Domain Adaptation Network for Unsupervised Nighttime Semantic Segmentation
            [2]
    """ 
    def __init__(self, num_classes, ndf = 64):
        super(FullyConvolutionalDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf*4, 1, kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

class DANN(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN) for semantic segmentation and domain adaptation.
    """
    def __init__(self, 
                 feature_backbone: nn.Module, 
                 semantic_classifier: Optional[nn.Module] = None,
                 aux_classifier: Optional[nn.Module] = None,
                 domain_classifier: Optional[nn.Module] = None, 
                 num_classes: Optional[int] = 13):
        super(DANN, self).__init__()
        self.grl = GradientReversalLayer()
        self.feature_backbone = feature_backbone
        
        # Use the provided semantic_classifier or default to the DefaultConvClassifier
        if semantic_classifier is None:
            if num_classes is None:
                raise ValueError("num_classes must be provided if semantic_classifier is not given")
            num_channels = feature_backbone[list(feature_backbone.keys())[-1]].out_channels
            semantic_classifier = DefaultConvClassifier(num_channels, num_classes)
        self.semantic_classifier = semantic_classifier
        self.aux_classifier = aux_classifier

        # Use the provided domain_classifier or default to the DefaultDomainClassifier
        if domain_classifier is None:
            num_channels = feature_backbone[list(feature_backbone.keys())[-1]].out_channels
            domain_classifier = DefaultDomainClassifier(num_channels)
        self.domain_classifier = domain_classifier

    def forward(self, x: torch.Tensor, lamda: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DANN.
        
        Args:
            x (torch.Tensor): Input tensor.
            lamda (float, optional): Lambda parameter for gradient reversal.
            
        Returns:
            torch.Tensor: Semantic segmentation output.
            torch.Tensor: Domain classification output.
        """
        features = self.feature_backbone(x)
        if self.aux_classifier:
            main_outputs = self.semantic_classifier(features['out'])
            aux_outputs = self.aux_classifier(features['aux'])

            # Upsample the main outputs
            main_outputs = F.interpolate(main_outputs, size=x.shape[2:], mode='bilinear', align_corners=True)
            
            # Upsample the aux outputs
            aux_outputs = F.interpolate(aux_outputs, size=x.shape[2:], mode='bilinear', align_corners=True)
            
            semantic_outputs = OrderedDict({'out': main_outputs, 'aux': aux_outputs})
        else:
            semantic_outputs = self.semantic_classifier(features)
            # Upsample the semantic outputs
            semantic_outputs = F.interpolate(semantic_outputs, size=x.shape[2:], mode='bilinear', align_corners=True)

        # if type(features) == OrderedDict:
        #     features_for_discriminator = features['out']
        # else:
        #     features_for_discriminator = features
        
        if lamda is not None:
            # Assuming you have a GradientReversalLayer named "grl" in your model
            features_for_discriminator = self.grl(semantic_outputs['out'], lamda)
        else:
            features_for_discriminator = semantic_outputs['out']
        domain_outputs = self.domain_classifier(features_for_discriminator)

        return semantic_outputs, domain_outputs

    def freeze_bn(self):
        """Freezes the Batch Normalization layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                
class SegFormerDANN(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN) for semantic segmentation and domain adaptation.
    """
    def __init__(self, 
                 segformer: nn.Module,
                 domain_classifier: Optional[nn.Module] = None, 
                 num_classes: Optional[int] = 13):
        super(SegFormerDANN, self).__init__()
        self.grl = GradientReversalLayer()
        self.segformer = segformer

        # Use the provided domain_classifier or default to the DefaultDomainClassifier
        if domain_classifier is None:
            domain_classifier = FullyConvolutionalDiscriminator(num_classes=13)

        self.domain_classifier = domain_classifier

    def forward(self, x: torch.Tensor, lamda: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DANN.
        
        Args:
            x (torch.Tensor): Input tensor.
            lamda (float, optional): Lambda parameter for gradient reversal.
            
        Returns:
            torch.Tensor: Semantic segmentation output.
            torch.Tensor: Domain classification output.
        """
        semantic_outputs = self.segformer(x)[0]
        
        if lamda is not None:
            # Assuming you have a GradientReversalLayer named "grl" in your model
            features_for_discriminator = self.grl(semantic_outputs, lamda)
        else:
            features_for_discriminator = semantic_outputs
        domain_outputs = self.domain_classifier(features_for_discriminator)

        return semantic_outputs, domain_outputs

    def freeze_bn(self):
        """Freezes the Batch Normalization layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()



class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL)
    """

    @staticmethod
    def forward(ctx, x, lamda):
        """
        In the forward pass we receive a tensor containing the input and return
        a tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return -ctx.lamda * grad_output, None  # Multiply the gradient by -lambda

class GradientReversalLayer(nn.Module):
    """
    Wrapper module for the GradientReversalFunction
    """
    def forward(self, x, lamda=1.0):
        return GradientReversalFunction.apply(x, lamda)
