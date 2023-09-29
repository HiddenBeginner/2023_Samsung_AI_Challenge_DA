"""
This script provides the functionality to create a Domain-Adversarial Neural Network (DANN) 
for the purpose of domain adaptation in semantic segmentation tasks. It allows the user to 
utilize different feature backbones and classifiers, including pre-trained models, and provides 
default implementations for both semantic and domain classification.
"""
from typing import Optional, Tuple, Union
from collections import OrderedDict
import math

from transformers import SegformerPreTrainedModel


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

    @property
    def module(self):
        """
        If the model is being used with DataParallel, 
        this property will return the actual model.
        """
        return self._modules.get('module', self)

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


class SegFormerDANN2(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN) for semantic segmentation and domain adaptation.
    """
    def __init__(self, 
                 segformer: nn.Module,
                 domain_classifier: Optional[nn.Module] = None, 
                 num_classes: Optional[int] = 13):
        super(SegFormerDANN2, self).__init__()
        self.grl = GradientReversalLayer()
        self.segformer = segformer
        self.encoder = self.segformer.segformer
        self.decoder = self.segformer.decode_head



        # Use the provided domain_classifier or default to the DefaultDomainClassifier
        hidden_sizes = [64, 128, 320, 512]
        decoder_hidden_size = 768
        num_encoder_blocks = len(hidden_sizes)
        classifier_dropout_prob = 0.1
        reshape_last_stage = True
        
        if domain_classifier is None:
            domain_classifier = SegformerDomainDiscriminator(
                hidden_sizes=hidden_sizes,
                decoder_hidden_size=decoder_hidden_size,
                num_encoder_blocks=num_encoder_blocks,
                classifier_dropout_prob=classifier_dropout_prob,
                reshape_last_stage=reshape_last_stage
                )

        self.domain_classifier = domain_classifier

    @property
    def module(self):
        """
        If the model is being used with DataParallel, 
        this property will return the actual model.
        """
        return self._modules.get('module', self)

    def forward(self, x: torch.Tensor, lamda: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        """
        x = self.encoder(pixel_values=x, output_hidden_states=True)
        #print('x',len(x))
        semantic_outputs = self.decoder(x[1])
        
        if lamda is not None:
            # Assuming you have a GradientReversalLayer named "grl" in your model
            x_discriminator = self.grl(x[1], lamda)
            domain_outputs = self.domain_classifier(x_discriminator)

            #print('x_d',len(x_discriminator))
        else:
            domain_outputs = self.domain_classifier(x[1])

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
        return x

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


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states
    
    



class SegformerDomainDiscriminator(nn.Module):
    def __init__(self,
                hidden_sizes: list,
                decoder_hidden_size: int,
                num_encoder_blocks: int,
                classifier_dropout_prob: float,
                reshape_last_stage: bool):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.decoder_hidden_size = decoder_hidden_size
        self.num_encoder_blocks = num_encoder_blocks
        self.classifier_dropout_prob = classifier_dropout_prob
        self.reshape_last_stage = reshape_last_stage
        
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i, hidden_size in enumerate(self.hidden_sizes):
            mlp = SegformerMLP(input_dim=hidden_size, out_dim=768)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=self.decoder_hidden_size * self.num_encoder_blocks,
            out_channels=self.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(self.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(self.classifier_dropout_prob)
        self.classifier = nn.Conv2d(self.decoder_hidden_size, 1, kernel_size=1)


    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = torch.sigmoid(self.classifier(hidden_states))

        return logits