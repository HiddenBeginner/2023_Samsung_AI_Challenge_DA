"""
This script contains the definition of several classes used to build and run a 
Domain-Adversarial Neural Network (DANN) that is adapted for semantic segmentation 
tasks using the SegFormer architecture.

Classes:
    - DefaultConvClassifier: A convolutional classifier for semantic segmentation tasks.
    - DefaultDomainClassifier: A domain classifier used for domain adaptation tasks.
    - FullyConvolutionalDiscriminator: A domain discriminator with a fully convolutional architecture.
    - SegFormerDANN: Prototype SegFormerDANN architecture (Not used).
    - SegFormerDANN2: Main class that integrates SegFormer with DANN.
    - GradientReversalFunction: Autograd function defining forward and backward passes for gradient reversal.
    - GradientReversalLayer: A wrapper module to apply the GradientReversalFunction.
    - SegformerMLP: A simple MLP for linear embedding.
    - SegformerDomainDiscriminator: A domain discriminator designed for SegFormerDANN.

Each class is equipped with necessary methods for their specific operations and 
tasks, including initialization, forward pass, and other utility functions.

References)
    [1] Xie, Enze, et al. "SegFormer: Simple and efficient design for semantic segmentation with transformers." Advances in Neural Information Processing Systems 34 (2021): 12077-12090.
    [2] Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks." The journal of machine learning research 17.1 (2016): 2096-2030.
    [3] Wu, Xinyi, et al. "Dannet: A one-stage domain adaptation network for unsupervised nighttime semantic segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
"""

import math
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerPreTrainedModel


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
    FullyConvolutionalDiscriminator is a PyTorch module that serves as a domain 
    discriminator[3] in a Domain-Adversarial Neural Network (DANN)[2] for semantic segmentation.
    
    The module employs a series of 2D convolutional layers to process the input tensor, 
    followed by leaky ReLU activation functions. The final layer acts as a classifier 
    to distinguish between source and target domain features.
    
    Attributes:
    -----------
    conv1 (nn.Conv2d): First convolutional layer with 'num_classes' input channels and 
                       'ndf' output channels, using a 4x4 kernel, stride of 2, and padding of 1.
    conv2 (nn.Conv2d): Second convolutional layer with 'ndf' input channels and 
                       'ndf*2' output channels, using a 4x4 kernel, stride of 2, and padding of 1.
    conv3 (nn.Conv2d): Third convolutional layer with 'ndf*2' input channels and 
                       'ndf*4' output channels, using a 4x4 kernel, stride of 1, and padding of 1.
    conv4 (nn.Conv2d): Fourth convolutional layer with 'ndf*4' input channels and 
                       'ndf*4' output channels, using a 4x4 kernel, stride of 1, and padding of 1.
    classifier (nn.Conv2d): Classifier convolutional layer with 'ndf*4' input channels and 
                            1 output channel, using a 4x4 kernel, stride of 1, and padding of 1.
    leaky_relu (nn.LeakyReLU): Leaky ReLU activation function with a negative slope of 0.2.
    
    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor:
        Conducts the forward pass through the module, taking an input tensor 'x', 
        and returning the output tensor after the series of convolutional layers and activations.
    
    Parameters:
    -----------
    num_classes : int
        Number of classes in the segmentation task, which defines the number of input channels.
    ndf : int, optional, default=64
        Number of filters in the first convolutional layer, which defines the depth of the feature maps.
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

                
class SegFormerDANN(nn.Module):
    """
    A prototype of Domain-Adversarial Neural Network (DANN) for semantic segmentation 
    and domain adaptation with SegFormer[1] as the base model. This model is designed to 
    enhance the performance of semantic segmentation tasks when there is a domain 
    shift between the source and target domains.

    Attributes:
    -----------
    grl (GradientReversalLayer):
        Instance of the GradientReversalLayer, which inverses the gradients flowing through it 
        multiplied by a specified factor, facilitating the domain adaptation process.
        
    segformer (nn.Module):
        SegFormer model responsible for semantic segmentation.
        
    domain_classifier (nn.Module):
        Classifier module used to distinguish between the source and target domain features.
        It is trained adversarially against the segformer to improve domain invariance of features.
        
    Methods:
    --------
    forward(x: torch.Tensor, lamda: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        Defines the forward pass for the DANN. It first computes the semantic segmentation 
        output, then applies the gradient reversal layer, and finally computes the domain 
        classification output.
        
    freeze_bn():
        Freezes the Batch Normalization layers in the model. Useful during the fine-tuning process.
        
    Parameters:
    -----------
    segformer : nn.Module
        Pre-trained SegFormer model.
        
    domain_classifier : nn.Module, optional, default=None
        Domain classifier module. If None, defaults to using FullyConvolutionalDiscriminator with 13 classes.
        
    num_classes : int, optional, default=13
        Number of classes for the segmentation task.
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
    SegFormerDANN2 is an enhanced version of the prototype SegFormerDANN model, 
    designed for domain-adversarial training for semantic segmentation tasks. 
    The model consists of a SegFormer encoder, a decoder, and a domain classifier, 
    aiming to minimize the distribution discrepancy between the source and target domain.

    Attributes:
    -----------
    grl : GradientReversalLayer
        Instance of the GradientReversalLayer.
        
    segformer : nn.Module
        The base SegFormer module, which is utilized for the main task of semantic segmentation.
        
    encoder : nn.Module
        The encoder module extracted from the provided SegFormer.
        
    decoder : nn.Module
        The decoder module extracted from the provided SegFormer.
        
    domain_classifier : nn.Module
        A classifier network responsible for performing domain classification (source vs. target).

    Methods:
    --------
    forward(x: torch.Tensor, lamda: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        Forward pass of the model. Computes the semantic segmentation and domain classification.
        
    freeze_bn():
        Freezes Batch Normalization layers across the model.

    Parameters:
    -----------
    segformer : nn.Module
        Pre-trained SegFormer model for semantic segmentation.
        
    domain_classifier : Optional[nn.Module] (default=None)
        Optional custom domain classifier. If None, initializes a default domain classifier.
        
    num_classes : Optional[int] (default=13)
        Number of classes for the segmentation task.
    """
    def __init__(self, 
                 segformer: nn.Module,
                 domain_classifier: Optional[nn.Module] = None, 
                 num_classes: Optional[int] = 13):
        super(SegFormerDANN2, self).__init__()
        self.grl = GradientReversalLayer()  # Instantiate Gradient Reversal Layer
        self.segformer = segformer  # Base SegFormer module for semantic segmentation
        self.encoder = self.segformer.segformer  # Encoder extracted from base SegFormer
        self.decoder = self.segformer.decode_head  # Decoder extracted from base SegFormer




        # Set domain classifier, either provided or default to SegformerDomainDiscriminator
        if domain_classifier is None:
            # Define domain classifier parameters
            hidden_sizes = [64, 128, 320, 512]
            decoder_hidden_size = 768
            num_encoder_blocks = len(hidden_sizes)
            classifier_dropout_prob = 0.1
            reshape_last_stage = True
            
            # Instantiate SegformerDomainDiscriminator with specified parameters
            domain_classifier = SegformerDomainDiscriminator(
                hidden_sizes=hidden_sizes,
                decoder_hidden_size=decoder_hidden_size,
                num_encoder_blocks=num_encoder_blocks,
                classifier_dropout_prob=classifier_dropout_prob,
                reshape_last_stage=reshape_last_stage
            )
        self.domain_classifier = domain_classifier  # Set domain classifier

    @property
    def module(self):
        """
        If the model is being used with DataParallel, 
        this property will return the actual model.
        """
        return self._modules.get('module', self)

    def forward(self, x: torch.Tensor, lamda: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of SegFormerDANN2.
        
        1. Encode the input using the SegFormer encoder.
        2. Decode the encoded features for semantic segmentation.
        3. Apply gradient reversal (if lambda is provided) and classify domain using domain_classifier.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.
        lamda : float, optional
            Lambda factor for the Gradient Reversal Layer. When provided, GRL is applied.
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing semantic segmentation output and domain classification output.
        """
        # Encode input, retrieve hidden states
        x = self.encoder(pixel_values=x, output_hidden_states=True)
        semantic_outputs = self.decoder(x[1])   # Decode for semantic segmentation
        
        # Apply Gradient Reversal Layer if lambda is provided, then classify domain
        if lamda is not None:
            # Assuming you have a GradientReversalLayer named "grl" in your model
            x_discriminator = self.grl(x[1], lamda)
            domain_outputs = self.domain_classifier(x_discriminator)
        else:
            domain_outputs = self.domain_classifier(x[1])

        return semantic_outputs, domain_outputs
    
    
    def freeze_bn(self):
        """
        Freezes all Batch Normalization layers in the model to maintain the statistics of 
        the source domain during the domain adaptation process.
        """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer Function (GRL Function)[2].
    
    The GRL Function is designed to reverse the gradient during backpropagation,
    with an optional scaling factor lambda. It is primarily used for domain
    adaptation problems where the model learns to perform well on both source
    and target domains by aligning the feature distributions.
    
    Methods:
    --------
    forward(ctx, x, lamda):
        Forward pass of the GRL Function. It returns the input as it is but
        saves the lambda value for the backward pass.
        
    backward(ctx, grad_output):
        Backward pass of the GRL Function. It returns the negative of the
        gradient multiplied by the saved lambda value.
    """
    @staticmethod
    def forward(ctx, x, lamda):
        """
        Forward pass for the GRL Function.

        Parameters:
        -----------
        ctx : torch.autograd.Context
            Context object that can be used to stash information
            for backward computation.
            
        x : torch.Tensor
            Input tensor.
            
        lamda : float
            Lambda is a scaling factor that modulates the magnitude of the
            gradient reversal.

        Returns:
        --------
        torch.Tensor
            The input tensor x is returned as output without any change.
        """
        ctx.lamda = lamda
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the GRL Function.

        Parameters:
        -----------
        ctx : torch.autograd.Context
            Context object with saved lambda.
            
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns:
        --------
        Tuple[torch.Tensor, None]
            Negative of the gradient multiplied by lambda, and None.
        """
        return -ctx.lamda * grad_output, None  # Multiply the gradient by -lambda


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) [2].
    
    The GRL is a wrapper module around the GRL Function. It's used in training
    models for domain adaptation by reversing the gradient during backpropagation,
    making the model invariant to the domain shift between source and target domains.

    Methods:
    --------
    forward(x, lamda):
        Forward pass of the GRL. Calls the GRL Function with input x and lambda.
        
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor to the GRL.
        
    lamda : float, default=1.0
        Lambda is a scaling factor that modulates the magnitude of the
        gradient reversal.

    Returns:
    --------
    torch.Tensor
        Output tensor after applying the GRL Function.
    """
    def forward(self, x, lamda=1.0):
        """
        Forward pass of the Gradient Reversal Layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.
            
        lamda : float, optional, default=1.0
            Scaling factor for the magnitude of gradient reversal.

        Returns:
        --------
        torch.Tensor
            Tensor after applying gradient reversal.
        """
        return GradientReversalFunction.apply(x, lamda)


class SegformerMLP(nn.Module):
    """
    SegformerMLP (Linear Embedding Module).
    
    This module applies a linear transformation to the input data, effectively 
    projecting the input feature vectors into a specified output dimension.

    Attributes:
    -----------
    proj : nn.Linear
        Linear layer that projects the input feature vectors into the output dimension.
        
    Methods:
    --------
    forward(hidden_states: torch.Tensor):
        Forward pass for the SegformerMLP.

    Parameters:
    -----------
    input_dim : int
        Dimensionality of the input feature vectors.
        
    out_dim : int
        Desired dimensionality of the output feature vectors.
    """

    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor):
        """
        Forward pass for SegformerMLP.

        Parameters:
        -----------
        hidden_states : torch.Tensor
            Input tensor of feature vectors.

        Returns:
        --------
        torch.Tensor
            Output tensor of projected feature vectors.
        """
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states
    

class SegformerDomainDiscriminator(nn.Module):
    """
    SegformerDomainDiscriminator (Domain Discriminator for Segformer).
    
    This module serves as the domain discriminator in the Domain-Adversarial Training
    of Segformers, designed to distinguish between source and target domain features.

    Attributes:
    -----------
    hidden_sizes : list
        List of hidden sizes of the encoder's output at each stage.
        
    decoder_hidden_size : int
        The hidden size of the decoder.
        
    num_encoder_blocks : int
        Number of encoder blocks.
        
    classifier_dropout_prob : float
        Dropout probability for the classifier.
        
    reshape_last_stage : bool
        Flag to reshape the last stage.
        
    linear_c : nn.ModuleList
        List of SegformerMLP modules to unify channel dimension of encoder blocks.
        
    linear_fuse : nn.Conv2d
        Convolutional layer to fuse the encoder block features.
        
    batch_norm : nn.BatchNorm2d
        Batch Normalization layer.
        
    activation : nn.ReLU
        Activation function.
        
    dropout : nn.Dropout
        Dropout layer.
        
    classifier : nn.Conv2d
        Classifier Convolutional layer.

    Methods:
    --------
    forward(encoder_hidden_states: torch.FloatTensor):
        Forward pass for the SegformerDomainDiscriminator.

    Parameters:
    -----------
    hidden_sizes : list
        List of hidden sizes for each encoder block.
        
    decoder_hidden_size : int
        Hidden size for the decoder.
        
    num_encoder_blocks : int
        Number of encoder blocks.
        
    classifier_dropout_prob : float
        Dropout probability for the classifier.
        
    reshape_last_stage : bool
        Whether to reshape the last stage.
    """

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
        
        # Initialize list of SegformerMLP modules
        mlps = []
        for i, hidden_size in enumerate(self.hidden_sizes):
            mlp = SegformerMLP(input_dim=hidden_size, out_dim=768)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # Convolutional layer to fuse the features from different encoder blocks
        self.linear_fuse = nn.Conv2d(
            in_channels=self.decoder_hidden_size * self.num_encoder_blocks,
            out_channels=self.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )

        # Batch Normalization, Activation, Dropout, and Classifier layers
        self.batch_norm = nn.BatchNorm2d(self.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(self.classifier_dropout_prob)
        self.classifier = nn.Conv2d(self.decoder_hidden_size, 1, kernel_size=1)


    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass for SegformerDomainDiscriminator.

        Parameters:
        -----------
        encoder_hidden_states : torch.FloatTensor
            List of tensors representing encoder hidden states for each block.

        Returns:
        --------
        torch.Tensor
            Output tensor of logits after domain classification.
        """
        batch_size = encoder_hidden_states[-1].shape[0]

        # Process and unify channel dimension of each encoder block
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