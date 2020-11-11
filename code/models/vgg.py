#POCOVID-Net model in pytorch

import torch
import torch.nn as nn
import torchvision.models as models

class VGG16_model(object):
    """
    A network with the following architecture:
   
        VGG-16 base model (layers frozen)
        hidden layer of 64 neurons
        ReLU
        dropout(0.5)
        batch normalization
        softmax (output)
        
    """
    def __init__(self,
                 input_size: tuple = (224, 224, 3),
                 hidden_size: int = 64,
                 dropout: float = 0.5,
                 num_classes: int = 3,
                 **kwargs
                ):
        """
        Initialize a new network
        
        Inputs: 
        - input_size: Tuple, size of input data
        - hidden_size: Integer, number of units to use in hidden layer
        - dropout: Float, dropout coefficient
        - num_classes: Integer, number of classes
        """
        # load the VGG16 network
        model = models.vgg16(pretrained=True)

        # freeze weights of base model
        for param in model.parameters():
            param.requires_grad = False
    
        # construct the head of the model 
        model.classifier[6] = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(hidden_size,hidden_size),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax()
            )

        return model