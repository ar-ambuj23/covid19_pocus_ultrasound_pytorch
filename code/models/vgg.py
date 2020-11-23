#POCOVID-Net model in pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG16_model(nn.Module):
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
        
        super(VGG16_model, self).__init__()
        
        # load the VGG16 network
        self.model = models.vgg16(pretrained=True)

        # freeze weights of base model
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.avgpool = nn.AvgPool2d(4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1000, hidden_size)
        self.bn = nn.BatchNorm2d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self,x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.fc1(x)
        #x = self.bn(x) TODO figure out how batch norm is used here
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x