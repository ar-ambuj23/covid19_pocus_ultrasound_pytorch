#POCOVID-Net model in pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class resnet18_model(nn.Module):
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
        
        super(resnet18_model, self).__init__()
        
        # Use GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # load the resnet18 network
        self.model = models.resnet18(pretrained=True).to(device)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        # freeze weights of base model except last couple layers
        last_frozen = 56
        count = 0
        for param in self.model.parameters():
            count += 1
            if count < last_frozen:
                param.requires_grad = False
            
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self,x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x