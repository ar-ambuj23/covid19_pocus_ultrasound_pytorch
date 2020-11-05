#POCOVID-Net model in pytorch

import torch
import torch.nn as nn
import torchvision.models as models

def get_VGG16_model(
    input_size: tuple = (224, 224, 3),
    hidden_size: int = 64,
    dropout: float = 0.5,
    num_classes: int = 3,
    log_softmax: bool = True,
    **kwargs
):
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