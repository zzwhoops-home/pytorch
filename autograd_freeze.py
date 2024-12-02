# you can use 'frozen parameters' to increase computation speed
# useful during fine tuning: only modify classifier layers to make predictions, keep the rest of the model the same
import torch
from torch import nn, optim
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

# example: finetune model on new dataset with 10 labels - just replace the classifier (last in this case) layer
model.fc = nn.Linear(512, 10)

# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)