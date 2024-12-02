import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# single image, 3 channels, 64x64 pixels
data = torch.rand(1, 3, 64, 64)
# labels have size (1, 1000)
labels = torch.rand(1, 1000)

prediction = model(data) # forward pass

loss = (prediction - labels).sum()
loss.backward() # backward pass

# optimizer for gradient descent
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
print(optim.step())