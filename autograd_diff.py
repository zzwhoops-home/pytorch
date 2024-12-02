import torch

# create tensors
# requires_grad tells autograd that "every operation on them should be tracked"
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# 3a^3 - b^2
Q = 3*a**3 - b**2