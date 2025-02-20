import torch
import numpy as np

# tensors created directly from data, datatype automatically inferred
# can't have jagged matrices
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# create data "like" another set of data
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.double) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# just generates data (comma is stylistic choice, maybe easier to add elements)
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape) # by default, initializes as float32
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# tensor attributes
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")