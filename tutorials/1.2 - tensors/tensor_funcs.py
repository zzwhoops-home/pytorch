import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# trying functions
print(torch.is_tensor(data))
print(torch.is_tensor(x_data))

print("\n")

# splicing
shape = (4, 4,)
# tensor = torch.ones(4, 4)
tensor = torch.ones(size=shape)
print(tensor)
tensor[:, 1] = 0
tensor[1, :] = 2
print(tensor)

print("\n")

# concatenation
# interesting that dim = -2 = 0, dim = -1 = 1
# it works, if dim is negative, total_dims + dim --> 2 + (-2) = 0, 2 + (-1) = 1
t1 = torch.cat([tensor, tensor, tensor], dim=-1)
print(t1)

print("\n")

# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

print("\n")

# matrix multiplication
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# same syntax, .T is obviously easier
print(tensor.T == torch.transpose(tensor, 0, 1))

print("\n")

# in-place operations have _ suffix
print(tensor, "\n")
print(tensor.add(5))
print("Still original\n")
print(tensor)
print("No longer original:")
tensor.add_(5)
print(tensor)

print("\n")

# integration with numpy
t = torch.ones(5) # remember this is float32 by default
# t = torch.ones(5, dtype=torch.float64)
print(f"t: {t}")
print(f"type t: {type(t)}")
n = t.numpy()
print(f"n: {n}")
print(f"type n: {type(n)}")

# bridged with numpy: tensor and numpy share memory locations
t.add_(1)
print(f"n: {n}")

print("\n")

# other way around: still sharing memory locations
# interesting that numpy uses float64s
n = np.ones(5)
t = torch.from_numpy(n)

print(f"t: {t}")
print(f"n: {n}")

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")