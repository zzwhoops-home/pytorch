import torch

# create tensors
# requires_grad tells autograd that "every operation on them should be tracked"
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# 3a^3 - b^2
Q = 3*a**3 - b**2

Q.sum().backward()
print(a.grad) # 36. 81.
print(b.grad) # -12. -8.

# same thing as above
a.grad.zero_()
b.grad.zero_()
Q = 3*a**3 - b**2

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

print(a.grad)
print(b.grad)
print(9*a**2)
print(9*a**2 == a.grad)
print(-2*b == b.grad)

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

print("\n")

# only one tensor requires grad for both to require it
a = x + y
print(f"Does `a` require gradients?: {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

print(a)
print(b)

print("\n")