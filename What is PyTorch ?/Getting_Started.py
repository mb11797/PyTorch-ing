# What is PyTorch?
# It’s a Python based scientific computing package targeted at two sets of audiences:
#
# A replacement for NumPy to use the power of GPUs
# a deep learning research platform that provides maximum flexibility and speed
#
# Tensors
# Tensors are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.

from __future__ import print_function
import torch

# Construct a 5x3 matrix, uninitialized:
x = torch.empty(5,3)
print(type(x))
print(x.shape)
print(x)

# Construct a randomly initialized matrix:
x = torch.rand(5,3)
print(type(x))
print(x.shape)
print(x)

# Construct a matrix filled zeros and of dtype long:
x = torch.zeros(5,3, dtype=torch.long)
print(type(x))
print(x.shape)
print(x)

##### Construct a tensor directly from data :
x = torch.tensor([5.5, 3])
print(type(x))
print(x.shape)
print(x)

# Create a tensor based on an existing tensor. These methods will reuse the properties of the input tensor, e.g, dtype, unless new values are provided by the user.
x = x.new_ones(5, 3, dtype=torch.double)            # new_* methods take in sizes
print(type(x))
print(x.shape)
print(x)

x = torch.randn_like(x, dtype=torch.float)          # overwrite dtype!
print(type(x))
print(x.shape)
print(x)

# Get the sizes
print(x.size())

# Note : torch.Size is in fact a tuple,so it supports all tuple operations:

### Operations :
## Multiple syntaxes for operations :

# Addition : syntax 1
y = torch.rand(5,3)
print(type(y))
print(y.shape)
print(y.size())
print(x+y)

# Addition : Syntax 2
print(torch.add(x,y))

# Addition providing an output tensor as argument :
result = torch.empty(5,3)
torch.add(x,y, out=result)
print(result)

# using a third argument to assign values :
z = x + y
print(type(z))
print(z.shape)
print(z.size())
print('z : ', z)

# Addition in place :
# adds x to y
y.add_(x)
print(type(y))
print(y.shape)
print(y.size())
print(y)

# Note : Any operation that mutates a tensor in-place is post-fixed with an _.For example, x.copy_(y), x.t_(), will change x.

# You can use standard NumPy-like indexing with all bells and whistles!
print(x[:, 1])

# Resizing : If u want to resize/reshape tensor, u can use torch.view :
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8)        # the size -1 is inferred from other dimensions
print(type(x), type(y), type(z))
print(x.size(), y.size(), z.size())

# If u have a one element tensor, use .item() to get the value as a Python number
x = torch.rand(1)
print(x)
print(type(x))
x = torch.randn(1)
print(x)
print(type(x))

print(x.item())









