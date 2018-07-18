import torch

# NumPy Bridge
# Converting a Torch Tensor to a NumPy array and vice versa is a breeze.
#
# The Torch Tensor and NumPy array will share their underlying memory locations, and changing one will change the other.

##### Converting a torch tensor to a Numpy Array
a = torch.ones(5)
print(a)
print(type(a))
print(a.shape)
print(a.size())

print('\n#########################\n')

b = a.numpy()                           # it is function of torch tensor
print(b)
print(type(b))
print(b.shape)
# print(b.size())      # wrong way

print('\n#########################\n')

# See how the numpy array changed in value :
a.add_(1)
print('a : ', a)
print('b : ', b)

print('\n#########################\n')

#####  Converting NumPy Array to Torch Tensor
# See how changing the np array changed the Torch Tensor automatically
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print('a : ', a, '\n', type(a), '\t', a.shape)
print('b : ', b, '\n', type(b), '\t', b.shape)

##### Note : All the Tensors on the CPU except a CharTensor support converting to NumPy and back.

print('\n#########################\n')





