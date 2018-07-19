# Neural Networks can be constructed using the torch.nn package.

# Now that u had a glimpse of autograd, nn depends on autograd to define models and differentiate them
# An nn.Module contains layers and a method frward(input) that returns the output:

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernels
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation : y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        print('x : ', x.size())
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print('x : ', x.size())
        x = x.view(-1, self.num_flat_features(x))
        print('x : ', x.size())
        x = F.relu(self.fc1(x))
        print('x : ', x.size())
        x = F.relu(self.fc2(x))
        print('x : ', x.size())
        x = self.fc3(x)
        print('x : ', x.size())
        # x = self.conv1(x)
        # print('x : ', x.size())
        # x = F.relu(x)
        # print('x : ', x.size())
        # x = F.max_pool2d(x, (2,2))
        # print('x : ', x.size())
        # # If the size is a square you can only specify a single number
        # x = self.conv2(x)
        # print('x : ', x.size())
        # x = F.relu(x)
        # print('x : ', x.size())
        # x = F.max_pool2d(x, 2)
        # print('x : ', x.size())
        # x = x.view(-1, self.num_flat_features(x))
        # print('x : ', x.size())
        # x = self.fc1(x)
        # print('x : ', x.size())
        # x = F.relu(x)
        # print('x : ', x.size())
        # x = self.fc2(x)
        # print('x : ', x.size())
        # x = F.relu(x)
        # print('x : ', x.size())
        # x = self.fc3(x)
        # print('x : ', x.size())

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]         # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# The learnable parameters of a model are returned by net.parameters.

params = list(net.parameters())
print(len(params))
print(params[0].size())      # conv1's weight
print(params[1].size())
print(params[2].size())
print(params[3].size())
print(params[4].size())
print(params[5].size())
print(params[6].size())
print(params[7].size())
print(params[8].size())
print(params[9].size())



# Lets try a random 32x32 input.
# Note : Expected input size to this net(LeNet) is 32x32. To use this net on MNIST dataset, please resize the images from the dataset to 32x32

input = torch.randn(1,1,32,32)
out = net(input)
print(out)















