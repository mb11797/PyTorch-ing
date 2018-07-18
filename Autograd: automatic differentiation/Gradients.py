import torch
x = torch.ones(2,2, requires_grad=True)
print(x)

print(x.requires_grad)
print(x.grad_fn)
print('#####################')

y = x + 2
print(y)
print(y.requires_grad)
print(y.grad_fn)
print('#####################')

z = y * y * 3
out = z.mean()

print(z)
print(z.requires_grad)
print(z.grad_fn)
print('#####################')

print(out)
print(out.requires_grad)
print(out.grad_fn)
print('#####################')

# Let’s backprop now Because out contains a single scalar, out.backward() is equivalent to out.backward(torch.tensor(1)).
out.backward()

# print gradients d(out)/dx
print(x.grad)
print(y.grad)
print(z.grad)
print(out.grad)
print('#####################')

# z.backward()
# print(x.grad)
# print(y.grad)
# print(z.grad)


print('#####################')
# You should have got a matrix of 4.5. Let’s call the out Tensor “o”. We have that o = (1/4) * ∑zi, zi=3((xi+2)^2)/2 and zi(xi=1)=27. Therefore, ∂o/∂xi=(3/2) *(xi+2),
# hence ∂o/∂xi (xi=1) = 9/2 = 4.5.

# You can do many crazy things with autograd!

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)
print(y.requires_grad)
print(y.grad_fn)
print(y.grad)
print(x)
print(x.grad)

print('#####################')
gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)

print(x.grad)

print('#####################')

# You can also stop autograd from tracking history on Tensors
# with .requires_grad``=True by wrapping the code block in ``with torch.no_grad():

print(x.requires_grad)
print((x ** 2).requires_grad)
print((x ** 2).grad_fn)

with torch.no_grad():
    print((x ** 2).requires_grad)
    print((x ** 2).grad_fn)


