"""
There are illustration codes on how to use PyTorch utilities to reduce the work
"""

import numpy as np
import torch

"""
Prepare Data
"""
n = 1000
rv = np.random.RandomState(0)
x = rv.uniform(0, 1, [n, 1])
y = 1 + 2 * x + 0.1 * rv.normal(0, 1, [n, 1])

idx = np.arange(n)
rv.shuffle(idx)

train_idx = idx[:int(0.8 * n)]
test_idx = idx[int(0.8 * n):]

x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

"""
PyTorch Training #1:
Use autograd to reduce the work
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
n_epochs = 1000
lr = 1e-1

# when create a new tensor, requires_grad often set as false, so we need to set to True explicitly.
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

x_train_tensor = torch.from_numpy(x_train).to(device, dtype=torch.float)
y_train_tensor = torch.from_numpy(y_train).to(device, dtype=torch.float)

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()

    # No more manual computation of gradients!
    # a_grad = -2 * error.mean()
    # b_grad = -2 * (x_tensor * error).mean()
    # We just tell PyTorch to work its way BACKWARDS from the specified loss!
    loss.backward()

    with torch.no_grad():
        a -= lr * a.grad # <1>
        b -= lr * b.grad

    a.grad.zero_()
    b.grad.zero_()

print(a, b)

# <1> this code is different than a = a - lr*a.grad, the -= replace 'a' in place, but the later one create a new a.
# Also when a new 'a' is created, we lose its gradients. it is also important to set an enviorment with
# torch.no_grad() so that it will not add this update into gradient computation.

"""
PyTorch Training #2:
Use optimizer to reduce the work
"""
import torch.optim as optim

optimizer = optim.SGD([a,b], lr=lr)

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()

    # No more manual computation of gradients!
    # a_grad = -2 * error.mean()
    # b_grad = -2 * (x_tensor * error).mean()
    # We just tell PyTorch to work its way BACKWARDS from the specified loss!
    loss.backward()

    # with torch.no_grad():
    #     a -= lr * a.grad # <1>
    #     b -= lr * b.grad
    optimizer.step()
    # a.grad.zero_()
    # b.grad.zero_()
    optimizer.zero_grad()

print(a, b)

"""
PyTorch Training #3:
Use Loss to reduce the work
"""
import torch.nn as nn
loss_fn = nn.MSELoss(reduction='mean')

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor

    # error = y_train_tensor - yhat
    # loss = (error ** 2).mean()
    loss = loss_fn(y_train_tensor, yhat)
    # No more manual computation of gradients!
    # a_grad = -2 * error.mean()
    # b_grad = -2 * (x_tensor * error).mean()
    # We just tell PyTorch to work its way BACKWARDS from the specified loss!
    loss.backward()

    # with torch.no_grad():
    #     a -= lr * a.grad # <1>
    #     b -= lr * b.grad
    optimizer.step()
    # a.grad.zero_()
    # b.grad.zero_()
    optimizer.zero_grad()

print(a, b)

"""
PyTorch Training #4:
Use model to reduce the work
"""

class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.a + self.b*x

# create the model and sent to device
model = ManualLinearRegression().to(device)

for epoch in range(n_epochs):
    # yhat = a + b * x_train_tensor
    model.train()
    yhat = model(x_train_tensor)
    # error = y_train_tensor - yhat
    # loss = (error ** 2).mean()
    loss = loss_fn(y_train_tensor, yhat)
    # No more manual computation of gradients!
    # a_grad = -2 * error.mean()
    # b_grad = -2 * (x_tensor * error).mean()
    # We just tell PyTorch to work its way BACKWARDS from the specified loss!
    loss.backward()

    # with torch.no_grad():
    #     a -= lr * a.grad # <1>
    #     b -= lr * b.grad
    optimizer.step()
    # a.grad.zero_()
    # b.grad.zero_()
    optimizer.zero_grad()

print(a, b)

"""
PyTorch Training #5:
Improve writing the model using built in sequential models
"""

class LayerLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

model = LayerLinearRegression().to(device)

for epoch in range(n_epochs):
    model.train()
    yhat = model(x_train_tensor)
    loss = loss_fn(y_train_tensor, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(a, b)

"""
PyTorch Training #6:
Put model, loss_fn, and optimizer into a train step function
"""

def make_train_step(model, loss_fn, optimizer):
    def train_step(x,y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss)
        return loss.item()
    return train_step

train_step = make_train_step(model,loss_fn, optimizer)

losses =[]

for epoch in range(n_epochs):
    loss = train_step(x_train_tensor,y_train_tensor)
    losses.append(loss)
print(a, b)

