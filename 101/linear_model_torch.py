"""
This is the final code using PyTorch to fit a linear model. It contains
1. How to prepare dataset and how to use data loader
2. How to write a make a train step function
3. Integrated model, loss function, and optimizer
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split

"""
Generate Data
"""
n = 1000
rv = np.random.RandomState(0)
x = rv.uniform(0, 1, [n,1])
y = 1 + 2 * x + 0.1*rv.normal(0, 1, [n,1])

"""
Fit Model Using PyTorch
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Prepare for the data
"""
class CustomDataSet(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

# x_train_tensor = torch.from_numpy(x_train).float()
# y_train_tensor = torch.from_numpy(y_train).float()
# train_data = CustomDataSet(x_train_tensor, y_train_tensor)
# the above lines generate the same results as using TensorDataset,
# however when data are complicated or very large, we need to write our own dataset by loading data from disk.
# print(train_data[0])
# train_data = TensorDataset(x_train_tensor, y_train_tensor)
# print(train_data[0])

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

dataset = TensorDataset(x_tensor, y_tensor)
lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
train_dataset, test_dataset = random_split(dataset, lengths)

train_loader = DataLoader(dataset = train_dataset, batch_size = 16, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 20)

"""
Define Model, Loss Function, Optimizer
"""

class LayerLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

loss_fn = nn.MSELoss(reduction='mean')
model = LayerLinearRegression().to(device)
optimizer = optim.SGD(model.parameters(), lr = 1e-1)

"""
Make A Train Step
"""
def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        # set the model in train state
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

""""Train The Model"""
losses = []
test_losses = []
n_epochs = 1000

train_step = make_train_step(model, loss_fn, optimizer)

for epoch in range(n_epochs):
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss = train_step(x_batch,y_batch)
        losses.append(loss)

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            model.eval()
            yhat = model(x_test)
            test_loss = loss_fn(y_test, yhat)
            test_losses.append(test_loss.item())

print(model.state_dict())




