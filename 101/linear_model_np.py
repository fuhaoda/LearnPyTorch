import numpy as np

"""
Data Generation 
"""
n = 100
rv = np.random.RandomState(0)  # use random.RandomState to protect thread safe

x = rv.uniform(0,1,[n,1])
y = 1 + 2*x + 0.1*rv.normal(0,1,[n,1])

idx = np.arange(n)
rv.shuffle(idx)
train_idx = idx[:int(0.8*n)]
test_idx = idx[int(0.8*n):]
x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

"""
Fit Model by Numpy
"""
# initializes parameters "a" and "b" randomly
a = rv.normal(0,1,1)
b = rv.normal(0,1,1)

# define learning rate, number of epochs
lr = 1e-1
n_epochs = 1000

for epoch in range(n_epochs):
    # pathforward to calculate the loss
    yhat = a + b*x_train
    error = y_train - yhat
    loss = (error**2).mean()

    a_grad = -2*error.mean()
    b_grad = -2*(x_train*error).mean()

    a -= lr*a_grad
    b -= lr*b_grad

print(f'a={a}, b={b}')

# compare results with linear regression package
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train) # reshape the x into a 2D array
print(lm.intercept_,lm.coef_)

# change the learning rate will chagne the results slightly
# if the dataset is very large, we need to split it into mini-batches





