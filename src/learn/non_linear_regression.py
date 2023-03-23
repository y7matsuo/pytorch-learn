import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def quadratic(x):
    return x ** 2

def cos(x):
    return np.cos(x * 2)

def random_choice(fn, min_x, max_x, size, choice_size):
    xs = np.linspace(min_x, max_x, size).astype(float)
    choiced_xs = np.random.choice(xs, size=choice_size)
    choiced_ys = fn(choiced_xs)
    return choiced_xs, choiced_ys

def plot_random_choice(fn, min_x, max_x, size, choice_size):
    xs, ys = random_choice(fn, min_x, max_x, size, choice_size)
    plt.scatter(xs, ys)
    plt.grid()
    plt.show()

def ml_non_linear_regression(fn, min_x, max_x, size, choice_size):
    inputs, labels = random_choice(fn, min_x, max_x, size, choice_size)

    inputs_tensor = torch.tensor(inputs).float().view(-1, 1)
    labels_tensor = torch.tensor(labels).float().view(-1, 1)

    lr = 0.01
    num_epochs = 30000

    net = nn.Sequential(
        nn.Linear(1, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    history = np.zeros((0, 2))
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        outputs = net(inputs_tensor)
        loss = criterion(outputs, labels_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'epoch: {epoch} loss: {loss.item():.5f}')
            items = [epoch, loss.item()]
            history = np.vstack((history, items))

    return net, inputs, labels, history

def do_ml_quadratic():
    net, inputs, labels, history = ml_non_linear_regression(quadratic, -2.0, 2.0, 1000, 50)
    xs = np.arange(-2.0, 2.0, 0.01)
    ys = net(torch.tensor(xs).float().view(-1, 1)).detach().numpy()
    plot_inputs_labels(inputs, labels)
    plot_history(history)
    plot_net(xs, ys, inputs, labels)

def do_ml_cos():
    net, inputs, labels, history = ml_non_linear_regression(cos, -2.0 * np.pi, 2.0 * np.pi, 1000, 100)
    xs = np.arange(-2.0 * np.pi, 2.0 * np.pi, 0.01)
    ys = net(torch.tensor(xs).float().view(-1, 1)).detach().numpy()
    plot_inputs_labels(inputs, labels)
    plot_history(history)
    plot_net(xs, ys, inputs, labels)

def plot_inputs_labels(inputs, labels):
    plt.scatter(inputs, labels)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

def plot_net(xs, ys, inputs, labels):
    plt.scatter(inputs, labels, marker='x', c='k')
    plt.plot(xs, ys, c='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

def plot_history(history):
    plt.plot(history[:, 0], history[:, 1])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()
