import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

def std_transform(x):
  scaler = StandardScaler()
  return scaler.fit_transform(x), scaler

def get_diabetes_data():
  dataset = load_diabetes()
  data = np.array(dataset.data)
  target = np.array(dataset.target)
  attrs = np.array(['age', 'sex', 'bmi', 'blood pressure', 's1', 's2', 's3', 's4', 's5', 's6'])
  return data, target, attrs

def get_std_transformed_data(attr):
  data, target, attrs = get_diabetes_data()
  data, data_scaler = std_transform(data[:, attrs == attr])
  target, target_scaler = std_transform(target.reshape(-1, 1))
  return data, target, data_scaler, target_scaler

def plot_scatter(attr):
  data, target, attrs = get_diabetes_data()
  data = data[:, attrs == attr]
  plt.scatter(data, target)
  plt.xlabel(attr)
  plt.ylabel('disease progression')
  plt.grid()
  plt.show()

def ml_bp_disease_linear_regression():
  data, target, _, _ = get_std_transformed_data('blood pressure')

  inputs = torch.tensor(data).float()
  labels = torch.tensor(target).float()

  lr = 0.01
  num_epochs = 1000
  
  net = nn.Linear(1, 1)
  criterion = nn.MSELoss()
  optimizer = optim.SGD(net.parameters(), lr=lr)

  history = np.zeros((0, 2))
  for epoch in range(num_epochs):
    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
      print(f'epoch: {epoch} loss: {loss.item():.5f}')
      items = [epoch, loss.item()]
      history = np.vstack((history, items))

  return net, history

def plot_learning_curve(history):
  plt.plot(history[:, 0], history[:, 1])
  plt.grid()
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.show()

def plot_net(net):
  attr = 'blood pressure'
  data, target, attrs = get_diabetes_data()

  plt.scatter(data[:, attrs == attr], target, c='k')
  plt.xlabel(attr)
  plt.ylabel('disease progression')
  plt.grid()
  
  xs = torch.tensor(np.arange(-3, 3.1, 0.1).reshape(-1, 1)).float()
  ys = net(xs)

  _, _, data_scaler, target_scaler = get_std_transformed_data(attr)
  xs = data_scaler.inverse_transform(xs)
  ys = target_scaler.inverse_transform(ys.detach())

  plt.plot(xs.reshape(-1), ys.reshape(-1), c='b')
  plt.show()
