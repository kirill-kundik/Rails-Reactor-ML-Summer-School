# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch

from torch.utils.data import Dataset, DataLoader

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import time
import os

print(os.listdir("../input"))

from abc import abstractmethod

import torch
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.len = X.shape[0]
        self.x_data = torch.tensor(X, dtype=torch.float32)
        self.y_data = torch.tensor(y, dtype=torch.float32)
        self.dim = self.x_data.shape[1]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def get_dim(self):
        return self.dim


class LinearRegression(torch.nn.Module):
    def __init__(self, in_size, out_size, device='cpu'):
        super().__init__()
        self.linear = torch.nn.Linear(in_size, out_size).to(device)

    def forward(self, X):
        return self.linear(X)


class LogisticRegression(torch.nn.Module):
    def __init__(self, in_size, out_size, device='cpu'):
        super().__init__()
        self.linear = torch.nn.Linear(in_size, out_size).to(device)

    def forward(self, X):
        return torch.sigmoid(self.linear(X))


def train_model(model, optimizer, criterion, train_loader, epochs, device):
    for epoch in range(epochs):

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()


def train_test_split(x, y, test_size=0.2, random_state=42):
    test = np.round(x.shape[0] * test_size)

    np.random.seed(random_state)
    test_indices = np.random.choice(np.arange(x.shape[0]), size=np.int(test), replace=False)

    x_train = np.zeros((0, x.shape[1]))
    x_test = np.zeros((0, x.shape[1]))
    y_train = np.zeros((0, y.shape[1]))
    y_test = np.zeros((0, y.shape[1]))

    for i in range(x.shape[0]):
        if np.any(test_indices == i):
            x_test = np.vstack([x_test, x[i, :]])
            y_test = np.vstack([y_test, y[i, :]])
        else:
            x_train = np.vstack([x_train, x[i, :]])
            y_train = np.vstack([y_train, y[i, :]])

    return x_train, x_test, y_train, y_test


def read_file(dataset_path: str, target: str, na: str, categorical: str):
    data = pd.read_csv(dataset_path).reset_index().drop(["index"], axis=1)

    print(f'DATASET LOADED FROM {dataset_path}, target: {target}, NA: {na}, categorical: {categorical}')

    targets = data[[target]].values
    data.drop([target], axis=1, inplace=True)

    if na:
        for n in na.split(','):
            median = data[n].median()
            data[n] = data[n].fillna(median)

    standard_scaler = StandardScaler()

    if categorical:
        num_columns = list(set(data.columns) - set(categorical.split(',')))

        data[num_columns] = standard_scaler.fit_transform(data[num_columns])
        data = pd.get_dummies(data, columns=categorical.split(','))
        columns = data.columns
        data = np.asarray(data)
    else:
        data = pd.get_dummies(data)
        columns = data.columns
        data = standard_scaler.fit_transform(data)

    return data, targets, columns


random_state = 42

X, y, columns = read_file("../input/heart_data.csv", "target", "age,thal", "sex,cp,fbs,restecg,exang,slope,ca,thal")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
X_train, X_test, y_train, y_test = (
    torch.from_numpy(X_train).float(),
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_train).float(),
    torch.from_numpy(y_test).float()
)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_dataset = MyDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=32)

test_dataset = MyDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=32)

model = LogisticRegression(train_dataset.get_dim(), 1, device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

start_time = time.time()
train_model(model, optimizer, criterion, train_loader, 1000, device)
fit_time = time.time() - start_time


def accuracy(y_true, y_pred):
    return ((y_true == y_pred).sum().type(torch.float32)) / y_true.shape[0]


def recall(y_true, y_pred):
    return (y_true.type(torch.int32) & y_pred.type(torch.int32)).sum() / y_true.sum()


def precision(y_true, y_pred):
    return (y_true.type(torch.int32) & y_pred.type(torch.int32)).sum() / y_pred.sum()


def fbeta(y_true, y_pred, beta):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return (1 + beta ** 2) * (prec * rec) / ((beta ** 2) * prec + rec)


def f1(y_true, y_pred):
    return fbeta(y_true, y_pred, 1)


def log_loss(y_true, y_pred):
    return (-y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).mean()


preds = []
trues = []
with torch.no_grad():
    for data in test_loader:
        X_test, y_test = data
        X_test, y_test = X_test.to(device), y_test.to(device)

        pred = model(X_test)

        pred = torch.tensor([i[0] for i in pred], dtype=torch.float32)
        y_test = torch.tensor([i[0] for i in y_test], dtype=torch.float32)

        preds.extend(pred)
        trues.extend(y_test)

trues = torch.tensor(trues, dtype=torch.float32)
preds = torch.tensor(preds, dtype=torch.float32)

loss = log_loss(trues, preds)
for i, pred in enumerate(preds):
    if pred >= 0.5:
        preds[i] = 1
    else:
        preds[i] = 0

m = {
    'accuracy': accuracy(trues, preds),
    'recall': recall(trues, preds),
    'precision': precision(trues, preds),
    'f1': f1(trues, preds),
    'log-loss': loss
}

metrics_out = '\n'.join(f'               {k}: {v}' for k, v in m.items())
print(f'Model metrics: \n{metrics_out}')
print(f'Fit time:\n          {fit_time} s')
# Any results you write to the current directory are saved as output.
