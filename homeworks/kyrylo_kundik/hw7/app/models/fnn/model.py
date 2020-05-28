import torch
from sklearn.metrics import mean_squared_error


class Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num):
        super(Model, self).__init__()

        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(hidden_num - 1):
            self.hidden_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(torch.nn.ReLU())
        self.out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.input(x)
        out = self.activation(out)
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers[i](out)

        out = self.out(out)

        return out


def train_model(model, alpha, epochs, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):

        for data in loader:
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()


def validate_model(model, train_loader, test_loader):
    mse_train, mse_test = [], []

    model.eval()

    with torch.no_grad():

        for inputs, labels in train_loader:
            outputs = model(inputs)
            mse_train.append(mean_squared_error(labels, outputs))

        for inputs, labels in test_loader:
            outputs = model(inputs)
            mse_test.append(mean_squared_error(labels, outputs))

    model.train()

    return mse_train, mse_test
