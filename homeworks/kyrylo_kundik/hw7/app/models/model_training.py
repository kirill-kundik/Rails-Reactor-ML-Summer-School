import time

import torch
from torch.utils.data import DataLoader

from app.config import PROJECT_ROOT
from app.models.fnn.dataset import ApartmentsDataset
from app.models.fnn.model import Model, validate_model, train_model
from app.utilities import load_env


def train():
    train_dataset = ApartmentsDataset(PROJECT_ROOT / 'data' / 'train.csv')
    test_dataset = ApartmentsDataset(PROJECT_ROOT / 'data' / 'test.csv')

    train_loader = DataLoader(dataset=train_dataset, batch_size=45)
    test_loader = DataLoader(dataset=test_dataset, batch_size=45)

    model = Model(input_dim=train_dataset.get_dim(), hidden_dim=20, hidden_num=3)
    print('Fitting nn model')
    start_time = time.time()
    train_model(model=model, alpha=0.01, epochs=100, loader=train_loader)
    print(f'Fit time: {time.time() - start_time} s')
    mse_train, mse_test = validate_model(model=model,
                                         train_loader=train_loader,
                                         test_loader=test_loader)
    info = f"""
NN MODEL:
    Train mse: {sum(mse_train) / len(mse_train)}
    Test mse: {sum(mse_test) / len(mse_test)}
    """
    print(info)
    print('Saving model')
    torch.save(model.state_dict(), str(PROJECT_ROOT / 'data' / 'nn.model'))


if __name__ == '__main__':
    load_env()

    train()
