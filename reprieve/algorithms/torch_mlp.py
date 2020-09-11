import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


DEVICE = "cuda"


class MLPClassifier(nn.Module):
    def __init__(self, hidden_layers, hidden_dim, input_dim, n_classes):
        super().__init__()
        layers = [nn.Flatten()]
        dims = [input_dim,
                *[hidden_dim for _ in range(hidden_layers)],
                n_classes]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.LogSoftmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_algorithm(input_shape, n_classes):
    def init_fn(seed):
        torch.manual_seed(seed)
        input_dim = np.array(input_shape).prod()
        model = MLPClassifier(hidden_layers=2, hidden_dim=512,
                              input_dim=input_dim, n_classes=n_classes)
        model = model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        state = (model, optimizer)
        return state

    def train_step_fn(state, batch):
        model, optimizer = state
        model.train()
        x, y = batch
        x = torch.as_tensor(x).to(DEVICE)
        y = torch.as_tensor(y, dtype=torch.long).to(DEVICE)

        optimizer.zero_grad()
        pred = model(x)
        loss = F.nll_loss(pred, y)
        loss.backward()
        optimizer.step()
        return (model, optimizer), loss.item()

    def eval_fn(state, batch):
        model, optimizer = state
        model.eval()
        with torch.no_grad():
            x, y = batch
            x = torch.as_tensor(x).to(DEVICE)
            y = torch.as_tensor(y, dtype=torch.long).to(DEVICE)

            pred = model(x)
            loss = F.nll_loss(pred, y)
        return loss.item()
    return init_fn, train_step_fn, eval_fn
