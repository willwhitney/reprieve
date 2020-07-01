import math
import numpy as np

import jax
from jax import numpy as jnp, random, jit, lax

import flax
from flax import nn, optim

import torch
from torch.utils.data import DataLoader
import torchvision

import api


class MLPClassifier(nn.Module):
    def apply(self, x, hidden_layers, hidden_dim, n_classes):
        x = jnp.reshape(x, (x.shape[0], -1))
        for layer in range(hidden_layers):
            x = nn.Dense(x, hidden_dim, name=f'fc{layer}')
            x = nn.relu(x)
        x = nn.Dense(x, hidden_dim, name=f'fc{hidden_layers}')
        preds = nn.log_softmax(x)
        return preds


@jax.vmap
def cross_entropy(logits, label):
    return -logits[label]


def loss_fn(model, batch):
    logits = model(batch[0])
    loss = jnp.mean(cross_entropy(logits, batch[1]))
    return loss


grad_loss_fn = jax.grad(loss_fn)


@jax.jit
def train_step(optimizer, batch):
    grad = grad_loss_fn(optimizer.target, batch)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer


def evaluate(model, dataset):
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    loss = 0
    for x, y in loader:
        loss += loss_fn(model, (jnp.array(x), jnp.array(y))) * x.shape[0]
    return loss / len(dataset)


def algorithm(dataset, seed):
    torch.manual_seed(seed)
    rng = random.PRNGKey(seed)
    # hidden_layers = 2
    # hidden_dim = 512
    # n_classes = 10
    classifier = MLPClassifier.partial(hidden_layers=1, hidden_dim=128, n_classes=10)
    _, initial_params = classifier.init_by_shape(rng, [(128, 784)])
    initial_model = nn.Model(classifier, initial_params)
    optimizer = optim.Adam(1e-3).create(initial_model)
    # import ipdb; ipdb.set_trace()
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    steps = 1e3
    step = 0
    while step < steps:
        for x, y in loader:
            train_step(optimizer, (jnp.array(x), jnp.array(y)))
            step += 1
            if step >= steps:
                break
    final_model = optimizer.target
    return final_model, evaluate


if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST(
        '../data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    loss_data_estimator = api.LossDataEstimator(algorithm, dataset)
    loss_data_estimator.compute_curve(n_points=2)
