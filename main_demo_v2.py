import math
import numpy as np

import jax
from jax import numpy as jnp, random, jit, lax

import flax
from flax import nn, optim

import torch
from torch.utils.data import DataLoader
import torchvision

import apiv2


class MLPClassifier(nn.Module):
    def apply(self, x, hidden_layers, hidden_dim, n_classes):
        x = jnp.reshape(x, (x.shape[0], -1))
        for layer in range(hidden_layers):
            x = nn.Dense(x, hidden_dim, name=f'fc{layer}')
            x = nn.relu(x)
        x = nn.Dense(x, n_classes, name=f'fc{hidden_layers}')
        preds = nn.log_softmax(x)
        return preds


def batch_to_jax(batch):
    return (jnp.array(batch[0]), jnp.array(batch[1]))


@jax.vmap
def cross_entropy(logits, label):
    return -logits[label]


def loss_fn(model, batch):
    logits = model(batch[0])
    loss = jnp.mean(cross_entropy(logits, batch[1]))
    return loss
grad_loss_fn = jax.grad(loss_fn)


def init_fn(seed):
    rng = random.PRNGKey(seed)
    classifier = MLPClassifier.partial(hidden_layers=2,
                                       hidden_dim=512,
                                       n_classes=10)
    _, initial_params = classifier.init_by_shape(rng, [(128, 784)])
    initial_model = nn.Model(classifier, initial_params)
    optimizer = optim.Adam(1e-3).create(initial_model)
    return optimizer


@jax.jit
def train_step(optimizer, batch):
    batch = batch_to_jax(batch)
    grad = grad_loss_fn(optimizer.target, batch)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer


def eval_fn(optimizer, batch):
    batch = batch_to_jax(batch)
    return loss_fn(optimizer.target, batch)


if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST(
        '../data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    loss_data_estimator = apiv2.LossDataEstimator(
        init_fn, train_step, eval_fn, dataset,
        train_steps=1e3, use_vmap=True)
    loss_data_estimator.compute_curve(n_points=10)
