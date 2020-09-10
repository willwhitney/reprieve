# import math
# import numpy as np
import pandas as pd

import jax
from jax import numpy as jnp, random

# import flax
from flax import nn, optim

# import torch
# from torch.utils.data import DataLoader
import torchvision

import apiv2
import mnist_vae_repr
from mnist_noisygt_dataset import MNISTNoisyGTDataset


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
    x, y = batch
    return jnp.array(x), jnp.array(y)


@jax.vmap
def cross_entropy(logits, label):
    return -logits[label]


def loss_fn(model, batch):
    logits = model(batch[0])
    loss = jnp.mean(cross_entropy(logits, batch[1]))
    return loss
grad_loss_fn = jax.grad(loss_fn)  # noqa: E305
loss_and_grad_fn = jax.value_and_grad(loss_fn)


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
def train_step_fn(optimizer, batch):
    batch = batch_to_jax(batch)
    loss, grad = loss_and_grad_fn(optimizer.target, batch)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss


@jax.jit
def eval_fn(optimizer, batch):
    batch = batch_to_jax(batch)
    return loss_fn(optimizer.target, batch)


def main(args):
    dataset_mnist = torchvision.datasets.MNIST(
        '../data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    raw_loss_data_estimator = apiv2.LossDataEstimator(
        init_fn, train_step_fn, eval_fn, dataset_mnist,
        train_steps=args.train_steps, n_seeds=args.seeds,
        use_vmap=args.use_vmap, verbose=True)
    raw_loss_data_estimator.compute_curve(n_points=10)

    dataset_noisygt = MNISTNoisyGTDataset(
        split='train', ntrain=60000, nval=0, ntest=0,
        p_corrupt=0.05)
    noisy_loss_data_estimator = apiv2.LossDataEstimator(
        init_fn, train_step_fn, eval_fn, dataset_noisygt,
        train_steps=args.train_steps, n_seeds=args.seeds,
        use_vmap=args.use_vmap, verbose=True)
    noisy_loss_data_estimator.compute_curve(n_points=10)

    raw_results = raw_loss_data_estimator.to_dataframe()
    raw_results['name'] = 'Raw'
    noisy_results = noisy_loss_data_estimator.to_dataframe()
    noisy_results['name'] = 'Noisy labels'

    outcome_df = pd.concat([
        raw_results,
        noisy_results,
    ])

    save_path = ('results/'
                 f'results_vmap{args.use_vmap}'
                 f'_train{args.train_steps}'
                 f'_seed{args.seeds}.png')
    apiv2.render_curve(outcome_df, save_path=save_path)
    # return outcome_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_vmap', dest='use_vmap', action='store_false')
    parser.add_argument('--train_steps', type=float, default=4e3)
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    import time
    start = time.time()
    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
    end = time.time()
    print(f"Time: {end - start :.3f} seconds")
