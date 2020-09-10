import jax
from jax import numpy as jnp, random

from flax import nn, optim

from algorithms import common


class MLPClassifier(nn.Module):
    def apply(self, x, hidden_layers, hidden_dim, n_classes):
        x = jnp.reshape(x, (x.shape[0], -1))
        for layer in range(hidden_layers):
            x = nn.Dense(x, hidden_dim, name=f'fc{layer}')
            x = nn.relu(x)
        x = nn.Dense(x, n_classes, name=f'fc{hidden_layers}')
        preds = nn.log_softmax(x)
        return preds


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
    batch = common.batch_to_jax(batch)
    loss, grad = common.loss_and_grad_fn(optimizer.target, batch)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss


@jax.jit
def eval_fn(optimizer, batch):
    batch = common.batch_to_jax(batch)
    return common.loss_fn(optimizer.target, batch)
