import jax
from jax import numpy as jnp


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
