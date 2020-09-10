import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision

import jax
from jax import numpy as jnp, random

import dataset_utils
import utils


"""Benchmarking data loading speed for random subset of a dataset.

The task:
- We want to draw random subsets of a dataset representing variation in the
task distribution
- To make an update to a list of models, we want to sample a batch from each
subset
"""

NSEEDS = 5
POINTS = np.logspace(3, np.log(50000), 10)
BATCH_SIZE = 128
jobs = [(point, seed) for point in POINTS for seed in range(NSEEDS)]

dataset = torchvision.datasets.MNIST(
    '../data', train=True, download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
dataset = dataset_utils.DatasetCache(dataset)

DATASET_SIZE = len(dataset)

"""Method 1: multi_loader_iterator.

Create a separate Dataset and Loader for each subset and use
utils.multi_iterator_loader to simultaneously iterate each loader.
"""

def method_multi_loader():
    loaders = []
    for job in jobs:
        shuffled_dataset = dataset_utils.DatasetShuffle(dataset)
        subset_dataset = dataset_utils.DatasetSubset(shuffled_dataset,
                                                     stop=int(job[0]))
        loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, drop_last=True)
        loaders.append(loader)

    loader_iter = utils.multi_loader_iterator(loaders)
    return loader_iter


"""Method 2: MultiSubsetSampler.

Create a regular Torch DataLoader but give it a custom Sampler that returns
multiple batches of data at once.
"""

def method_multi_subset_sampler():
    data_subsets = []
    for point, seed in jobs:
        indices = np.arange(DATASET_SIZE)
        np.random.shuffle(indices)
        indices = indices[:int(point)]
        data_subsets.append(indices)

    data_sampler = utils.MultiSubsetSampler(data_subsets, BATCH_SIZE)
    loader_iter = iter(DataLoader(dataset, batch_sampler=data_sampler))
    return loader_iter


"""Method 3: MultiSubsetLoader.

Uses a custom DataLoader equivalent that pulls one element at a time out of the
dataset according to a MultiSubsetSampler.
"""

def method_multi_subset_loader():
    data_subsets = []
    for point, seed in jobs:
        indices = np.arange(DATASET_SIZE)
        np.random.shuffle(indices)
        indices = indices[:int(point)]
        data_subsets.append(indices)

    loader = utils.MultiSubsetLoader(dataset, data_subsets, BATCH_SIZE)
    loader_iter = iter(loader)
    return loader_iter


"""Method 4: JAX one big tensor + vmap

Puts the whole dataset in a big tensor. Implements a function which samples a
single element and vmaps it to get a batch. vmaps that function to get a batch
of batches.
"""

def method_jax_tensor():
    def stack_data(loader):
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.float())
            ys.append(y)
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    loader = DataLoader(dataset)
    train_x, train_y = stack_data(loader)
    train_x = train_x.numpy().astype(np.float32)
    train_y = train_y.numpy().astype(np.float32)
    train_x, train_y = jnp.array(train_x), jnp.array(train_y)
    train_x = jax.device_put(train_x, jax.local_devices(backend='cpu')[0])
    train_y = jax.device_put(train_y, jax.local_devices(backend='cpu')[0])

    @jax.jit
    def get_example(i, point):
        index = random.randint(random.PRNGKey(i), shape=(1,),
                               minval=0, maxval=DATASET_SIZE)
        index = index[0] % point
        x_i = jax.lax.dynamic_index_in_dim(train_x, index, keepdims=False)
        target_i = jax.lax.dynamic_index_in_dim(train_y, index,
                                                keepdims=False)
        return x_i, target_i, index

    get_batch = jax.vmap(get_example, in_axes=(0, None))
    get_multibatch = jax.vmap(get_batch, in_axes=(None, 0))

    points = jnp.array([j[0] for j in jobs], dtype=jnp.int32)

    def iterate_multibatch():
        i = 0
        while True:
            indices = jnp.arange(i, i + BATCH_SIZE, dtype=jnp.int32)
            yield get_multibatch(indices, points)
            i += BATCH_SIZE

    loader_iter = iterate_multibatch()
    return loader_iter


"""The evaluation loop.
"""

methods = [method_multi_loader, method_multi_subset_sampler,
           method_multi_subset_loader, method_jax_tensor]
# methods = [method_jax_tensor]
for method in methods:
    loader_iter = method()

    STEPS = 1000
    import time
    start = time.time()

    for batch_i in range(STEPS):
        data = next(loader_iter)
        if batch_i % 100 == 0:
            print(batch_i)

    end = time.time()
    time_per_batch = (end - start) / STEPS
    print(f"Method {method.__name__} time per batch: {time_per_batch:.4f}")
