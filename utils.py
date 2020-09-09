import numpy as np
import torch
import random


def batch_to_numpy(batch):
    return (batch[0].numpy(), batch[1].numpy())


def make_cpu_tensor(shape, dtype=float):
    import jax
    from jax import numpy as jnp
    tiny = jnp.zeros((), dtype=dtype)
    tiny_cpu = jax.device_put(tiny, jax.local_devices(backend='cpu')[0])
    big_cpu = jnp.tile(tiny_cpu, shape)
    return big_cpu


def torch_to_jax(tensor):
    """Zero-copy transfer of a torch CPU tensor to a JAX CPU ndarray."""
    import jax
    from jax import dlpack as jdlpack
    from torch.utils import dlpack as tdlpack

    cpu_backend = jax.local_devices(backend='cpu')[0]
    packed_tensor = tdlpack.to_dlpack(tensor)
    return jdlpack.from_dlpack(packed_tensor, backend=cpu_backend)


def dataset_to_jax(dataset, batch_transforms,
                   batch_size=256, whiten=False, stats=None):
    """Transforms a dataset one batch at a time and returns a JAX tensor.

    The code is convoluted in order to avoid ever having two copies of the
    dataset in memory.
    """
    x_shape = dataset[0][0].shape
    y_shape = dataset[0][1].shape

    def stack_data(loader):
        xs = torch.empty((len(dataset), *x_shape), dtype=dataset[0][0].dtype)
        ys = torch.empty((len(dataset), *y_shape), dtype=dataset[0][1].dtype)
        i = 0
        for x, y in loader:
            x = apply_transforms(batch_transforms, x.numpy())
            xs[i: i + x.shape[0]] = x
            ys[i: i + y.shape[0]] = y
        return xs, ys

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    train_x, train_y = stack_data(loader)

    if whiten:
        if stats is None:
            stats = (train_x.mean(), train_x.std())
        train_x.sub_(stats[0])
        train_x.div_(stats[1])
    train_x = torch_to_jax(train_x)
    train_y = torch_to_jax(train_y)
    return (train_x, train_y), stats


def jax_multi_iterator(dataset, batch_size, seeds, subset_sizes):
    import jax
    from jax import numpy as jnp, random as jr
    seeds = jnp.array(seeds)
    subset_sizes = jnp.array(subset_sizes, dtype=jnp.int32)
    train_x, train_y = dataset
    dataset_size = len(train_x)

    @jax.jit
    def get_example(subset_size, seed, i):
        # generate a new dataset seed to prevent overlap of consecutive seeds
        dataset_seed = jr.split(jr.PRNGKey(seed), 1)[0][0]

        # ensure that there are only subset_size seeds using modulo
        i = i % subset_size

        point_seed = dataset_seed + i
        point_index = jr.randint(jr.PRNGKey(point_seed), shape=(),
                                 minval=0, maxval=dataset_size)
        x_i = jax.lax.dynamic_index_in_dim(train_x, point_index,
                                           keepdims=False)
        y_i = jax.lax.dynamic_index_in_dim(train_y, point_index,
                                           keepdims=False)
        return x_i, y_i

    get_batch = jax.vmap(get_example, in_axes=(None, None, 0))
    get_multibatch = jax.vmap(get_batch, in_axes=(0, 0, None))

    def iterate_multibatch():
        i = 0
        while True:
            indices = jnp.arange(i, i + batch_size, dtype=jnp.int32)
            yield get_multibatch(subset_sizes, seeds, indices)
            i += batch_size

    loader_iter = iterate_multibatch()
    return loader_iter


def apply_transforms(batch_transforms, x):
    for t in batch_transforms:
        x = t(x)
    return x


def compute_stats(dataset, batch_transforms, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size)
    n = len(dataset) * dataset[0][0].nelement()

    data_sum, data_sum_of_squares = 0, 0
    offset = None
    for x, y in loader:
        x = apply_transforms(batch_transforms, x)
        x = x.reshape((x.shape[0], -1))
        if offset is None:
            offset = x.mean()

        x_sum = x.sum()
        data_sum += (x_sum - offset)
        data_sum_of_squares += ((x - offset) ** 2).sum()
    data_mean = offset + data_sum / n
    data_variance = (data_sum_of_squares - data_sum ** 2 / n) / n
    return data_mean, data_variance ** 0.5


def make_whiten_transform(mean, std):
    def _helper(x):
        return (x - mean) / std
    return _helper


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    import torchvision
    dataset = torchvision.datasets.MNIST(
        '../data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    # def test_iteration():
    #     batch_size = 128
    #     seeds = list(range(5))
    #     subset_sizes = [1, 1, 4, 4, 10000]
    #     multi_iterator = jax_multi_iterator(
    #         dataset, batch_size, seeds, subset_sizes)

    #     batch = next(multi_iterator)
    #     print("done")

    # if args.debug:
    #     import jax
    #     with jax.disable_jit():
    #         print("running without jit")
    #         test_iteration()
    # else:
    #     print("running with jit")
    #     test_iteration()

    import dataset_utils
    dataset = dataset_utils.DatasetCache(dataset)
    white_dataset = dataset_utils.DatasetWhiten(dataset)
    print(white_dataset.mean, white_dataset.std)

    mean, std = compute_stats(dataset, [], 256)
    print(mean, std)

    # train_x, _ = dataset_to_jax(dataset)
    # print(train_x.mean(), train_x.std())
