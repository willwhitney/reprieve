import numpy as np
import torch
import random


def batch_to_numpy(batch):
    return (batch[0].numpy(), batch[1].numpy())


# @profile
def multi_loader_iterator(loaders):
    iterators = [iter(loader) for loader in loaders]

    # TODO: figure out how to deal with small / differently-sized batches
    while True:
        xs, ys = [], []
        for i in range(len(iterators)):
            try:
                x, y = next(iterators[i])
            except StopIteration:
                iterators[i] = iter(loaders[i])
                x, y = next(iterators[i])
            xs.append(x)
            ys.append(y)
        yield torch.stack(xs), torch.stack(ys)
    # return _helper


# @profile
def forever_iterator(loader):
    iterator = iter(loader)
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        yield batch


class MultiSubsetLoader():
    def __init__(self, dataset, subsets, batch_size=128):
        self.dataset = dataset
        self.subsets = subsets
        self.batch_size = batch_size
        self.sampler = MultiSubsetSampler(subsets, batch_size)
        # self.sampler_iter = iter(self.sampler)

    # @profile
    def __iter__(self):
        for index_batch in self.sampler:
            xs = []
            ys = []
            for i in index_batch:
                point = self.dataset[i]
                xs.append(torch.as_tensor(point[0]))
                ys.append(torch.as_tensor(point[1]))
            flat_x = torch.stack(xs)
            flat_y = torch.stack(ys)
            stacked_x = flat_x.reshape(
                (len(self.subsets), self.batch_size, *flat_x.shape[2:]))
            stacked_y = flat_y.reshape(
                (len(self.subsets), self.batch_size, *flat_y.shape[2:]))

            yield (stacked_x, stacked_y)


class MultiSubsetSampler(torch.utils.data.Sampler):
    def __init__(self, subsets, batch_size=128):
        self.subsets = subsets
        self.batch_size = batch_size

    # @profile
    def __iter__(self):
        while True:
            all_indices = []
            for subset in self.subsets:
                indices = np.random.choice(subset, size=(self.batch_size,))
                all_indices.append(indices)
            yield [el for lst in all_indices for el in lst]

    def __len__(self):
        return int(1e10)


@profile
def dataset_to_jax(dataset):
    import jax
    from jax import numpy as jnp

    @profile
    def stack_data(loader):
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.float())
            ys.append(y)
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10000)
    train_x, train_y = stack_data(loader)

    train_x = train_x.numpy()  # .astype(np.float32)
    train_y = train_y.numpy()  # .astype(np.float32)
    train_x, train_y = jnp.array(train_x), jnp.array(train_y)
    train_x = jax.device_put(train_x, jax.local_devices(backend='cpu')[0])
    train_y = jax.device_put(train_y, jax.local_devices(backend='cpu')[0])
    return train_x, train_y


def jax_multi_iterator(dataset, batch_size, seeds, subset_sizes):
    import jax
    from jax import numpy as jnp, random as jr
    seeds = jnp.array(seeds)
    subset_sizes = jnp.array(subset_sizes, dtype=jnp.int32)

    train_x, train_y = dataset_to_jax(dataset)
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

    # subsets = [[0, 1, 2],
    #            [0, 0, 0],
    #            [1]]
    # batch_sampler = MultiSubsetSampler(dataset, subsets)
    # loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)

    def test_iteration():
        batch_size = 128
        seeds = list(range(5))
        subset_sizes = [1, 1, 4, 4, 10000]
        multi_iterator = jax_multi_iterator(
            dataset, batch_size, seeds, subset_sizes)

        batch = next(multi_iterator)
        print("done")

    if args.debug:
        import jax
        with jax.disable_jit():
            print("running without jit")
            test_iteration()
    else:
        print("running with jit")
        test_iteration()
