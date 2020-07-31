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


""" iteration time:
   188       100   12,114,448.0 121144.5     46.7              stacked_batch = next(multi_iterator)
   237       214   5,353,277.0  25015.3      27.4              for batch in loader:

   189      1000   85,347,443.0  85347.4     81.9              stacked_batch = next(multi_iterator)
   237     10369   28,449,941.0   2743.7     34.2              for batch in loader:

"""


class MultiSubsetLoader():
    def __init__(self, dataset, subsets, batch_size=128):
        self.dataset = dataset
        self.subsets = subsets
        self.batch_size = batch_size
        self.sampler = MultiSubsetSampler(subsets, batch_size)
        # self.sampler_iter = iter(self.sampler)

    @profile
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

    @profile
    def __iter__(self):
        while True:
            all_indices = []
            for subset in self.subsets:
                indices = np.random.choice(subset, size=(self.batch_size,))
                all_indices.append(indices)
            yield [el for lst in all_indices for el in lst]

    def __len__(self):
        return int(1e10)


if __name__ == "__main__":
    import torchvision
    dataset = torchvision.datasets.MNIST(
        '../data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    subsets = [[0, 1, 2],
               [0, 0, 0],
               [1]]
    batch_sampler = MultiSubsetSampler(dataset, subsets)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)
