import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np
import numpy.random as npr

import mnist_dataset
from dataset_utils import DatasetSubset, DatasetCache

FLAG = "[MNIST_NOISYGT]"
old_print = print


def print(*args, **kwargs):
    return old_print(FLAG, *args, **kwargs)


class MNISTNoisyGTDataset(Dataset):
    def __init__(self, split='train', ntrain=50000, nval=10000, ntest=10000, p_corrupt=0.1):
        self.p_corrupt = p_corrupt
        if split == 'train':
            train = True
            start = 0
            stop = ntrain
        elif split == 'val':
            train = True
            start = ntrain
            stop = ntrain + nval
        else:
            train = False
            start = 0
            stop = ntest

        self.data = mnist_dataset.MNIST('../data', train=train, download=True, n_examples=60000,)
        self.data = DatasetSubset(self.data, start=start, stop=stop)
        self.data = DatasetCache(self.data)
        # self.classes = self.data.classes


    def __getitem__(self, index):
        x, y = self.data[index]
        y_fake_probs = np.zeros((10,)) + self.p_corrupt / 9.
        y_fake_probs[int(y)] = 1 - self.p_corrupt
        # w/ prob 1 - p_corrupt, use right label, else uniform over wrong labels
        y_fake = npr.choice(range(10), p=y_fake_probs)

        # make every 10th element a 1, starting at the sampled y_fake
        # i.e. make a tiled one-hot vector
        x_fake = torch.zeros_like(x).flatten()
        x_fake[y_fake::10] = 1
        return x_fake, y


    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    dataset = MNISTNoisyGTDataset()
    import ipdb; ipdb.set_trace()
