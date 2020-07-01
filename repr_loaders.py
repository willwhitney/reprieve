import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import mnist_dataset
from dataset_utils import DatasetSubset, DatasetUnion, DatasetWhiten, DatasetCache, DatasetShuffle


def make_repr(repr_type, DEVICE, repr_dim, repr_level):
    if repr_type == "raw":
        def repr_fn(x):
            return x
    else:
        if repr_type == 'mnist_supervised':
            import mnist_supervised_repr
            repr_fn = mnist_supervised_repr.build_repr(DEVICE, repr_dim, repr_level)
        if repr_type == 'mnist_binsup':
            import mnist_binsup_repr
            repr_fn = mnist_binsup_repr.build_repr(DEVICE, repr_dim, repr_level)
        elif repr_type == 'cifar_supervised':
            import cifar_supervised_repr
            repr_fn = cifar_supervised_repr.build_repr(DEVICE, repr_dim, repr_level)
        elif repr_type == 'fashionmnist_supervised':
            import fashionmnist_supervised_repr
            repr_fn = fashionmnist_supervised_repr.build_repr(DEVICE, repr_dim, repr_level)
        elif repr_type == 'mnist_vae':
            import mnist_vae_repr
            repr_fn = mnist_vae_repr.build_repr(DEVICE, repr_dim)
        repr_fn.cpu()
    return repr_fn


def get_mnist_loaders(repr_type, DEVICE, repr_dim, repr_level,
                      ntrain=50000, nval=10000, ntest=10000, batch_size=256):
    repr_fn = make_repr(repr_type, DEVICE, repr_dim, repr_level)
    if repr_type == 'raw':
        unwhite_mean = 0
        unwhite_std = 1
    else:
        unwhite_data = mnist_dataset.MNIST(
            '../data', train=True, download=True, n_examples=ntrain,
            transform=transforms.Compose([
                transforms.Normalize((0.1307,), (0.3081,)),
            ]))
        all_unwhite_x = torch.stack([unwhite_data[i][0]
                                    for i in range(min(len(unwhite_data), 10000))])
        all_unwhite_repr = repr_fn(all_unwhite_x)
        unwhite_mean = all_unwhite_repr.mean()
        unwhite_std = all_unwhite_repr.std()
        repr_fn.cpu()

    def repr_transform(x): return repr_fn(x.unsqueeze(0))
    kwargs = {'num_workers': 0, 'pin_memory': True} if DEVICE.type == 'cuda' else {}
    all_train_data = mnist_dataset.MNIST(
        '../data', train=True, download=True, n_examples=60000,
        transform=transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(repr_transform),
            transforms.Lambda(lambda x: (x - unwhite_mean) / unwhite_std)
        ]))

    train_loader = torch.utils.data.DataLoader(
        DatasetSubset(all_train_data, start=0, stop=ntrain),
        batch_size=batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        DatasetSubset(all_train_data, start=ntrain, stop=ntrain + nval),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(mnist_dataset.MNIST(
        '../data', train=False, download=True, n_examples=ntest,
        transform=transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(repr_transform),
            transforms.Lambda(lambda x: (x - unwhite_mean) / unwhite_std)
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader, test_loader


def get_svhn_loaders(repr_type, DEVICE, ntrain=578388,
                     nval=26032, ntest=26032, batch_size=256):
    repr_fn = make_repr(repr_type, DEVICE)
    def repr_transform(x): return repr_fn(x.unsqueeze(0))
    transform = transforms.Compose([
        transforms.CenterCrop(28),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.1993,)),
        transforms.Lambda(repr_transform)])
    train_data = torchvision.datasets.SVHN(
        'svhn-data',
        split='train',
        download=True,
        transform=transform)
    extra_data = torchvision.datasets.SVHN(
        'svhn-data',
        split='extra',
        download=True,
        transform=transform)
    test_data = torchvision.datasets.SVHN(
        'svhn-data',
        split='test',
        download=True,
        transform=transform)

    all_train_data = DatasetShuffle(DatasetUnion([train_data, extra_data]))
    # all_train_data = DatasetShuffle(DatasetUnion([train_data]))
    all_train_data.classes = 10
    test_data.classes = 10

    train_data = DatasetSubset(all_train_data, start=0, stop=ntrain)
    val_data = DatasetSubset(all_train_data, start=ntrain, stop=ntrain + nval)
    test_data = DatasetSubset(test_data, start=0, stop=ntest)

    train_data = DatasetCache(train_data)
    val_data = DatasetCache(val_data)
    test_data = DatasetCache(test_data)
    train_data = DatasetWhiten(train_data)
    val_data = DatasetWhiten(val_data)
    test_data = DatasetWhiten(test_data)

    kwargs = {'num_workers': 0, 'pin_memory': True} if DEVICE.type == 'cuda' else {}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, val_loader, test_loader
