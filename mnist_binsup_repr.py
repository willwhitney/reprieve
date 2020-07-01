import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from models import LevelBottleneckSupervisedReprNet
import mnist_dataset

FLAG = "[MNIST_BINSUP]"
old_print = print


def print(*args, **kwargs):
    return old_print(FLAG, *args, **kwargs)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = (target > 4).long()
        data, target = data.to(model.device), target.to(model.device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = (target > 4).long()
            data, target = data.to(model.device), target.to(model.device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def build_repr(device, repr_dim, repr_level):
    train_n = 60000
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
        mnist_dataset.MNIST('../data', train=True, download=True, n_examples=train_n,
                            transform=transforms.Compose([
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
        batch_size=batch_size, shuffle=True, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        mnist_dataset.MNIST('../data', train=False, download=True, n_examples=10000,
                            transform=transforms.Compose([
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
        batch_size=batch_size, shuffle=True, pin_memory=True)

    repr_model = LevelBottleneckSupervisedReprNet(
        bottleneck_width=repr_dim, level=repr_level).to(device)
    optimizer = optim.Adadelta(repr_model.parameters(), lr=1.)

    for epoch in range(1, 30):
        train(repr_model, train_loader, optimizer, epoch)
        test(repr_model, test_loader)
    repr_model.eval()
    return repr_model.repr


if __name__ == "__main__":
    build_repr(torch.device('cuda'))
