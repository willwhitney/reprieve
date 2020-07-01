import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision

from models import LevelBottleneckSupervisedReprNet
from dataset_utils import DatasetCache

FLAG = "[FMNIST_SUPER]"
old_print = print


def print(*args, **kwargs):
    return old_print(FLAG, *args, **kwargs)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
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
    batch_size = 256

    transform = transforms.Compose([
        # transforms.CenterCrop(28),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.2857,), (0.3530,))])

    train_data = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                   download=True, transform=transform)
    train_data = DatasetCache(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=0)

    test_data = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                  download=True, transform=transform)
    test_data = DatasetCache(test_data)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, num_workers=0)

    repr_model = LevelBottleneckSupervisedReprNet(
        bottleneck_width=repr_dim, level=repr_level).to(device)
    optimizer = optim.Adam(repr_model.parameters(), lr=1e-3)

    for epoch in range(1, 50):
        train(repr_model, train_loader, optimizer, epoch)
        test(repr_model, test_loader)

    repr_model.eval()
    return repr_model.repr


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cifar_repr = build_repr(DEVICE)
