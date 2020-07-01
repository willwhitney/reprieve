import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

import mnist_dataset

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args([
    "--epochs", "50",
])
args.cuda = not args.no_cuda and torch.cuda.is_available()
FLAG = "[MNIST_VAE]"
old_print = print


def print(*args, **kwargs):
    return old_print(FLAG, *args, **kwargs)


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        # bottleneck_width = 30
        self.encoder_layers = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
        )
        self.fc21 = nn.Linear(400, self.latent_dim)
        self.fc22 = nn.Linear(400, self.latent_dim)
        # self.fc3 = nn.Linear(self.latent_dim, 400)
        # self.fc4 = nn.Linear(400, 784)

        self.decoder_layers = nn.Sequential(
            nn.Linear(self.latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
        )

    def encode(self, x):
        x = self.encoder_layers(x)
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return torch.sigmoid(self.decoder_layers(z))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @property
    def repr(self):
        return nn.Sequential(
            nn.Flatten(),
            self.encoder_layers,
            self.fc21
        )


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(model, train_loader, optimizer, epoch, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))

    print('Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(model, test_loader, optimizer, epoch, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('Test set loss: {:.4f}'.format(test_loss))


def build_repr(device, repr_dim):
    model = VAE(repr_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = torch.utils.data.DataLoader(
        mnist_dataset.MNIST('../data', train=True, download=True, n_examples=60000,
                            transform=transforms.Compose([
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # test_loader = torch.utils.data.DataLoader(
    #     mnist_dataset.MNIST('../data', train=False, download=True, n_examples=10000,
    #                         transform=transforms.Compose([
    #                             transforms.Normalize((0.1307,), (0.3081,))
    #                         ])),
    #     batch_size=args.batch_size, shuffle=True, pin_memory=True)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch, device)

    with torch.no_grad():
        sample = torch.randn(64, model.latent_dim).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28), 'mnist_vae_sample.png')
    model.eval()
    return model.repr
