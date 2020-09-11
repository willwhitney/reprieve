"""Adapted from https://github.com/pytorch/examples/tree/master/vae."""

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from . import common

DEVICE = "cuda"
BATCH_SIZE = 128
EPOCHS = 10
FLAG = "[MNIST_VAE]"

old_print = print
def print(*args, **kwargs):  # noqa: E302
    return old_print(FLAG, *args, **kwargs)


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
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


def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def build_repr(repr_dim):
    model = VAE(repr_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, epoch)

    with torch.no_grad():
        sample = torch.randn(64, model.latent_dim).to(DEVICE)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28), 'mnist_vae_sample.png',
                   normalize=True)
    model.eval()
    return common.numpy_wrap_torch(model.repr, DEVICE)
