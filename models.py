import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

# make the import work when running the main file in this package
# or when importing the whole package
try:
    from .truncnormal import trunc_normal_
except ImportError:
    from truncnormal import trunc_normal_


class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = td.Normal(torch.zeros_like(self.mu),
                                torch.ones_like(self.mu))

    @property
    def device(self):
        return self.mu.device

    @property
    def sigma(self):
        return F.softplus(self.rho)

    # @profile
    def sample(self, n=1):
        if n == 1:
            epsilon = torch.randn_like(self.mu)
        else:
            epsilon = torch.randn((n,) + self.mu.shape, device=self.device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class GaussianLinear(nn.Module):
    def __init__(self, in_features, out_features, init_sigma):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_rho = np.log(np.exp(init_sigma) - 1.0)

        # Weight parameters
        sigma_weights = 1 / np.sqrt(in_features)
        threshold = 2 * sigma_weights

        # mu_tensor = torch.Tensor(out_features, in_features).normal_(0, sigma_weights)
        mu_tensor = torch.Tensor(out_features, in_features)
        trunc_normal_(mu_tensor, 0, sigma_weights, -threshold, threshold)
        self.weight_mu = nn.Parameter(mu_tensor)
        self.weight_rho = nn.Parameter(
            torch.ones_like(self.weight_mu) * init_rho)
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.ones_like(self.bias_mu) * init_rho)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

    def forward(self, x):
        weight = self.weight.sample()
        bias = self.bias.sample()

        return F.linear(x, weight, bias)

    @property
    def device(self):
        return self.weight.device

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight.normal.to(*args, **kwargs)
        self.bias.normal.to(*args, **kwargs)

    def get_means(self):
        return [self.weight.mu, self.bias.mu]

    def get_sigmas(self):
        return [self.weight.sigma, self.bias.sigma]


# class InitPriorGaussianLinear(GaussianLinear):
#     def __init__(self, in_features, out_features):
#         super().__init__(in_features, out_features, 1.0)
#         self.in_features = in_features
#         self.out_features = out_features

#         # Weight parameters
#         sigma_weights = 1 / np.sqrt(in_features)
#         threshold = 2 * sigma_weights
#         init_rho = np.log(np.exp(sigma_weights) - 1.0)

#         # mu_tensor = torch.Tensor(out_features, in_features).normal_(0, sigma_weights)
#         mu_tensor = torch.zeros(out_features, in_features)
#         trunc_normal_(mu_tensor, 0, sigma_weights, -threshold, threshold)
#         self.weight_mu = nn.Parameter(mu_tensor)
#         self.weight_rho = nn.Parameter(torch.ones_like(self.weight_mu) * init_rho)
#         self.weight = Gaussian(self.weight_mu, self.weight_rho)

#         # Bias parameters
#         self.bias_mu = nn.Parameter(torch.zeros(out_features))
#         self.bias_rho = nn.Parameter(torch.ones_like(self.bias_mu) * init_rho)
#         self.bias = Gaussian(self.bias_mu, self.bias_rho)


class PerSampleGaussianLinear(nn.Module):
    def __init__(self, in_features, out_features, init_sigma):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_rho = np.log(np.exp(init_sigma) - 1.0)

        # Weight parameters
        sigma_weights = 1 / np.sqrt(in_features)
        threshold = 2 * sigma_weights
        mu_tensor = torch.Tensor(out_features, in_features)
        trunc_normal_(mu_tensor, 0, sigma_weights, -threshold, threshold)
        self.weight_mu = nn.Parameter(mu_tensor)
        self.weight_rho = nn.Parameter(torch.ones_like(self.weight_mu) * init_rho)
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.ones_like(self.bias_mu) * init_rho)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

    # @profile
    def forward(self, x):
        bsize = x.shape[0]
        weight = self.weight.sample(bsize)
        bias = self.bias.sample(bsize)
        return torch.baddbmm(bias.view(bsize, -1, 1), weight, x.view(bsize, -1, 1)).view(bsize, -1)

    @property
    def device(self):
        return self.weight.device

    def get_means(self):
        return [self.weight.mu, self.bias.mu]

    def get_sigmas(self):
        return [self.weight.sigma, self.bias.sigma]


class NoisyNet(nn.Module):
    def __init__(self, init_sigma=3e-2, clipping='tanh', per_sample=False, init_prior=False,
                 PMIN=1e-3, n_inputs=784, n_outputs=10):
        super().__init__()
        self.clipping = clipping
        self.PMIN = PMIN
        if per_sample:
            self.l1 = PerSampleGaussianLinear(n_inputs, 600, init_sigma)
            self.l2 = PerSampleGaussianLinear(600, 600, init_sigma)
            self.l3 = PerSampleGaussianLinear(600, 600, init_sigma)
            self.lout = PerSampleGaussianLinear(600, n_outputs, init_sigma)
        # elif init_prior:
        #     self.l1 = InitPriorGaussianLinear(n_inputs, 600)
        #     self.l2 = InitPriorGaussianLinear(600, 600)
        #     self.l3 = InitPriorGaussianLinear(600, 600)
        #     self.lout = InitPriorGaussianLinear(600, n_outputs)
        else:
            self.l1 = GaussianLinear(n_inputs, 600, init_sigma)
            self.l2 = GaussianLinear(600, 600, init_sigma)
            self.l3 = GaussianLinear(600, 600, init_sigma)
            self.lout = GaussianLinear(600, n_outputs, init_sigma)

    def output_transform(self, x):
        # lower bound output prob
        if self.clipping == 'relu':
            x = F.relu(x + 10) - 10
        elif self.clipping == 'tanh':
            x = F.tanh(x / 6) * 6
        elif self.clipping == 'hard':
            pass
        elif self.clipping == 'none':
            pass
        else:
            assert False

        output = F.log_softmax(x, dim=1)
        if self.clipping == 'hard':
            output = torch.clamp(output, np.log(self.PMIN))
        return output

    def forward(self, x):
        x = torch.flatten(x, 1, -1)

        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)

        x = self.lout(x)
        return self.output_transform(x)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_means(self):
        return [m for child in self.children() for m in child.get_means() if hasattr(child, 'get_means')]

    def get_sigmas(self):
        return [s for child in self.children() for s in child.get_sigmas() if hasattr(child, 'get_sigmas')]


class SmallNoisyNet(NoisyNet):
    def forward(self, x):
        x = torch.flatten(x, 1, -1)

        x = self.l1(x)
        x = F.relu(x)
        x = self.lout(x)
        return self.output_transform(x)


class TinyNoisyNet(NoisyNet):
    def __init__(self, init_sigma, clipping='tanh', per_sample=False,
                 PMIN=1e-3, n_inputs=784, n_outputs=10):
        super().__init__(init_sigma, clipping, per_sample, PMIN, n_inputs, n_outputs)
        if per_sample:
            self.l1 = PerSampleGaussianLinear(n_inputs, 10, init_sigma)
            self.lout = PerSampleGaussianLinear(10, n_outputs, init_sigma)
        else:
            self.l1 = GaussianLinear(n_inputs, 10, init_sigma)
            self.lout = GaussianLinear(10, n_outputs, init_sigma)

    def forward(self, x):
        x = torch.flatten(x, 1, -1)

        x = self.l1(x)
        x = F.relu(x)
        x = self.lout(x)
        return self.output_transform(x)


class SmallRegressionNoisyNet(NoisyNet):
    def forward(self, x):
        x = torch.flatten(x, 1, -1)

        x = self.l1(x)
        x = F.relu(x)
        return self.lout(x)


class Lagrangian(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        if init_value > 1:
            init_param = init_value
        else:
            init_param = math.log(math.exp(init_value + 1e-5) - 1)

        self.lam = nn.Parameter(torch.tensor(init_param).float())

    def forward(self, *args):
        return F.softplus(self.lam)


class SupervisedReprNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.repr = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(1600, 784)
        )
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(784, 10)

    def forward(self, x):
        x = F.relu(self.repr(x))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    @property
    def device(self):
        return next(self.parameters()).device


class BottleneckSupervisedReprNet(nn.Module):
    def __init__(self, bottleneck_width=2):
        super().__init__()
        self.repr = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(1600, bottleneck_width),
        )
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(bottleneck_width, 10)

    def forward(self, x):
        x = F.relu(self.repr(x))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    @property
    def device(self):
        return next(self.parameters()).device


class LevelBottleneckSupervisedReprNet(nn.Module):
    def __init__(self, level, bottleneck_width):
        super().__init__()
        self.blocks = [
            [nn.Conv2d(1, 16, 3, 1),
             nn.ReLU(),
             nn.MaxPool2d(2), ],
            [nn.Conv2d(16, 32, 3, 1),
             nn.ReLU(),
             nn.MaxPool2d(2), ],
            [nn.Flatten(),
             nn.Linear(800, bottleneck_width),
             nn.ReLU(), ],
        ]
        # import ipdb; ipdb.set_trace()
        self.repr = nn.Sequential(*[l for block in self.blocks[:level] for l in block])

        self.head = nn.Sequential(
            *[l for block in self.blocks[level:] for l in block],
            nn.Linear(bottleneck_width, 10),
        )

    def forward(self, x):
        x = self.repr(x)
        x = self.head(x)
        output = F.log_softmax(x, dim=1)
        return output

    @property
    def device(self):
        return next(self.parameters()).device


class LinearNet(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super().__init__()
        self.repr = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        x = self.repr(x)
        output = F.log_softmax(x, dim=1)
        return output

    @property
    def device(self):
        return next(self.parameters()).device


class DeepPredictorNet(nn.Module):
    def __init__(self, n_inputs=784, n_outputs=10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, n_outputs))

    def forward(self, x):
        return F.log_softmax(self.body(x), dim=1)

    @property
    def device(self):
        return next(self.parameters()).device


class SmallReprNet(nn.Module):
    def __init__(self, n_inputs=784, n_outputs=784):
        super().__init__()
        hidden_dim = max(n_outputs, 600)
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, x):
        return self.body(x)

    @property
    def device(self):
        return next(self.parameters()).device
