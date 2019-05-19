import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from datasets.bmnist import bmnist
import matplotlib
from scipy.stats import norm

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        input_dim = 28 * 28
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.std = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        h = self.fc1(input)
        h = self.relu(h)
        mean = self.mean(h)
        std = self.std(h)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        input_dim = 28 * 28
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(z_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        """
        Perform forward pass of decoder.

        Returns mean with shape [batch_size, 784].
        """

        a = self.fc2(z)
        b = self.tanh(a)
        c = self.fc3(b)

        y = self.sigmoid(c)

        return y


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, std = self.encoder(input)

        e = torch.zeros(mean.shape).normal_()
        z = std * e + mean

        y = self.decoder(z)

        eps = 1e-8
        L_reconstruction = input * y.log() + (1 - input) * (1 - y).log()
        KLD = 0.5 * (std.pow(2) + mean.pow(2) - 1 - torch.log(std.pow(2)+eps))
        elbo = KLD.sum(dim=-1) - L_reconstruction.sum(dim=-1)
        average_negative_elbo = elbo.mean()

        return average_negative_elbo


    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        samples = torch.randn((n_samples, self.z_dim))
        y = self.decoder(samples)

        im_means = y.reshape(n_samples, 1, 28, 28)
        sampled_ims = torch.bernoulli(im_means)

        return sampled_ims, im_means

    def sample_manifold(self, frequency=10):
        n = frequency
        X = np.linspace(0, 1, n + 2)
        Y = np.linspace(0, 1, n + 2)

        ppfX = norm.ppf(X[1:-1])
        ppfY = norm.ppf(Y[1:-1])

        zs = [[x, y] for x in ppfX for y in ppfY]
        zs = torch.tensor(zs)

        out = self.decoder(zs)
        im_means = out.reshape(n * n, 1, 28, 28)
        return im_means

def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0
    size = len(data)

    for sample in data:
        # get the average elbo of the batch
        input = sample.reshape(sample.shape[0], -1)

        elbo = model.forward(input)
        average_epoch_elbo -= elbo

        # train the model
        if model.training:
            model.zero_grad()
            elbo.backward()
            optimizer.step()

    average_epoch_elbo /= size

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def save_sample(sample, imw, epoch, nrow=8, slug='sample'):
    sample = sample.view(-1, 1, imw, imw)
    sample = make_grid(sample, nrow=nrow).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"images/vae_{slug}_{epoch}.png", sample)

def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())
    size_width = 28

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

        _, mean_sample = model.sample(64)
        save_sample(mean_sample, size_width, epoch)

    if ARGS.zdim == 2:
        manifold = model.manifold_sample(100)
        save_sample(manifold, size_width, epoch, 10, 'manifold')

    np.save('curves.npy', {'train': train_curve, 'val': val_curve})
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
