import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        self.generator = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),

            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),

            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),

            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.generator(z)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        self.discriminator = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.discriminator(img)

        return out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            batches_done = epoch * len(dataloader) + i

            batch_size = imgs.shape[0]
            imgs = imgs.reshape(batch_size, -1)

            # Generate
            z = torch.randn(batch_size, args.latent_dim, device=device)
            generated_images = generator(z)

            # Discriminate
            D_x = discriminator(imgs)
            D_G_x = discriminator(generated_images)

            # Train generator

            target_ones = torch.ones(D_G_x.shape, device=device)

            loss_Generator = nn.functional.binary_cross_entropy(D_G_x, target_ones)
            optimizer_G.zero_grad()
            loss_Generator.backward(retain_graph=True)
            optimizer_G.step()

            # Train discriminator

            target_zeros = torch.zeros(D_G_x.shape, device=device)
            loss_Discriminator = nn.functional.binary_cross_entropy(D_x, 0.9 * target_ones) +\
                     nn.functional.binary_cross_entropy(D_G_x, target_zeros)

            optimizer_D.zero_grad()
            loss_Discriminator.backward()
            optimizer_D.step()

            # Save Images
            # -----------

            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                generated_images = generated_images.reshape(args.batch_size, 1, 28, 28)
                save_image(generated_images[:25],
                           'images/gan/{}.png'.format(batches_done),
                           nrow=5, normalize=True)



def main(use_cuda=0):
    # Create output image directory
    os.makedirs('images', exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("hi cuda")
    else:
        device = torch.device('cpu')

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    generator.to(device)
    discriminator = Discriminator()
    discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=2000,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
