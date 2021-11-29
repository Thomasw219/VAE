import torch
import torchvision

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributions.normal as Normal
import torch.distributions.kl as KL
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2))

        self.encoder_linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024, 64),
                nn.ReLU())

        self.encoder_mean = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.latent_dim))

        self.encoder_log_var = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.latent_dim))

        # Decoder
        self.decoder_linear = nn.Sequential(
                nn.Linear(self.latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1024),
                nn.ReLU())

        self.decoder_conv = nn.Sequential(
                nn.Upsample(scale_factor=(2, 2)),
                nn.ConvTranspose2d(64, 64, 3),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3),
                nn.ReLU(),
                nn.Upsample(scale_factor=(2, 2)),
                nn.ConvTranspose2d(32, 32, 3),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 3))

    def forward_encoder(self, x):
        features = self.encoder_conv(x)
        condensed_features = self.encoder_linear(features)
        means = self.encoder_mean(condensed_features)
        log_vars = self.encoder_log_var(condensed_features)
        return means, log_vars

    def forward_decoder(self, x):
        features = self.decoder_linear(x)
        reconstructed_images = self.decoder_conv(features.view((-1, 64, 4, 4)))
        return reconstructed_images

    def get_loss(self, x):
        batch_size = torch.tensor(x.size()[0], device=x.device)
        device = x.device
        means, log_vars = self.forward_encoder(x)
        stds = torch.exp(0.5 * log_vars)
        encoder_dist = Normal.Normal(means, stds)
        prior_dist = Normal.Normal(torch.zeros_like(means), torch.ones_like(stds))
        kl_loss = torch.sum(KL.kl_divergence(encoder_dist, prior_dist)) / batch_size
        noise = torch.normal(torch.zeros_like(means), torch.ones_like(stds))
        latent_points = means + stds * noise
        reconstructed_images = self.forward_decoder(latent_points)
        reconstruction_loss = torch.sum(torch.square(reconstructed_images - x)) / batch_size
        loss = kl_loss + reconstruction_loss
        return loss, kl_loss, reconstruction_loss

    def forward(self, x):
        means, log_vars = self.forward_encoder(x)
        stds = torch.exp(0.5 * log_vars)
        noise = torch.normal(torch.zeros_like(means), torch.ones_like(stds))
        latent_points = means + stds * noise
        reconstructed_images = self.forward_decoder(latent_points)
        return reconstructed_images

import matplotlib.pyplot as plt
def save_original_and_reconstruction(image, vae, update):
    reconstruction = vae.forward(image)
    plt.imshow(image.cpu().detach().numpy().reshape((28, 28)))
    plt.savefig("../figures/vae_train_{:04d}_original.png".format(update))
    plt.imshow(np.clip(reconstruction.cpu().detach().numpy().reshape((28, 28)), 0, 1))
    plt.savefig("../figures/vae_train_{:04d}_reconstruction.png".format(update))

if __name__ == "__main__":
    embedding_dim = 3
    mnist_trainset = datasets.MNIST(root="../figures", train=True,
            download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(root="../figures", train=False,
            download=True, transform=transforms.ToTensor())

    """
    device = torch.device("cuda:0")
    vae = VAE(embedding_dim)
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=0.0001)

    train_dataloader = DataLoader(mnist_trainset, batch_size=512, shuffle=True, pin_memory=True)

    n_epochs = 6

    update = 0
    for epoch in range(n_epochs):
        print("Epoch: {}".format(epoch))
        for train_images, _ in iter(train_dataloader):
            train_images = train_images.to(device)
            if update % 25 == 0:
                print("Saving images")
                image = train_images[0:1]
                save_original_and_reconstruction(image, vae, update)

            loss, kl_loss, reconstruction_loss = vae.get_loss(train_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Loss: {}, KL Loss: {}, Reconstruction Loss: {}".format(loss.cpu().item(), kl_loss.cpu().item(), reconstruction_loss.cpu().item()))

            update += 1

    torch.save(vae.state_dict(), "./vae_state_dict.pt")

    """

    vae = VAE(embedding_dim)
    vae.load_state_dict(torch.load("./vae_state_dict.pt"))
    test_dataloader = DataLoader(mnist_testset, batch_size=1, shuffle=True)
    image_embeddings = [[] for i in range(10)]

    """
    n = 0
    for test_images, test_labels in iter(test_dataloader):
        with torch.no_grad():
            embedding, _ = vae.forward_encoder(test_images)
        label = test_labels.item()
        numpy_embedding = embedding.cpu().detach().numpy().reshape((embedding_dim,))
        image_embeddings[label].append(numpy_embedding)
        n += 1
        if n >= 3000:
            break

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i, embeddings in enumerate(image_embeddings):
        embeddings = np.array(embeddings)
        print(i, np.mean(embeddings, axis=0))
        ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2], label=str(i))

    plt.legend()
    plt.savefig("../figures/vae_embedding_space.png")

    """

    p1 = np.array([-0.7435723, 0.19734043, 1.297068])
    p2 = np.array([1.2224902, -0.4694287, -0.36215362])

    for i, a in enumerate(np.linspace(0, 1.2, num=50)):
        embedding = p1 * (1 - a) + p2 * a
        with torch.no_grad():
            image = vae.forward_decoder(torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)).cpu().numpy().reshape((28, 28))
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(image)
            fig.savefig("../figures/vae_traversal_{:02}.png".format(i))
            plt.close(fig)

    """
    k = 0
    for test_images, test_labels in iter(test_dataloader):
        plt.imshow(np.clip(test_images.cpu().detach().numpy(), 0, 1).reshape((28, 28)))
        plt.savefig("../figures/vae_test_{:02}_original.png".format(k))
        with torch.no_grad():
            image = vae.forward(test_images)
        plt.imshow(np.clip(image.cpu().detach().numpy(), 0, 1).reshape((28, 28)))
        plt.savefig("../figures/vae_test_{:02}_reconstruction.png".format(k))
        k += 1
        if k >= 50:
            break
    """
