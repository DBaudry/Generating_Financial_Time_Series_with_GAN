import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.is_available():
    loadmap = {'cuda:0': 'gpu'}
else:
    loadmap = {'cuda:0': 'cpu'}

import matplotlib.pyplot as plt
from utils import get_data, generate_batch


# Define the generator
class Generator(nn.Module):
    def __init__(self, window, WDTH=0, PRIOR_N=1, DPTH=0, PRIOR_STD=1.):
        super().__init__()
        self.PRIOR_N = PRIOR_N
        self.PRIOR_STD = PRIOR_STD
        # First layer
        self.fc1 = nn.Linear(PRIOR_N, WDTH)
        # Hidden layers
        self.hidden_layers = []
        for _ in range(DPTH):
            self.hidden_layers.append(nn.Linear(WDTH, WDTH))
        # Transform list into layers using nn.ModuleList
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        # Output layer
        self.fc2 = nn.Linear(WDTH, window)  ######## Change 2 by window

    def __call__(self, z):
        h = F.relu(self.fc1(z))
        for hidden_layer in self.hidden_layers:
            h = F.relu(hidden_layer(h))
        return self.fc2(h)

    def generate(self, batchlen):
        z = torch.normal(torch.zeros(batchlen, self.PRIOR_N), self.PRIOR_STD)
        return self.__call__(z)


# Define the discriminator.
class Discriminator(nn.Module):
    def __init__(self, window, WDTH=0, DPTH=0):
        super().__init__()
        # First layer
        self.fc1 = nn.Linear(window, WDTH) ###### Change 2 by window
        # Hidden layers
        self.hidden_layers = []
        for _ in range(DPTH):
            self.hidden_layers.append(nn.Linear(WDTH, WDTH))
        # Transform list into layers using nn.ModuleList
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        # Output layer
        self.fc2 = nn.Linear(WDTH, 1)

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        for hidden_layer in self.hidden_layers:
            h = F.relu(hidden_layer(h))
        return self.fc2(h)


def GAN(serie, window, TRAIN_RATIO=1, N_ITER=40001, BATCHLEN=128,
        WDTH_G=0, DPTH_G=0, WDTH_D=0, DPTH_D=0,
        PRIOR_N=1, PRIOR_STD=1., frame=1000, is_notebook=True, batchlen_plot=5):
    """
    serie: Input Financial Time Serie
    TRAIN_RATIO : int, number of times to train the discriminator between two generator steps
    N_ITER : int, total number of training iterations for the generator
    BATCHLEN : int, Batch size to use
    WDTH_G : int, width of the generator (number of neurons in hidden layers)
    DPTH_G : int, number of hidden FC layers of the generator
    WDTH_D : int, width of the discriminator (number of neurons in hidden layers)
    DPTH_D : int, number of hidden FC layers of the discriminator
    PRIOR_N : int, dimension of input noise
    PRIOR_STD : float, standard deviation of p(z)
    frame : int, display data each 'frame' iteration
    """
    if is_notebook:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
    G = Generator(window, WDTH=WDTH_G, DPTH=DPTH_G, PRIOR_N=PRIOR_N, PRIOR_STD=PRIOR_STD)
    solver_G = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    D = Discriminator(window, WDTH=WDTH_D, DPTH=DPTH_D)
    solver_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for i in tqdm(range(N_ITER)):
        # train the discriminator
        for _ in range(TRAIN_RATIO):
            D.zero_grad()
            real_batch = generate_batch(serie, window, BATCHLEN)
            fake_batch = G.generate(BATCHLEN)
            h_real = D(real_batch)
            h_fake = D(fake_batch)
            loss_real = torch.mean(torch.sum(F.softplus(-h_real)))
            loss_fake = torch.mean(torch.sum(F.softplus(h_fake)))
            disc_loss = loss_real + loss_fake
            disc_loss.backward()
            solver_D.step()
        # train the generator
        G.zero_grad()
        fake_batch = G.generate(BATCHLEN)
        # Compute here the generator loss, using fake_batch
        h_fake = D(fake_batch)
        gen_loss = - torch.mean(torch.sum(F.softplus(h_fake)))
        gen_loss.backward()
        solver_G.step()
        if i % frame == 0:
            print('step {}: discriminator: {:.3e}, generator: {:.3e}'.format(i, float(disc_loss), float(gen_loss)))
            # plot the result
            fake_batch = G.generate(batchlen_plot).detach()
            plt.plot(fake_batch.numpy().T)
            plt.show()


if __name__ == '__main__':
    # VIX = get_data('VIX.csv', array=False)
    # VIX.plot()
    # plt.show()

    VIX = get_data('VIX.csv')
    # X = generate_batch(VIX, 100, 10)
    # plt.plot(X.numpy().T)
    # plt.show()
    GAN(VIX, 125, TRAIN_RATIO=1, N_ITER=4001, BATCHLEN=128,
        WDTH_G=8, DPTH_G=3, WDTH_D=8, DPTH_D=3, PRIOR_N=5, PRIOR_STD=1., frame=100, is_notebook=False)
