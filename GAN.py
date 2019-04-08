import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
if torch.cuda.is_available():
    loadmap = {'cuda:0': 'gpu'}
else:
    loadmap = {'cuda:0': 'cpu'}

import matplotlib.pyplot as plt
from utils import get_data, generate_batch


def GAN(serie, window, Generator, Discriminator , generator_args, discriminator_args,
        TRAIN_RATIO=10, N_ITER=40001, BATCHLEN=128,
        frame=1000, is_notebook=True, batchlen_plot=5,
        lr_G=1e-3, betas_G=(0.5, 0.9), lr_D=1e-3, betas_D=(0.5, 0.9)):
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
    G = Generator(window, **generator_args)
    solver_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=betas_G)
    D = Discriminator(window, **discriminator_args)
    solver_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=betas_D)

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
