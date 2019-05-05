import numpy as np
import torch
from Helpers import utils

import matplotlib.pyplot as plt
from copy import copy
from time import time

if torch.cuda.is_available():
    loadmap = {'cuda:0': 'gpu'}
else:
    loadmap = {'cuda:0': 'cpu'}


def GAN(Generator, Discriminator , generator_args, discriminator_args,
        generate_batch, param_gen_batch,
        TRAIN_RATIO=10, N_ITER=40001, BATCHLEN=128,
        frame=1000, frame_plot=1000, is_notebook=True, batchlen_plot=5,
        lr_G=1e-3, betas_G=(0.5, 0.9), lr_D=1e-3, betas_D=(0.5, 0.9),
        loss=utils.softplus_loss, argloss_real=-1, argloss_fake=1, argloss_gen=1,
        save_model=False, save_name='model', tol=1e-6, plot=True, time_max=3000):
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
    t0 = time()
    diff_loss_mean = 1
    prev_disc_loss, prev_gen_loss = 1, 1

    # if is_notebook:
    #     from tqdm import tqdm_notebook as tqdm
    # else:
    #     from tqdm import tqdm
    T = param_gen_batch['T']
    G = Generator(T, **generator_args)
    solver_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=betas_G)
    D = Discriminator(T, **discriminator_args)
    solver_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=betas_D)
    for i in range(N_ITER):  #tqdm(range(N_ITER)):
        # train the discriminator
        for _ in range(TRAIN_RATIO):
            D.zero_grad()
            real_batch = generate_batch(**param_gen_batch)
            fake_batch = G.generate(BATCHLEN)
            h_real = D(real_batch)
            h_fake = D(fake_batch)
            loss_real = loss(h_real, argloss_real)
            loss_fake = loss(h_fake, argloss_fake)
            disc_loss = loss_real + loss_fake
            disc_loss.backward()
            solver_D.step()

        # train the generator
        G.zero_grad()
        fake_batch = G.generate(BATCHLEN)
        # Compute here the generator loss, using fake_batch
        h_fake = D(fake_batch)
        gen_loss = - loss(h_fake, argloss_gen)
        gen_loss.backward()
        solver_G.step()

        diff_loss_mean = 0.9 * diff_loss_mean +\
                         0.1 * (torch.abs(disc_loss - prev_disc_loss) + torch.abs(gen_loss - prev_gen_loss))
        prev_disc_loss, prev_gen_loss = copy(disc_loss), copy(gen_loss)

        if diff_loss_mean < tol or time()-t0 > time_max:
            if save_model:
                torch.save(G.state_dict(), 'Generator/'+save_name+'.pth')
                torch.save(D.state_dict(), 'Discriminator/'+save_name+'.pth')
            return None

        if i % frame == 0:
            print('step {}: discriminator: {:.3e}, generator: {:.3e}'.format(i, float(disc_loss), float(gen_loss)))
            if save_model:
                torch.save(G.state_dict(), 'Generator/'+save_name+'.pth')
                torch.save(D.state_dict(), 'Discriminator/'+save_name+'.pth')

        if plot and i % frame_plot == 0:
            # plot the result
            real_batch = generate_batch(**param_gen_batch)
            fake_batch = G.generate(batchlen_plot).detach()
            fig, axs = plt.subplots(2)
            fig.suptitle('Real Batch vs Generated Batch')
            axs[0].plot(real_batch.numpy().T)
            axs[1].plot(fake_batch.numpy().T)
            plt.show()
    return G, D

