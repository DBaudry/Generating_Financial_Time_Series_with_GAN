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
    def __init__(self, input_layer, hidden_layers, output_layer,
                DPTH=0, PRIOR_N=1, PRIOR_STD=1.):
        super().__init__()
        self.PRIOR_N = PRIOR_N
        self.PRIOR_STD = PRIOR_STD
        self.L_in = input_layer
        self.hidden_layers = []
        for _ in range(DPTH):
            self.hidden_layers.append(hidden_layers)
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.L_out = output_layer

    def __call__(self, z):
        h = F.relu(self.L_in(z))
        for hidden_layer in self.hidden_layers:
            h = F.relu(hidden_layer(h))
        return self.L_out(h)

    def generate(self, batchlen):
        z = torch.normal(torch.zeros(batchlen, self.PRIOR_N), self.PRIOR_STD)
        return self.__call__(z)


class Discriminator(nn.Module):
    def __init__(self, input_layer, hidden_layers, output_layer, DPTH=1):
        super().__init__()
        self.L_in = input_layer
        self.hidden_layers = []
        for _ in range(DPTH):
            self.hidden_layers.append(hidden_layers)
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.L_out = output_layer

    def __call__(self, x):
        h = F.relu(self.L_in(x))
        for hidden_layer in self.hidden_layers:
            h = F.relu(hidden_layer(h))
        return self.L_out(h)


def GAN(serie, window,
        input_layer_G, hidden_layers_G, output_layer_G,
        input_layer_D, hidden_layers_D, output_layer_D,
        TRAIN_RATIO=10, N_ITER=4001, BATCHLEN=200,
        WDTH_G=100, DPTH_G=3, WDTH_D=100, DPTH_D=3,
        PRIOR_N=20, PRIOR_STD=100., frame=100, is_notebook=True, batchlen_plot=5,
        lr_G=1e-3, betas_G=(0.5,0.9), lr_D=1e-3, betas_D=(0.5,0.9)):
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
    G = Generator(DPTH=DPTH_G, PRIOR_N=PRIOR_N, PRIOR_STD=PRIOR_STD,
                  input_layer=input_layer_G, hidden_layers=hidden_layers_G, output_layer=output_layer_G)
    solver_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=betas_G)
    D = Discriminator(DPTH=DPTH_D, input_layer=input_layer_D,
                      hidden_layers=hidden_layers_D, output_layer=output_layer_D)
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


if __name__ == '__main__':
    # VIX = get_data('VIX.csv', array=False)
    # VIX.plot()
    # plt.show()

    VIX = get_data('VIX.csv')
    # X = generate_batch(VIX, 100, 10)
    # plt.plot(X.numpy().T)
    # plt.show()

    param = {
        'serie': VIX,
        'window': 60,
        'frame': 100,
        'is_notebook': False,
        'batchlen_plot': 5
    }
    training_param = {
        'N_ITER': 2001,
        'TRAIN_RATIO': 10,
        'BATCHLEN': 20,
        # Random Noise used by the Generator
        'PRIOR_N': 20,
        'PRIOR_STD': 500.,
        # Depth and Withdraw of Hidden Layers
        'WDTH_G': 100,
        'DPTH_G': 1,
        'WDTH_D': 100,
        'DPTH_D': 1,
        # Adam Optimizer parameters for G/D
        'lr_G': 1e-3,
        'betas_G': (0.5, 0.9),
        'lr_D': 1e-3,
        'betas_D': (0.5, 0.9)
    }

    param.update(training_param)

    layers = {
        # Generator Layers
        'input_layer_G':  nn.Linear(param['PRIOR_N'], param['WDTH_G']),
        'hidden_layers_G': nn.Linear(param['WDTH_G'], param['WDTH_G']),
        'output_layer_G': nn.Linear(param['WDTH_G'], param['window']),
        # Discriminator Layers
        'input_layer_D':  nn.Linear(param['window'], param['WDTH_D']),
        'hidden_layers_D': nn.Linear(param['WDTH_D'], param['WDTH_D']),
        'output_layer_D': nn.Linear(param['WDTH_D'], 1)
    }

    param.update(layers)
    GAN(**param)



################ BONNES ARCHI ######################

# GAN(VIX, window=60, TRAIN_RATIO=10, N_ITER=2001, BATCHLEN=500,
#     WDTH_G=100, DPTH_G=1, WDTH_D=100, DPTH_D=1,
#     PRIOR_N=20, PRIOR_STD=500., frame=100, is_notebook=False, batchlen_plot=5,
#     input_layer_G=nn.Linear, hidden_layers_G=nn.Linear, output_layer_G=nn.Linear,
#     input_layer_D=nn.Linear, hidden_layers_D=nn.Linear, output_layer_D=nn.Linear,
#     lr_G=1e-3, betas_G=(0.5, 0.9), lr_D=1e-3, betas_D=(0.5, 0.9))