from GAN import *


# Define the generator
class Generator(nn.Module):
    def __init__(self, window, WDTH=0, PRIOR_N=10, PRIOR_CHANNEL=20,
                 START=126, NB_UPSCALE=0, PRIOR_STD=1., DPTH=0):
        super().__init__()
        self.PRIOR_N = PRIOR_N
        self.PRIOR_STD = PRIOR_STD
        self.PRIOR_CHANNEL = PRIOR_CHANNEL
        self.window = window
        self.main = nn.Sequential(
        nn.ConvTranspose1d(PRIOR_CHANNEL, 8*window, 4, 1, 0, bias=False),  #Kernel size/stride/padding
        nn.BatchNorm1d(8*window),
        nn.ReLU(True),
        nn.ConvTranspose1d(8*window, 4*window, 4, 2, 1, bias=False),
        nn.BatchNorm1d(4*window),
        nn.ReLU(True),
        nn.ConvTranspose1d(4*window, 2*window, 4, 2, 1, bias=False),
        nn.BatchNorm1d(2*window),
        nn.ReLU(True),
        nn.ConvTranspose1d(2*window, window, 4, 2, 1, bias=False),
        nn.BatchNorm1d(window),
        nn.ReLU(True),
        nn.ConvTranspose1d(window, 1, 4, 2, 1, bias=False),
        nn.Tanh())

    def generate(self, batchlen):
        z = torch.normal(torch.zeros(batchlen, self.PRIOR_CHANNEL, self.PRIOR_N), self.PRIOR_STD)
        z = self.main(z)
        return z[:, 0, :self.window]


# Define the discriminator.
class Discriminator(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.main = nn.Sequential(
        nn.BatchNorm1d(1),  # I add this layer to normalize the real data inputs
        nn.Conv1d(1, window, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(window, 2*window, 4, 2, 1, bias=False),
        nn.BatchNorm1d(2*window),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(2*window, 4*window, 4, 2, 1, bias=False),
        nn.BatchNorm1d(4*window),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(4*window, 8*window, 4, 2, 1, bias=False),
        nn.BatchNorm1d(8*window),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(8*window, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
        )

    def __call__(self, x):
        x = x.unsqueeze(1)
        return self.main(x)[:, 0, :]


if __name__ == '__main__':
    param = {
        'serie': get_data('VIX.csv'),
        'window': 125,
        'frame': 10,
        'is_notebook': False,
        'batchlen_plot': 1,
        'Generator': Generator,
        'Discriminator': Discriminator
    }
    training_param = {
        'N_ITER': 101,
        'TRAIN_RATIO': 10,
        'BATCHLEN': 30,
        # Random Noise used by the Generator
        'generator_args': {
        'PRIOR_N': 5,
        'PRIOR_STD': 1.,
        'WDTH': 100,
        'DPTH': 1,
        'PRIOR_CHANNEL': 3,
        'START': 60,
        'NB_UPSCALE': 0,
        },
        'discriminator_args': {},
        # Adam Optimizer parameters for G/D
        'lr_G': 1e-4,
        'betas_G': (0.5, 0.9),
        'lr_D': 1e-4,
        'betas_D': (0.5, 0.9)
    }

    param.update(training_param)
    GAN(**param)