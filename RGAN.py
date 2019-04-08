from GAN import *


# Define the generator
class Generator(nn.Module):
    def __init__(self, window, WDTH=0, PRIOR_N=1, DPTH=0, PRIOR_STD=1.):
        super().__init__()
        self.PRIOR_N = PRIOR_N
        self.PRIOR_STD = PRIOR_STD
        self.fc1 = nn.Linear(PRIOR_N, WDTH)
        self.hidden_layers = []
        for _ in range(DPTH):
            self.hidden_layers.append(nn.Linear(WDTH, WDTH))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.fc2 = nn.Linear(WDTH, window)

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
        self.fc1 = nn.Linear(window, WDTH)
        self.hidden_layers = []
        for _ in range(DPTH):
            self.hidden_layers.append(nn.Linear(WDTH, WDTH))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.fc2 = nn.Linear(WDTH, 1)

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        for hidden_layer in self.hidden_layers:
            h = F.relu(hidden_layer(h))
        return self.fc2(h)


if __name__ == '__main__':
    param = {
        'serie': get_data('VIX.csv'),
        'window': 125,
        'frame': 200,
        'is_notebook': False,
        'batchlen_plot': 5,
        'Generator': Generator,
        'Discriminator': Discriminator
    }
    training_param = {
        'N_ITER': 601,
        'TRAIN_RATIO': 10,
        'BATCHLEN': 30,
        # Random Noise used by the Generator
        'PRIOR_N': 20,
        'PRIOR_STD': 500.,
        # Depth and Withdraw of Hidden Layers
        'WDTH_G': 100,
        'DPTH_G': 1,
        'WDTH_D': 100,
        'DPTH_D': 3,
        # Adam Optimizer parameters for G/D
        'lr_G': 1e-4,
        'betas_G': (0.5, 0.9),
        'lr_D': 1e-4,
        'betas_D': (0.5, 0.9)
    }

    param.update(training_param)
    GAN(**param)