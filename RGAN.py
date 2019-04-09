from GAN import *


# Define the generator
class Generator(nn.Module):
    def __init__(self, window=60, PRIOR_N=10, nlayers=2):
        super().__init__()
        self.window = window
        self.PRIOR_N = PRIOR_N
        self.nlayers = nlayers
        self.rnn = nn.LSTM(PRIOR_N, window, nlayers)

    def __call__(self, batchlen, input, hx, cx):
        output, (hx, cx) = self.rnn(input, (hx, cx))
        return output[:, 0, :]

    def generate(self, batchlen):
        input = torch.randn(batchlen, 1, self.PRIOR_N)
        hx = torch.randn(self.nlayers, 1, self.window)
        cx = torch.randn(self.nlayers, 1, self.window)
        return self.__call__(batchlen, input, hx, cx)


# Define the discriminator.
class Discriminator(nn.Module):
    def __init__(self, window, nlayers):
        super().__init__()
        self.nlayers = nlayers
        self.rnn = nn.LSTM(window, window, nlayers)

    def __call__(self, x):
        input = x.unsqueeze(1)
        output, (hx, cx) = self.rnn(input)
        return torch.sigmoid(output[:, 0, :])


if __name__ == '__main__':
    param = {
        'serie': get_data('VIX.csv'),
        'window': 60,
        'frame': 2000,
        'is_notebook': False,
        'batchlen_plot': 5,
        'Generator': Generator,
        'Discriminator': Discriminator,
        'BATCHLEN': 30
    }
    training_param = {
        'N_ITER': 2001,
        'TRAIN_RATIO': 5,
        # Random Noise used by the Generator
        'generator_args': {
            'PRIOR_N': 1,
            'nlayers': 1,
        },
        # Depth and Withdraw of Hidden Layers
        'discriminator_args': {
        'nlayers': 1},
        # Adam Optimizer parameters for G/D
        'lr_G': 1e-4,
        'betas_G': (0.5, 0.9),
        'lr_D': 1e-4,
        'betas_D': (0.5, 0.9),
        'loss': utils.negative_cross_entropy,
        'argloss_real': torch.ones(param['BATCHLEN'], dtype=torch.int64),
        'argloss_fake': torch.zeros(param['BATCHLEN'], dtype=torch.int64),
        'argloss_gen': torch.ones(param['BATCHLEN'], dtype=torch.int64)
    }

    param.update(training_param)
    GAN(**param)

