from GAN import *


# Define the generator
class Generator(nn.Module):
    def __init__(self, window=60, PRIOR_N=10):
        super().__init__()
        self.window = window
        self.PRIOR_N = PRIOR_N
        self.rnn = nn.LSTMCell(PRIOR_N, window)

    def __call__(self, batchlen, input, hx, cx):
        output = torch.zeros((batchlen, self.window))
        for i in range(batchlen):
            hx, cx = self.rnn(input[i], (hx, cx))
            output[i] = hx
        return output

    def generate(self, batchlen):
        input = torch.randn(batchlen, 1, self.PRIOR_N)
        hx = torch.randn(1, self.window)
        cx = torch.randn(1, self.window)
        return self.__call__(batchlen, input, hx, cx)


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
        'frame': 50,
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
        'generator_args': {
            'PRIOR_N': 10,
        },
        # Depth and Withdraw of Hidden Layers
        'discriminator_args': {
        'WDTH': 100,
        'DPTH': 3},
        # Adam Optimizer parameters for G/D
        'lr_G': 1e-4,
        'betas_G': (0.5, 0.9),
        'lr_D': 1e-4,
        'betas_D': (0.5, 0.9)
    }

    param.update(training_param)
    GAN(**param)


    rnn = nn.LSTMCell(10, 20)
    input = torch.randn(6, 3, 10) #Batch size= 6, input size = 3 features, prior_N=10
    hx = torch.randn(3, 20)  #3 features, T =20
    cx = torch.randn(3, 20)
    output = []
    for i in range(6):
        hx, cx = rnn(input[i], (hx, cx))
        output.append(hx)
