from GAN import *


# Define the generator
class Generator(nn.Module):
    def __init__(self, window=60, PRIOR_N=10, PRIOR_STD=1., nlayers=2, hidden_size=20):
        super().__init__()
        self.window = window
        self.PRIOR_N = PRIOR_N
        self.PRIOR_STD = PRIOR_STD
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.fc_in = nn.Linear(PRIOR_N, window)
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=nlayers,
                           batch_first=True)
        self.fc_out = nn.Linear(hidden_size, 1)

    def __call__(self, batchlen, input):
        output, (hx, cx) = self.rnn(input)
        return self.fc_out(output)[:, :, 0]

    def generate(self, batchlen):
        input = torch.normal(torch.zeros(batchlen, self.PRIOR_N), self.PRIOR_STD)
        input = self.fc_in(input).unsqueeze(2)
        return self.__call__(batchlen, input)


# Define the discriminator.
class Discriminator(nn.Module):
    def __init__(self, window=0, nlayers=1, hidden_size=1):
        super().__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=nlayers,
                            batch_first=True)
        self.fc_hidden = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(window, 1)

    def __call__(self, x):
        input = x.unsqueeze(2)
        output, (hx, cx) = self.rnn(input)
        output = self.fc_hidden(output)[:, :, 0]
        return self.fc_out(output)
        # return torch.sigmoid(output)


def random_xp(n_xp):
    window = [60, 125, 250]
    PRIOR_N = [60]
    Train_ratio = [1, 3, 5, 10, 20, 50]
    PRIOR_STD = [1., 10, 20, 100]
    N_LAYERS = [1, 2, 3, 5, 10]
    for i in range(n_xp):
        param['window'] = np.random.choice(window)
        param['generator_args']['PRIOR_N'] = np.random.choice(PRIOR_N)
        param['generator_args']['PRIOR_STD'] = np.random.choice(PRIOR_STD)
        param['TRAIN_RATIO'] = np.random.choice(Train_ratio)
        param['generator_args']['nlayers'] = np.random.choice(N_LAYERS)
        param['discriminator_args']['nlayers'] = np.random.choice(N_LAYERS)
        param['save_name'] = 'RGAN_'+str(int(np.random.uniform()*1e9))
        print('Iteration %f' % i)
        print((param['window'], param['TRAIN_RATIO']))
        print(param['generator_args'])
        print(param['discriminator_args'])
        if param['save_model']:
            pickle.dump(param, open('Parameters/'+param['save_name']+'.pk', 'wb'))
        GAN(**param)


if __name__ == '__main__':
    param = {
        'serie': get_data('VIX.csv'),
        'window': 60,
        'frame': 10,
        'frame_plot': 100,
        'is_notebook': False,
        'batchlen_plot': 5,
        'Generator': Generator,
        'Discriminator': Discriminator,
        'BATCHLEN': 50
    }
    training_param = {
        'N_ITER': 1001,
        'TRAIN_RATIO': 5,
        # Random Noise used by the Generator
        'generator_args': {
            'PRIOR_N': 300,
            'PRIOR_STD': 10,
            'nlayers': 1,
            'hidden_size': 50
        },
        # Depth and Withdraw of Hidden Layers
        'discriminator_args': {
        'nlayers': 1,
        'hidden_size': 10},
        # Adam Optimizer parameters for G/D
        'lr_G': 1e-4,
        'betas_G': (0.5, 0.9),
        'lr_D': 1e-4,
        'betas_D': (0.5, 0.9),
        # 'loss': utils.negative_cross_entropy,
        # 'argloss_real': torch.ones(param['BATCHLEN'], dtype=torch.int64),
        # 'argloss_fake': torch.zeros(param['BATCHLEN'], dtype=torch.int64),
        # 'argloss_gen': torch.ones(param['BATCHLEN'], dtype=torch.int64),
        'save_model': True,
        'save_name': 'RGAN_'+str(int(np.random.uniform()*1e9)),
        'plot': False,
        'time_max': 7200
    }

    param.update(training_param)

    # random_xp(100)
    if param['save_model']:
        pickle.dump(param, open('Parameters/'+param['save_name']+'.pk', 'wb'))
    GAN(**param)

