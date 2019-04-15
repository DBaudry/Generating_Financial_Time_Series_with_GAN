from GAN import *


# Define the generator
class Generator(nn.Module):
    def __init__(self, window, WDTH=0, PRIOR_N=10, PRIOR_CHANNEL=20,
                  PRIOR_STD=1.):
        super().__init__()
        self.PRIOR_N = PRIOR_N
        self.PRIOR_STD = PRIOR_STD
        self.PRIOR_CHANNEL = PRIOR_CHANNEL
        self.WDTH = WDTH
        self.window = window
        self.main = nn.Sequential(
        nn.ConvTranspose1d(PRIOR_CHANNEL, 8*WDTH, 4, 1, 0, bias=False),  #Kernel size/stride/padding
        nn.BatchNorm1d(8*WDTH),
        nn.ReLU(True),
        nn.ConvTranspose1d(8*WDTH, 4*WDTH, 4, 2, 1, bias=False),
        nn.BatchNorm1d(4*WDTH),
        nn.ReLU(True),
        nn.ConvTranspose1d(4*WDTH, 2*WDTH, 4, 2, 1, bias=False),
        nn.BatchNorm1d(2*WDTH),
        nn.ReLU(True),
        nn.ConvTranspose1d(2*WDTH, WDTH, 4, 2, 1, bias=False),
        nn.BatchNorm1d(WDTH),
        nn.ReLU(True),
        nn.ConvTranspose1d(WDTH, 1, 4, 2, 1, bias=False)
        #,nn.Tanh()
        )

    def generate(self, batchlen):
        z = torch.normal(torch.zeros(batchlen, self.PRIOR_CHANNEL, self.PRIOR_N), self.PRIOR_STD)
        z = self.main(z)
        return z[:, 0, :self.window]


# Define the discriminator.
class Discriminator(nn.Module):
    def __init__(self, window, WDTH=20):
        super().__init__()
        self.WDTH = WDTH
        self.main = nn.Sequential(
        nn.Conv1d(1, WDTH, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(WDTH, 2*WDTH, 4, 2, 1, bias=False),
        nn.BatchNorm1d(2*WDTH),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(2*WDTH, 4*WDTH, 4, 2, 1, bias=False),
        nn.BatchNorm1d(4*WDTH),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(4*WDTH, 8*WDTH, 4, 2, 1, bias=False),
        nn.BatchNorm1d(8*WDTH),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(8*WDTH, 1, 4, 1, 0, bias=False),
        nn.Linear(4, 1)
        #    , nn.Sigmoid()
        )

    def __call__(self, x):
        x = x.unsqueeze(1)
        return self.main(x)[:, 0, :]


def random_xp(n_xp):
    window = [60, 125, 250]
    PRIOR_N = [5]
    BATCHLEN = [30, 50, 100]
    Prior_channel = [1, 3, 5, 10, 20]
    Train_ratio = [1, 3, 5, 10]
    PRIOR_STD = [1., 10., 20., 100.]
    WDTH_G = [10, 50, 100, 200, 500, 1000]
    for i in range(n_xp):
        param['window'] = np.random.choice(window)
        param['BATCHLEN'] = np.random.choice(BATCHLEN)
        param['generator_args']['PRIOR_N'] = np.random.choice(PRIOR_N)
        param['generator_args']['PRIOR_STD'] = np.random.choice(PRIOR_STD)
        param['TRAIN_RATIO'] = np.random.choice(Train_ratio)
        param['generator_args']['WDTH'] = np.random.choice(WDTH_G)
        param['generator_args']['PRIOR_CHANNEL'] = np.random.choice(Prior_channel)
        WDTH_D = [x for x in WDTH_G if x <= param['generator_args']['WDTH']]
        param['discriminator_args']['WDTH'] = np.random.choice(WDTH_D)
        param['save_name'] = 'Lin_G_'+str(int(np.random.uniform()*1e9))
        print('Iteration %f' % i)
        print((param['window'], param['BATCHLEN'], param['TRAIN_RATIO']))
        print(param['generator_args'])
        print(param['discriminator_args'])
        if param['save_model']:
            pickle.dump(param, open('Parameters/'+param['save_name']+'.pk', 'wb'))
        GAN(**param)


if __name__ == '__main__':
    param = {
        'serie': get_data('VIX.csv'),
        'window': 125,
        'frame': 20,
        'is_notebook': False,
        'batchlen_plot': 20,
        'Generator': Generator,
        'Discriminator': Discriminator
    }
    training_param = {
        'N_ITER': 2001,
        'TRAIN_RATIO': 5,
        'BATCHLEN': 50,
        # Random Noise used by the Generator
        'generator_args': {
        'PRIOR_N': 5,
        'PRIOR_STD': 100.,
        'WDTH': 100,
        'PRIOR_CHANNEL': 3,
        },
        'discriminator_args': {'WDTH': 40},
        # Adam Optimizer parameters for G/D
        'lr_G': 1e-4,
        'betas_G': (0.5, 0.9),
        'lr_D': 1e-4,
        'betas_D': (0.5, 0.9),
        'plot': False,
        'frame_plot': 100,
        'time_max': 1800,
        'save_model': True,
        'save_name': 'CG_'+str(int(np.random.uniform()*1e9))
    }
    param.update(training_param)

    random_xp(2000)
    # if param['save_model']:
    #     pickle.dump(param, open('Parameters/'+param['save_name']+'.pk', 'wb'))
    # GAN(**param)