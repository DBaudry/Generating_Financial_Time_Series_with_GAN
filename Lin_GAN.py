from GAN import *
from tqdm import tqdm

# Define the generator
class Generator(nn.Module):
    def __init__(self, window, WDTH=100, PRIOR_N=10, DPTH=1, PRIOR_STD=100.):
        super().__init__()
        self.PRIOR_N = PRIOR_N
        self.PRIOR_STD = PRIOR_STD
        self.fc1 = nn.Linear(PRIOR_N, WDTH)
        self.hidden_layers = []
        for _ in range(DPTH):
            self.hidden_layers.append(nn.Linear(WDTH, WDTH))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.fc2 = nn.Linear(WDTH, window)
        self.bn = nn.BatchNorm1d(WDTH)
        self.bn_out = nn.BatchNorm1d(window)

    def __call__(self, z):
        h = self.bn(F.relu(self.fc1(z)))
        for hidden_layer in self.hidden_layers:
            h = self.bn(F.relu(hidden_layer(h)))
        return self.fc2(h)
        # return self.bn_out(self.fc2(h))

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


def random_xp(n_xp):
    window = [60, 125, 250]
    PRIOR_N = [1, 10, 20, 50, 100, 200]
    BATCHLEN = [10, 30, 50, 100]
    Train_ratio = [1, 3, 5, 10]
    PRIOR_STD = [1., 10, 20, 100]
    WDTH_G = [10, 50, 100, 200, 500, 1000]
    DPTH_G = [3, 5, 10, 15, 20]
    WDTH_D = [10, 50, 100, 200, 1000]
    DPTH_D = [1, 2, 3, 5]
    for i in range(n_xp):
        param['window'] = np.random.choice(window)
        param['BATCHLEN'] = np.random.choice(BATCHLEN)
        param['generator_args']['PRIOR_N'] = np.random.choice(PRIOR_N)
        param['generator_args']['PRIOR_STD'] = np.random.choice(PRIOR_STD)
        param['TRAIN_RATIO'] = np.random.choice(Train_ratio)
        param['generator_args']['WDTH'] = np.random.choice(WDTH_G)
        param['generator_args']['DPTH'] = np.random.choice(DPTH_G)
        param['discriminator_args']['WDTH'] = np.random.choice(WDTH_D)
        param['discriminator_args']['DPTH'] = np.random.choice(DPTH_D)
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
        'window': 60,
        'frame': 20,
        'frame_plot': 50,
        'is_notebook': False,
        'batchlen_plot': 100,
        'Generator': Generator,
        'Discriminator': Discriminator
    }
    training_param = {
        'N_ITER': 500,
        'TRAIN_RATIO': 5,
        'BATCHLEN': 100,
        # Depth and Withdraw of Hidden Layers
        'generator_args': {
        # Random Noise used by the Generator
        'PRIOR_N': 10,
        'PRIOR_STD': 10.,
        'WDTH': 500,
        'DPTH': 10},
        'discriminator_args': {
        'WDTH': 100,
        'DPTH': 3},
        # Adam Optimizer parameters for G/D
        'lr_G': 1e-4,
        'betas_G': (0.5, 0.9),
        'lr_D': 1e-4,
        'betas_D': (0.5, 0.9),
        'time_max': 600,
        'save_model': True,
        'save_name': 'Lin_G_'+str(int(np.random.uniform()*1e9)),
        'plot': False
    }
    param.update(training_param)
    if param['save_model']:
        pickle.dump(param, open('Parameters/'+param['save_name']+'.pk', 'wb'))
    # GAN(**param)

    random_xp(2000)
    # name = 'Lin_G_735900290'
    # G, D, param_name = utils.load_models(name, Generator, Discriminator)
    # print(param)
    # plt.plot(G.generate(30).detach().numpy().T)
    # plt.show()

