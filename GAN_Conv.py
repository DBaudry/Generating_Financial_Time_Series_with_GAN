from GAN import *

    
class Generator(nn.Module):
    def __init__(self, window, PRIOR_N=10, PRIOR_STD=1.):
        super().__init__()
        self.PRIOR_N = PRIOR_N
        self.PRIOR_STD = PRIOR_STD
        self.window = window
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16,
                                kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=30,
                                kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=30, out_channels=self.window,
                                kernel_size=3, padding=1)
        #In_channels de conv6 doit être le nombre sources de random
        self.conv5 = nn.Conv1d(in_channels=self.PRIOR_N, out_channels=5,
                                kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(in_channels=5, out_channels=1,
                                kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(window)

    def __call__(self, z):
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))
        z = z.view(z.size()[0], self.PRIOR_N, self.window) #La dim devient window
        
        z = F.relu(self.conv5(z))
        z = self.conv6(z)
        
        return self.bn(z.view(-1, self.window))

    def generate(self, batchlen):
        z = torch.normal(torch.zeros(batchlen, 1 , self.PRIOR_N), self.PRIOR_STD)
        return self.__call__(z)


class Discriminator(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3,
                               kernel_size=3, padding=1,)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=6,
                               kernel_size=3, padding=1,)
        self.conv3 = nn.Conv1d(in_channels=6, out_channels=3,
                               kernel_size=3, padding=1,)
        self.conv4 = nn.Conv1d(in_channels=3, out_channels=1,
                               kernel_size=3, padding=1,)
        self.fc1 = nn.Linear(self.window, 30)
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, 1)
        self.bn = nn.BatchNorm1d(window)
        self.fc_single = nn.Linear(self.window, 1)

    def __call__(self, z):
        

        z = z.view(z.size()[0], 1, z.size()[-1])
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))
        
        z = self.bn(z.view(-1, self.window))
        z = F.relu(self.fc_single(z))
        # z = F.relu(self.fc1(z))
        # z = F.relu(self.fc2(z))
        # z = self.fc3(z)
        return z


if __name__ == '__main__':
    param = {
        'serie': get_data('VIX.csv'),
        'window': 60,
        'frame': 10,
        'frame_plot': 20,
        'is_notebook': False,
        'batchlen_plot': 10,
        'Generator': Generator,
        'Discriminator': Discriminator
    }
    training_param = {
        'N_ITER': 1001,
        'TRAIN_RATIO': 10,
        'BATCHLEN': 50,
        # Depth and Withdraw of Hidden Layers
        'generator_args': {
        # Random Noise used by the Generator
        'PRIOR_N': 200,
        'PRIOR_STD': 10.},
        'discriminator_args': {},
        # Adam Optimizer parameters for G/D
        'lr_G': 1e-4,
        'betas_G': (0.5, 0.9),
        'lr_D': 1e-4,
        'betas_D': (0.5, 0.9)
    }

    param.update(training_param)
    GAN(**param)