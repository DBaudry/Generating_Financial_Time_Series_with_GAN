from GAN import *

    
class Generator(nn.Module):
    def __init__(self, window=60, PRIOR_N=10, PRIOR_STD=1., DPTH = 0,WDTH =100):
        super().__init__()
        self.PRIOR_N = PRIOR_N
        self.PRIOR_STD = PRIOR_STD
        self.window = window
        self.fc1 = nn.Linear(PRIOR_N , WDTH)
        
        self.hidden_layers = []
        for _ in range(DPTH):
            self.hidden_layers.append(nn.Linear(WDTH, WDTH))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        
        self.fc2 = nn.Linear(WDTH, window)
        
      
        #In_channels de conv6 doit être le nombre sources de random
        self.conv1 =  nn.Conv1d(in_channels = 1 , out_channels = 3,
                                kernel_size = 3, padding=1)
        self.conv2 =  nn.Conv1d(in_channels = 3 , out_channels = 6,
                                kernel_size = 3, padding=1)
        self.conv3 =  nn.Conv1d(in_channels = 6 , out_channels = 3,
                                kernel_size = 3, padding=1)
        self.conv4 =  nn.Conv1d(in_channels = 3 , out_channels = 1,
                                kernel_size = 3, padding=1)

    def __call__(self, z):
        
      
        h = F.relu(self.fc1(z))
        for hidden_layer in self.hidden_layers:
            h = F.relu(hidden_layer(h))
        h = self.fc2(z)
        
        h = h.view(h.size()[0],1,h.size()[-1])
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        
        h = h.view(h.size()[0], self.window)
        
    
        return h

    def generate(self, batchlen):
        z = torch.normal(torch.zeros(batchlen,1 , self.PRIOR_N), self.PRIOR_STD)
        return self.__call__(z)




class Discriminator(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 3,
                               kernel_size = 3, padding=1,)
        self.conv2 = nn.Conv1d(in_channels = 3, out_channels = 6,
                               kernel_size = 3, padding=1,)
        self.conv3 = nn.Conv1d(in_channels = 6, out_channels = 3,
                               kernel_size = 3, padding=1,)
        self.conv4 = nn.Conv1d(in_channels = 3, out_channels = 1,
                               kernel_size = 3, padding=1,)
        self.fc1 = nn.Linear(self.window, 30)
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, 1)

    def __call__(self, z):
        
        h = z.view(z.size()[0] ,1 , z.size()[-1])
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        
        h = z.view(-1, self.window)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h


if __name__ == '__main__':
    param = {
        'serie': get_data('VIX.csv'),
        'window': 60,
        'frame': 200,
        'is_notebook': False,
        'batchlen_plot': 10,
        'Generator': Generator,
        'Discriminator': Discriminator
    }
    training_param = {
        'N_ITER': 2001,
        'TRAIN_RATIO': 10,
        'BATCHLEN': 30,
        # Depth and Withdraw of Hidden Layers
        'generator_args': {
        # Random Noise used by the Generator
        'PRIOR_N': 20,
        'PRIOR_STD': 500.,
        'WDTH': 100,
        'DPTH': 1},
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