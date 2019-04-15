from GAN import *
from torch.nn.functional import interpolate


class Interpolate(nn.Module):
    def __init__(self, size):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        #self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size)
        return x

Upsample = Interpolate

class Generator(nn.Module):
    #Deconv block : conv1d, reakyrelu upslamping
    def __init__(self, kernel_size = 5, PRIOR_N = 2, PRIOR_STD=10, window = 60 ):
        super().__init__()
        self.window = window
        self.PRIOR_N = PRIOR_N
        self.PRIOR_STD = PRIOR_STD 
        self.kernel_size = kernel_size
        self.padding = int((self.kernel_size-1)/2)
        self.lin1 = nn.Linear(self.PRIOR_N, 15)
        self.bn1 = nn.BatchNorm1d(15)
        self.conv1 = nn.Conv1d(1, 32, kernel_size = self.kernel_size, padding = self.padding)
        
        self.bn2 = nn.BatchNorm1d(32)
        self.up1 = Upsample(size=30)
        self.conv2 = nn.Conv1d(32, 32, kernel_size = self.kernel_size, padding = self.padding)
        self.bn3 = nn.BatchNorm1d(32)
        self.up2 = Upsample(size = 60)
        self.conv3 = nn.Conv1d(32, 32, kernel_size = self.kernel_size, padding = self.padding)
        self.bn4 = nn.BatchNorm1d(32)
        self.up3 = Upsample(size = 120)
        self.bn5 = nn.BatchNorm1d(120)
        
        self.conv4 = nn.Conv1d(32, 1, kernel_size = 1)
        self.lin2 = nn.Linear(120, self.window)
        
    def __call__(self, x):
        x = self.lin1(x)
        x = x.view(x.size()[0], 15, 1)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = x.view(x.size()[0], 1, 15)
        x = self.up1(x)
        x = F.leaky_relu(self.bn2(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv2(x)), negative_slope=0.2)
        x = self.up2(x)
        x = F.leaky_relu(self.bn4(self.conv3(x)), negative_slope=0.2)
        x = self.up3(x)
        x = self.conv4(x)
        x = x.view(x.size()[0], x.size()[2])
        x = F.leaky_relu(self.bn5(x), negative_slope=0.2)
        x = x.view(x.size()[0], x.size()[-1])
        x = F.leaky_relu(self.lin2(x), negative_slope=0.5)
                
        return x

    def generate(self, batchlen):
        z = torch.normal(torch.zeros(batchlen,1 , self.PRIOR_N), self.PRIOR_STD)
        return self.__call__(z)



class Discriminator(nn.Module):
    def __init__(self, window = 60):
        super().__init__()
        self.window = window
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size = 3, padding=1)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3, padding=1)
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3, padding=1)
        self.conv4 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3, padding=1)
        self.fc1 = nn.Linear(32*int(self.window/8), 50)
        self.fc2 = nn.Linear(50, 15)
        self.fc3 = nn.Linear(15, 1)

    def __call__(self, x):
        x = x.view(x.size()[0], 1, x.size()[1])
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = self.maxpool(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = self.maxpool(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = self.maxpool(x)
        
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = x.view(x.size()[0], 32*int(self.window/8))
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        return x


if __name__=='__main__':
    param = {
        'serie': get_data('VIX.csv'),
        'window': 60,
        'frame': 1000,
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

