import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# global variable for setting the torch.load    map_location
if torch.cuda.is_available():
    loadmap = {'cuda:0': 'gpu'}
else:
    loadmap = {'cuda:0': 'cpu'}


import utils
from GAN import GAN as gan

import matplotlib.pyplot as plt
import pandas as pd
import datetime
from tqdm import tqdm

from CvGan import *


# VIX Index from 2014
VIX = utils.get_data('VIX.csv')

parameter = [(2,10), (2,50), (5,10), (5,50), (10,10), (10,50)]
for couple in parameter:
	print( 'train' + str(couple[0]) +'_' + str(couple[1]))
	gan(serie = VIX, window = 60, Generator = Generator, Discriminator =Discriminator , generator_args = {'PRIOR_N':couple[0], 'PRIOR_STD':couple[1]}, discriminator_args={} ,
        TRAIN_RATIO=10, N_ITER=200, BATCHLEN=128,
        frame=1, frame_plot=100, is_notebook=False, batchlen_plot=5,
        lr_G=1e-3, betas_G=(0.5, 0.9), lr_D=1e-3, betas_D=(0.5, 0.9),
        loss=utils.softplus_loss, argloss_real=-1, argloss_fake=1, argloss_gen=1,
        save_model=True, save_name='CvGan' + '_' + str(couple[0]) + '_' + str(couple[1]) )





