import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


def get_data(name, array=True):
    df = pd.read_csv(name, index_col=0, parse_dates=True, dayfirst=True)
    df.iloc[:, 0] = df.iloc[:, 0].astype('float')
    r = np.log(df).diff()[1:]
    dt = [(df.index[t+1]-df.index[t]).days for t in range(r.shape[0])]
    r = 365./np.array(dt) * r.iloc[:, 0]
    if array:
        return np.array(r)
    return r


def generate_batch(serie, length, BATCHLEN):
    """
    Returns random sequences of the given length from a numpy array
    """
    results = np.zeros((BATCHLEN, length))
    for i in range(BATCHLEN):
        random_start = np.random.choice(serie.shape[0]-length)
        results[i] = serie[random_start: random_start+length]
    return torch.tensor(results).float()


def softplus_loss(h, sgn):
    return torch.mean(torch.sum(F.softplus(h*sgn)))


def negative_cross_entropy(h, target):
    return torch.mean(torch.sum(-F.cross_entropy(h, target)))
