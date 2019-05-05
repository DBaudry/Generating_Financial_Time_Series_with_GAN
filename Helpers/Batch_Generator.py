import numpy as np
import matplotlib.pyplot as plt
import torch


def generate_BS(r, sigma, T, batchsize):
    """
    :param S0: price in t0
    :param r: drift
    :param sigma: volatility
    :param T: Time horizon
    :param batchsize: batch size
    :return: Black Scholes sample, computed by annualizing the provided drift and vol
    """
    mu = (r - sigma**2/2)/250
    dl = sigma/np.sqrt(250)*np.random.normal(size=(batchsize, T)) + mu
    # batch = S0 * np.cumprod(np.exp(dl), axis=1)
    return torch.tensor(dl).float()


if __name__ == '__main__':
    s = generate_BS(100, 0.02, 0.15, 1000, 200)
    plt.plot(s.T)
    plt.show()
