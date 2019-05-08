import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from scipy.stats import gaussian_kde, entropy
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_data(name, array=True):
    """
    Load a dataset and returns its log-returns time series.
    """
    df = pd.read_csv(name, index_col=0, parse_dates=True, dayfirst=True)
    df.iloc[:, 0] = df.iloc[:, 0].astype('float')
    r = np.log(df).diff()[1:]
    dt = [(df.index[t+1]-df.index[t]).days for t in range(r.shape[0])]
    r = 1/np.array(dt) * r.iloc[:, 0]
    if array:
        return np.array(r)
    return r


def generate_batch(serie, length, BATCHLEN, add_noise=True, noise_lvl=0.2, proba_noise=0.5):
    """
    Returns random sequences of the given length from a numpy array.
    """
    if add_noise:
        sd = np.std(serie)
        sigma = noise_lvl * sd
    results = np.zeros((BATCHLEN, length))
    for i in range(BATCHLEN):
        random_start = np.random.choice(serie.shape[0]-length)
        results[i] = serie[random_start: random_start+length]
        if add_noise:
            u = np.random.binomial(n=1, p=proba_noise)
            results[i] += u*sigma*np.random.normal(size=length)
    return torch.tensor(results).float()


def softplus_loss(h, sgn):
    return torch.mean(F.softplus(h*sgn))


def negative_cross_entropy(h, target):
    return torch.mean(F.cross_entropy(h, target))


def KL_div(Gen, serie, length, batchlen):
    real_batch = generate_batch(serie, length, batchlen).detach().numpy().flatten()
    real_batch = (real_batch-np.mean(serie))/np.std(serie)
    fake_batch = Gen.generate(batchlen).detach().numpy().flatten()
    r_kde = gaussian_kde(real_batch)
    f_kde = gaussian_kde(fake_batch)
    eval_points = np.linspace(-6, 6, 500)
    pdf_values_real = r_kde.pdf(eval_points)
    pdf_values_fake = f_kde.pdf(eval_points)
    kl_div = entropy(pdf_values_real, pdf_values_fake)
    return kl_div


def get_KL_div_list(data, label, batchlen, name_wdw, gen, disc):
    corresp = {'3M': 60, '6M': 125, 'Y': 250}
    name_order = []
    KL_div_list = []
    for name in tqdm(name_wdw[label]):
        G, D, param_name = load_models(name, gen, disc)
        name_order.append(name)
        KL_div_list.append(KL_div(G, data, corresp[label], batchlen))
    return name_order, KL_div_list


def compare_plots(name, serie, batchlen, window, Gen, Disc):
    G, D, param_name = load_models(name, Gen, Disc)
    print(param_name)
    real_batch = (generate_batch(serie, window, batchlen)-np.mean(serie))/np.std(serie)
    fake_batch = G.generate(batchlen).detach()
    random_batch = np.random.normal(size=(window, batchlen))

    fig, ax = plt.subplots(3, figsize=(20, 20), sharey=True)
    fig.suptitle('Real Batch vs Generated Batch and Random Batch')
    ax[0].plot(real_batch.detach().numpy().T)
    ax[0].set_title('Batch from Real Serie')
    ax[1].plot(fake_batch.detach().numpy().T)
    ax[1].set_title('Generated Batch')
    ax[2].plot(random_batch)
    ax[2].set_title('Random Batch with the Standard deviation of the original serie')
    plt.show()


def load_models(name, Generator, Discriminator):
    print(name)
    param = pickle.load(open('Parameters/'+name+'.pk', 'rb'))
    if 'window' not in param.keys():
        param['window'] = param['param_gen_batch']['T']
    G = Generator(param['window'], **param['generator_args'])
    G.load_state_dict(torch.load('Generator/'+name+'.pth'))
    D = Discriminator(param['window'], **param['discriminator_args'])
    D.load_state_dict(torch.load('Discriminator/'+name+'.pth'))
    return G, D, param
