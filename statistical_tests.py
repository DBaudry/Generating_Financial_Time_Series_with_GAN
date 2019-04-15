from GAN import *
from sklearn import svm
"""
Implement some unit test to check if a time serie verifies some facts that are generally true 
for financial time series
"""

from statsmodels.tsa.stattools import adfuller, acf
from scipy.stats import skew, kurtosis


def is_stationnary(serie, threshold, display_stats=True):
    result = adfuller(serie)
    if display_stats:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
    if result[0] < result[4][threshold]:
        return True


def check_autocorel(serie, nlags, alpha, qstat=True, score_min=0.8, display_stats=True):
    result = acf(serie, nlags=nlags, alpha=alpha, qstat=qstat)
    if display_stats:
        print('Autocorrelations: {}'.format(result[0]))
        print('Confidence intervals: {}'.format(result[1]))
        print('Q stats of Ljung Box test: {}'.format(result[2]))
        print('p-values: {}'.format(result[3]))
    score = sum(result[3] < alpha)/(len(result[3])-1)
    if score > score_min:
        return True, score
    return False, score, np.where(result[3] > alpha)


def get_moments(serie, display=True):
    mean = serie.mean()
    vol = serie.std()
    skw = skew(serie)
    kurt = kurtosis(serie)
    if display:
        print('Empirical Moments Statistics')
        print('Mean: %f' % mean)
        print('Volatility: %f' % vol)
        print('Skewness: %f' % skw)
        print('Excess kurtosis (compared with normal): %f' % kurt)
    return mean, vol, skw, kurt


def compare_moments(base_stats, generated_serie, display=False):
    M1 = get_moments(generated_serie, display=display)
    var = [M1[i]-base_stats[i] for i in range(len(base_stats))]
    if M1[2] > 0:
        pos_skew = True
    else:
        pos_skew = False
    if M1[3] > 0:
        is_leptokurtic = True
    else:
        is_leptokurtic = False
    if display:
        print('Variations between Moments Statistics of the two series: {}'.format(var))
        print('Positiveness of the skewness: %i' % pos_skew)
        print('Leptokurticity: %i' % is_leptokurtic)
    return (pos_skew, is_leptokurtic), M1, var


def compute_all_stats(base_stats, serie, nlags=10, alpha=0.10, score_min=0.8, display=True):
    """
    :param base_stats: Statistics of the original serie (get_moments style)
    :param serie: New serie to analyze
    :param nlags: Number of lags to consider in autocorrelation
    :param alpha: Percentage for confidence intervals in all stats
    :param score_min: Minimum number of non significant autocorrelations (under the threshold alpha) to
    set the non autocorrelation property to True
    :param display: Display all statistics or not
    :return: Results for Stationnarity test, autocorrelation test (for the serie and its square),
    """
    str_alpha = str(int(100 * alpha))+'%'
    statn = is_stationnary(serie, str_alpha, display_stats=display)
    autocor = check_autocorel(serie, nlags, alpha,
                              qstat=True, score_min=score_min, display_stats=display)
    square_autocor = check_autocorel(serie**2, nlags, alpha,
                              qstat=True, score_min=score_min, display_stats=display)
    res_moments = compare_moments(base_stats, serie, display=display)
    return statn, autocor, square_autocor, res_moments


def check_SVM(serie, generator, batchlen_train, batchlen_test):
    real_batch = utils.generate_batch(serie, generator.window, batchlen_train).detach().numpy()
    fake_batch = generator.generate(batchlen_train).detach().numpy()
    label_real = np.ones(batchlen_train)
    label_fake = np.zeros(batchlen_train)
    X = np.concatenate([real_batch, fake_batch], axis=0)
    y = np.concatenate([label_real, label_fake], axis=0).astype('int')
    clf = svm.SVC()
    clf.fit(X, y)
    test_real = utils.generate_batch(serie, generator.window, batchlen_test).detach().numpy()
    test_fake = generator.generate(batchlen_test).detach().numpy()
    test_r_label = np.ones(batchlen_test)
    test_f_label = np.zeros(batchlen_test)
    test_X = np.concatenate([test_real, test_fake], axis=0).astype('int')
    test_y = np.concatenate([test_r_label, test_f_label], axis=0)
    return clf.score(test_X, test_y)


if __name__ == '__main__':
    serie = get_data('VIX.csv')
    # batch = generate_batch(serie, 250, 2).numpy()
    # M0 = get_moments(batch[0])
    # compare_moments(M0, batch[1], display=True)
    # compute_all_stats(M0, serie,
    #                   nlags=10, alpha=0.10, score_min=0.8, display=True)
    from Lin_GAN import *
    from Lin_GAN import Generator as LG
    from Lin_GAN import Discriminator as LD
    name = 'Lin_G_268169440'
    G, D, param_name = utils.load_models(name, LG, LD)
    print(check_SVM(serie, G, 500, 500))
    plt.plot(G.generate(100).detach().numpy().T)
    plt.show()