"""This module contains a set of functions that are mostly 
related to the fitting of the drift-diffusion model and the
running time model. """

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from utils import *
from plotting import barplot_annotate_brackets

#######################################################################


# separate the data into time and reward bins
def prepare_data_idle_times(sequence, animalList, sessionList, memsize=3, time_bins=6):
    """prepare data for fitting
    cut the data into time bins and reward bins"""
    bin_size = 3600/time_bins
    targetlist = generate_targetList(memsize)[::-1]
    temp_data = {}
    for time_bin in range(time_bins):
        temp_data[time_bin] = {}
        for animal in animalList:
            temp_data[time_bin][animal] = {k: [] for k in meankeys(targetlist)}
            for session in matchsession(animal, sessionList):
                temp_data[time_bin][animal] = combine_dict(temp_data[time_bin][animal], get_waiting_times(sequence[animal, session], memsize=memsize, filter=[time_bin*bin_size, (time_bin+1)*bin_size]))

    data = {}
    for animal in animalList:
        data[animal] = np.zeros((time_bins, len(meankeys(targetlist)))).tolist()
        for avg_bin, avg in enumerate(meankeys(targetlist)):  # 1 -> 0
            for time_bin in range(time_bins):
                data[animal][time_bin][avg_bin] = np.asarray(temp_data[time_bin][animal][avg])
    return data

# generate target list and meankeys: more complex than what it should be, because in earlier versions we
# were interested in all possible combinations of rewards, not just the average reward
def generate_targetList(seq_len=1):
    """generate list of all reward combinations for specified memory length seq_len"""
    get_binary = lambda x: format(x, 'b')
    output = []
    for i in range(2**seq_len):
        # list binary number from 0 to 2**n, add leading zeroes when resulting seq is too short 
        binstr = "0" * abs(len(get_binary(i)) - seq_len) + str(get_binary(i))
        output.append(binstr)
    return output


def meankeys(targetlist):
    """get all possible average reward for a list of possible reward combinations"""
    result = []
    for target in targetlist:
        res = round(np.mean([int(elem) for elem in target]), 2)
        if res not in result:
            result.append(res)
    return result


def get_waiting_times(data, memsize=3, filter=[0, 3600], toolong=3600):
    """get waiting times from sequence of actions data and separate them
    according to the average reward of the sequence"""
    waiting_times = {k: [] for k in meankeys(generate_targetList(seq_len=memsize)[::-1])}
    for i in range(len(data)):
        if data[i][1] == 'stay':
            if filter[0] <= data[i][0] <= filter[1] and data[i][3] != 0:
                if data[i][3] < toolong:  # filter out
                    try:
                        avg_rwd = round(np.mean([data[i-n][2] for n in range(1, (memsize*2)+1, 2)]), 2)
                        waiting_times[avg_rwd].append(data[i][3])
                    except:  # put the first n waits in rwd=1 (because we don't have the previous n runs to compute the average reward)
                        waiting_times[1].append(data[i][3])
    return waiting_times


def combine_dict(d1, d2):
    """combine two dictionaries with the same keys"""
    keys = d1.keys()
    values = [np.concatenate([d1[k], d2[k]]) for k in keys]
    return dict(zip(keys, values))

######################################################


def generate_trials(mean, std, A, t0):
    """generate a single drift-diffusion trial"""
    # np.random.seed(0)
    dv = [0] * (t0 + 1)
    while dv[-1] < A:
        evidence = np.random.normal(mean, std)
        dv.append(dv[-1] + evidence)
    return dv


def round_nearest(x, a=0.04):
    return np.around(x / a) * a


def generate_running_time(loc, scale, N=1, seed=0):
    np.random.seed(seed=seed)
    res = stats.cauchy.rvs(loc, scale, size=int(N))
    res = res[res > 0.75]
    # res = round_nearest(res)
    return res


def generate_idle_time(alpha, gamma, N=1, maximum=500, seed=0):
    def p(x, a, g): 
        return a / np.sqrt(2 * np.pi * x ** 3) * np.exp(-((a-g * x) ** 2) / (2 * x))
    def normalization(x, a, g): 
        return simps(p(x, a, g), x)
    x = np.linspace(1e-8, maximum, maximum*100)

    pdf = p(x, alpha, gamma)/normalization(x, alpha, gamma)
    cdf = np.cumsum(pdf); cdf /= max(cdf)

    np.random.seed(seed=seed)
    u = np.random.uniform(0, 1, int(N))
    interp_function = interp1d(cdf, x)
    samples = interp_function(u)
    return samples
#############################################################


def Wald_pdf(x, alpha, theta, gamma):
    """Wald pdf"""
    x = np.asarray(x) - theta
    x[x < 0] = 1e-10
    arg = 2 * np.pi * x ** 3
    res = alpha / np.sqrt(arg) * np.exp(-((alpha-gamma * x) ** 2) / (2 * x))
    return np.array(res, dtype=np.float64)


def Wald_cdf(x, alpha, theta, gamma):
    """Wald cdf"""
    # from https://github.com/mark-hurlstone/RT-Distrib-Fit
    x = x - theta
    x[x < 0] = 1e-10
    return np.array(stats.norm.cdf((gamma*x-alpha)/np.sqrt(x)) + np.exp(2*alpha*gamma)*stats.norm.cdf(-(gamma*x+alpha)/np.sqrt(x)), dtype=np.float64)


# interactive plot
def plot_interactiveWald(alpha=1, gamma=2, t_0=0):
    """interactive plot of Wald pdf"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    x = np.linspace(0.01, 4, 400)
    axs[0].plot(x, Wald_pdf(x, 1, 0, 2), 'k-', label='Default')
    axs[0].plot(x, Wald_pdf(x, 2.5, 0, 2), 'c', label='increased alpha')
    axs[0].plot(x, Wald_pdf(x, 1, 0, 3.8), 'r-', label='increased gamma')
    axs[0].plot(np.linspace(0.81, 4, 1000), Wald_pdf(np.linspace(0.81, 4, 1000), 1, .8, 2), 'g-', label='increased theta')
    axs[0].set_ylabel('PDF')
    axs[0].set_xlabel('t')
    axs[0].set_xlim(0, 4)
    axs[0].set_ylim(0, 4)
    axs[0].legend()

    pdf = Wald_pdf(x, alpha, t_0, gamma)
    cdf = 1-Wald_cdf(x, alpha, t_0, gamma)
    axs[1].plot(x, pdf)
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('pdf')
    axs[1].set_title('pdf')
    axs[2].plot(x, cdf)
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('log t')
    axs[2].set_ylabel('log 1-cdf')
    axs[2].set_title('log 1-cdf')

    axs[1].set_xlim(0, 4)
    axs[1].set_ylim(0, 4)
    axs[2].set_xlim(0.01, 10)
    axs[2].set_ylim(0.01, 1.1)
    return


##########################################################
def log_lik_wald(x, params, robustness_param=1e-20):
    """log likelihood function for Wald distribution"""
    alpha, theta, gamma = params
    pdf_vals = Wald_pdf(x, alpha, theta, gamma) + robustness_param
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val


def crit(params, *args, robustness_param=1e-20):
    """negative log likelihood function for Wald distribution"""
    alpha, theta, gamma = params
    x = args
    pdf_vals = Wald_pdf(x, alpha, theta, gamma) + robustness_param
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val


def wald_fit(x, alpha_init=2, theta_init=0, gamma_init=.5):
    """fit Wald distribution"""
    params_init = np.array([alpha_init, theta_init, gamma_init])
    res = minimize(crit, params_init, args=x, bounds=((0, None), (0, 1e-8), (0, None)))
    return res.x, res.fun


def genWaldSamples(N, alpha, gamma, maximum=500):
    """generate Wald samples"""
    # 230x faster than drawfromDDM (pyDDM)
    # based on https://harry45.github.io/blog/2016/10/Sampling-From-Any-Distribution
    x = np.linspace(1e-8, maximum, maximum*100)

    def p(x, alpha, gamma):
        return alpha / np.sqrt(2 * np.pi * x ** 3) * np.exp(-((alpha-gamma * x) ** 2) / (2 * x))

    def normalization(x, alpha, gamma):
        return simps(p(x, alpha, gamma), x)

    pdf = p(x, alpha, gamma)/normalization(x, alpha, gamma)
    cdf = np.cumsum(pdf)
    cdf /= max(cdf)

    u = np.random.uniform(0, 1, int(N))
    interp_function = interp1d(cdf, x)
    samples = interp_function(u)
    return samples

######################################################################

def model_crit(params, *args, robustness_param=1e-20):
    """negative log likelihood function for full model"""
    alpha, theta, gamma, alpha_t, theta_prime, gamma_t, alpha_R, theta_second, gamma_R = params
    neg_log_lik_val = 0
    N_bins, N_avg = args[1]
    ALPHA = np.zeros((N_bins, N_avg))
    GAMMA = np.zeros((N_bins, N_avg))
    _theta = theta + theta_prime + theta_second

    for bin in range(N_bins):
        for avg in range(N_avg):
            ALPHA[bin, avg] = alpha + bin*alpha_t + avg*alpha_R
            GAMMA[bin, avg] = gamma + bin*gamma_t + avg*gamma_R

    for bin in range(N_bins):
        for avg in range(N_avg):
            _alpha = ALPHA[bin, avg] if ALPHA[bin, avg] > 0 else 1e-8
            _gamma = GAMMA[bin, avg]# if GAMMA[bin, avg] > 0 else 1e-8
            try:
                pdf_vals = Wald_pdf(args[0][bin][avg], _alpha, _theta, _gamma)
                ln_pdf_vals = np.log(pdf_vals + robustness_param)
                log_lik_val = ln_pdf_vals.sum()

                n = len(args[0][bin][avg]) if len(args[0][bin][avg]) > 0 else 1
                neg_log_lik_val += (-log_lik_val / n)
            except:
                neg_log_lik_val += 0  # add 0 instead of throwing an error when there is no data in a bin*avg
    return neg_log_lik_val


def model_compare(params, *args, robustness_param=1e-20):
    """BIC to compare models with different number of parameters and curves"""
    alpha, theta, gamma, alpha_t, theta_prime, gamma_t, alpha_R, theta_second, gamma_R = params
    BIC = 0
    N = 0
    sum_log_likelihood = 0

    N_bins, N_avg = args[1]
    N_params = args[2]
    ALPHA = np.zeros((N_bins, N_avg))
    GAMMA = np.zeros((N_bins, N_avg))
    _theta = theta + theta_prime + theta_second

    for bin in range(N_bins):
        for avg in range(N_avg):
            ALPHA[bin, avg] = alpha + bin*alpha_t + avg*alpha_R
            GAMMA[bin, avg] = gamma + bin*gamma_t + avg*gamma_R

    for bin in range(N_bins):
        for avg in range(N_avg):
            _alpha = ALPHA[bin, avg] if ALPHA[bin, avg] > 0 else 1e-8
            _gamma = GAMMA[bin, avg]  # if GAMMA[bin, avg] > 0 else 1e-8
            pdf_vals = Wald_pdf(args[0][bin][avg], _alpha, _theta, _gamma)
            ln_pdf_vals = np.log(pdf_vals + robustness_param)
            log_lik_val = ln_pdf_vals.sum()

            n = len(args[0][bin][avg]) if len(args[0][bin][avg]) > 0 else 1
            N += n
            sum_log_likelihood += log_lik_val

            # except:
            #     BIC += 0  # add 0 instead of throwing an error when there is no data in a bin*avg

    k = N_params
    BIC = k * np.log(N) - 2 * sum_log_likelihood
    return BIC


def modelwald_fit(data, init=[2, 0, .5, 0, 0, 0, 0, 0, 0],
                  f=model_crit, N_bins=6, N_avg=4, N_params=2,
                  alpha_t_fixed=False, gamma_t_fixed=False,
                  alpha_U_fixed=False, gamma_U_fixed=False,
                  ):
    """fit full model to data"""
    params_init = np.array(init)
    alpha_t_bounds = (None, None) if not alpha_t_fixed else (0, 1e-8)
    gamma_t_bounds = (None, None) if not gamma_t_fixed else (0, 1e-8)
    alpha_U_bounds = (None, None) if not alpha_U_fixed else (0, 1e-8)
    gamma_U_bounds = (None, None) if not gamma_U_fixed else (0, 1e-8)

    res = minimize(f, params_init, args=(data, [N_bins, N_avg], N_params),
                   bounds=((0, None), (0, 1e-8), (0, None),
                   alpha_t_bounds, (0, 1e-8), gamma_t_bounds,
                   alpha_U_bounds, (0, 1e-8), gamma_U_bounds))
    return res.x, res.fun


################################################


def dict_to_xticklabels(d, labels=['$\\alpha_{\mathrm{t}}$', '$\\gamma_{\mathrm{t}}$', '$\\alpha_R$', '$\\gamma_R$']):
    """convert dict keys to xticklabels for ablation plot"""
    allkeys = list(d.keys())
    conv = lambda x: "-" if x else "+"
    result = ["\n".join(labels)]
    for i in allkeys:
        result.append(f'{chr(10).join([conv(j) for j in i])}')
    return result


def exact_mc_perm_test(x, y, nmc=10000, return_shuffled=False):
    '''permutation test for two independent samples'''
    n = len(x)
    k = 0
    diff = np.abs(np.median(x) - np.median(y))
    z = np.concatenate([x, y])

    s = []
    for j in range(nmc):
        np.random.shuffle(z)
        k += diff <= np.abs(np.median(z[:n]) - np.median(z[n:]))
        s.append(np.abs(np.median(z[:n]) - np.median(z[n:])))
    p_value = k / nmc

    if return_shuffled:
        return p_value, s, diff
    else:
        return p_value


def exact_mc_perm_paired_test(x, y, nmc=10000):
    '''permutation test for two paired samples'''
    def effect(x, y): # paired median difference
        return np.abs(np.median(x) - np.median(y))
    
    k = 0
    obs = effect(x, y)

    def random_swap(x, y):
        n = len(x)
        k = x.shape[-1] if x.ndim > 1 else 1
        swaps = (np.random.random(n) < 0.5).repeat(k).reshape(n, k)
        x_ = np.select([swaps, ~swaps], [x.reshape(n, k), y.reshape(n, k)])
        y_ = np.select([~swaps, swaps], [x.reshape(n, k), y.reshape(n, k)])
        return x_, y_

    for _ in range(nmc):
        xs_, ys_ = random_swap(x, y)
        if effect(xs_, ys_) >= obs:
            k += 1
    
    p_value = k / nmc
    return p_value


def multipletests_bonferroni(pvals, alpha=0.05):
    pvals = np.asarray(pvals)
    sorted = np.argsort(pvals)
    pvals = np.take(pvals, sorted)

    ntests = len(pvals)
    alphacBonf = alpha / float(ntests)

    # bonferroni correction, return everything <= alphacBonf, clip pvals at 1
    reject = pvals <= alphacBonf
    pvals_corrected = np.clip(pvals * float(ntests), 0, 1)

    # reorder pvals and reject to original order of pvals
    pvals_corrected_ = np.empty_like(pvals_corrected)
    pvals_corrected_[sorted] = pvals_corrected
    reject_ = np.empty_like(reject)
    reject_[sorted] = reject

    return reject_, pvals_corrected_, alphacBonf


#############################################################
# Running time model functions

def get_running_times(data, memsize=3, filter=[0, 3600], tooshort=0.1):
    """get waiting times from data"""
    running_times = {k: [] for k in meankeys(generate_targetList(seq_len=memsize)[::-1])}
    for i in range(len(data)):
        if data[i][1] == 'run':
            if filter[0] <= data[i][0] <= filter[1] and data[i][3] != 0:
                if data[i][3] > tooshort:  # filter out runs shorter than 0.5s
                    try:
                        avg_rwd = round(np.mean([data[i-n-1][2] for n in range(1, (memsize*2)+1, 2)]), 2)
                        running_times[avg_rwd].append(data[i][3])
                    except:  # put the first n runs in rwd=1 (because we don't have the previous n runs to compute the average reward)
                        running_times[1].append(data[i][3])
    return running_times


# separate the data into time and reward bins
def prepare_data_running_times(sequence, animalList, sessionList, memsize=3, time_bins=6):
    bin_size = 3600/time_bins
    targetlist = generate_targetList(memsize)[::-1]
    temp_data = {}
    for bin in range(time_bins):
        temp_data[bin] = {}
        for animal in animalList:
            temp_data[bin][animal] = {k: [] for k in meankeys(targetlist)}
            for session in matchsession(animal, sessionList):
                temp_data[bin][animal] = combine_dict(temp_data[bin][animal], get_running_times(sequence[animal, session], memsize=memsize, filter=[bin*bin_size, (bin+1)*bin_size]))
    data = {}
    for animal in animalList:
        data[animal] = np.zeros((time_bins, len(meankeys(targetlist)))).tolist()
        for i, avg in enumerate(meankeys(targetlist)):  # 1 -> 0
            for bin in range(time_bins):
                data[animal][bin][i] = np.asarray(temp_data[bin][animal][avg])
    return data


def crit_cauchy(params, *args, robustness_param=1e-20):
    """negative log likelihood function for Wald distribution"""
    mu, sigma = params
    x = args
    pdf_vals = stats.cauchy.pdf(x, loc=mu, scale=sigma) + robustness_param
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val


def cauchy_fit(x, mu_init=1, sigma_init=.1):
    """fit Cauchy distribution"""
    params_init = np.array([mu_init, sigma_init])
    res = minimize(crit_cauchy, params_init, args=x, bounds=((0, None), (1e-8, None)))
    return res.x, res.fun


def modelrun_crit(params, *args, robustness_param=1e-20):
    """negative log likelihood function for full model"""
    mu, sigma, mu_prime, sigma_prime, mu_second, sigma_second = params
    neg_log_lik_val = 0
    N_bins, N_avg = args[1]
    MU = np.zeros((N_bins, N_avg))
    SIGMA = np.zeros((N_bins, N_avg))

    for bin in range(N_bins):
        for avg in range(N_avg):
            MU[bin, avg] = mu + bin*mu_prime + avg*mu_second
            SIGMA[bin, avg] = sigma + bin*sigma_prime + avg*sigma_second

    for bin in range(N_bins):
        for avg in range(N_avg):
            _mu = MU[bin, avg]# if MU[bin, avg] > 0 else 1e-8
            _sigma = SIGMA[bin, avg] if SIGMA[bin, avg] > 0 else 1e-8

            pdf_vals = stats.cauchy.pdf(args[0][bin][avg], scale=_sigma, loc=_mu,)
            ln_pdf_vals = np.log(pdf_vals + robustness_param)
            log_lik_val = ln_pdf_vals.sum()

            n = len(args[0][bin][avg]) if len(args[0][bin][avg]) > 0 else 1
            neg_log_lik_val += (-log_lik_val / n)
            # except:
            #     neg_log_lik_val += 0  # add 0 instead of throwing an error when there is no data in a bin*avg
    return neg_log_lik_val


def modelrun_compare(params, *args, robustness_param=1e-20):
    """BIC to compare models with different number of parameters and curves"""
    mu, sigma, mu_t, sigma_t, mu_R, sigma_R = params
    BIC = 0
    N = 0
    sum_log_likelihood = 0

    N_bins, N_avg = args[1]
    N_params = args[2]
    MU = np.zeros((N_bins, N_avg))
    SIGMA = np.zeros((N_bins, N_avg))

    for bin in range(N_bins):
        for avg in range(N_avg):
            MU[bin, avg] = mu + bin*mu_t + avg*mu_R
            SIGMA[bin, avg] = sigma + bin*sigma_t + avg*sigma_R

    for bin in range(N_bins):
        for avg in range(N_avg):
            _mu = MU[bin, avg] if MU[bin, avg] > 0 else 1e-8
            _sigma = SIGMA[bin, avg] if SIGMA[bin, avg] > 0 else 1e-8
            # try:
            pdf_vals = stats.cauchy.pdf(args[0][bin][avg], loc=_mu, scale=_sigma)
            ln_pdf_vals = np.log(pdf_vals + robustness_param)
            log_lik_val = ln_pdf_vals.sum()

            n = len(args[0][bin][avg]) if len(args[0][bin][avg]) > 0 else 1

            N += n
            sum_log_likelihood += log_lik_val

    k = N_params
    BIC = k * np.log(N) - 2 * sum_log_likelihood
    return BIC


def modelrun_fit(data, init=[1, 1, 1, 1, 1, 1], f=modelrun_crit,
                 N_bins=6, N_avg=4, N_params=2,
                 mu_t_fixed=False, sigma_t_fixed=False,
                 mu_U_fixed=False, sigma_U_fixed=False):
    params_init = np.array(init)
    mu_t_bounds = (None, None) if not mu_t_fixed else (0, 1e-8)
    sigma_t_bounds = (None, None) if not sigma_t_fixed else (0, 1e-8)
    mu_U_bounds = (None, None) if not mu_U_fixed else (0, 1e-8)
    sigma_U_bounds = (None, None) if not sigma_U_fixed else (0, 1e-8)

    res = minimize(f, params_init, args=(data, [N_bins, N_avg], N_params),
                   bounds=((None, None), (None, None),
                           mu_t_bounds, sigma_t_bounds,
                           mu_U_bounds, sigma_U_bounds))
    return res.x, res.fun


def process_error_idle_time(root, animalList, sessionList, n_simul=10000):
    N_bins = 6
    N_avg = 4
    _alpha, _alpha_t, _alpha_u, _gamma, _gamma_t, _gamma_u, _, _, _, _, _, _ = pickle.load(open("picklejar/ModelsFitsAllRats.p", "rb"))
    blocks =  [[0, 300],  [300, 600],  [600, 900],  [900, 1200],
                [1200, 1500],  [1500, 1800],  [1800, 2100],  [2100, 2400],
                [2400, 2700],  [2700, 3000],  [3000, 3300],  [3300, 3600]]
    error = {}

    for animal in animalList:
        print(f'Computing for {animal}')
        sessions = matchsession(animal, sessionList)
        data = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i, session in enumerate(sessions):
            example_idleTimeInLeftBin, example_idleTimeInRightBin = get_from_pickle(root, animal, session, name="timeinZone.p")
            for j in range(0, 12):
                data[j] = np.append(data[j], example_idleTimeInLeftBin[j]+example_idleTimeInRightBin[j])
    
        ex_alpha = _alpha[animal]['120']
        ex_alpha_t = _alpha_t[animal]['120']
        ex_alpha_u = _alpha_u[animal]['120']
        ex_gamma = _gamma[animal]['120']
        ex_gamma_t = _gamma_t[animal]['120']
        ex_gamma_u = _gamma_u[animal]['120']

        ALPHA = np.zeros((N_bins, N_avg))
        GAMMA = np.zeros((N_bins, N_avg))
        for bin in range(N_bins):
            for avg in range(N_avg):
                ALPHA[bin, avg] = ex_alpha + bin*ex_alpha_t + avg*ex_alpha_u
                GAMMA[bin, avg] = ex_gamma + bin*ex_gamma_t + avg*ex_gamma_u

        a = []
        g = []
        for i in range(6):
            a.append(.9*ALPHA[i][0]+0.1*ALPHA[i][1])
            a.append(.9*ALPHA[i][-1]+0.1*ALPHA[i][-2])
            g.append(.9*GAMMA[i][0]+0.1*GAMMA[i][1])
            g.append(.9*GAMMA[i][-1]+0.1*GAMMA[i][-2])


        error[animal] = np.zeros((n_simul, 12))
        for _ in range(n_simul):
            # if _ // 1000 == 0:
            #     print(f'Rat: {animal} || {_}/{n_simul}')
            res = [np.median(generate_idle_time(a[i], g[i], len(data[i]), seed=_)) for i in range(len(blocks))]   
            error[animal][_] = [np.sqrt((np.median(data[i]) - res[i])**2) for i in range(0, len(blocks))]
    
    return error



def process_error_crossing_time(root, animalList, sessionList, n_simul=10000):
    N_bins = 6
    N_avg = 4
    _, _, _, _, _, _, _mu, _mu_t, _mu_u, _sigma, _sigma_t, _sigma_u = pickle.load(open("picklejar/ModelsFitsAllRats.p", "rb"))
    blocks =  [[0, 300],  [300, 600],  [600, 900],  [900, 1200],
                [1200, 1500],  [1500, 1800],  [1800, 2100],  [2100, 2400],
                [2400, 2700],  [2700, 3000],  [3000, 3300],  [3300, 3600]]
    error = {}

    for animal in animalList:
        print(f'Computing for {animal}')
        sessions = matchsession(animal, sessionList)
        data = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i, session in enumerate(sessions):
            example_runningTimeInLeftBin, example_runningTimeInRightBin = get_from_pickle(root, animal, session, name="timeRun.p")
            for j in range(0, 12):
                data[j] = np.append(data[j], example_runningTimeInLeftBin[j]+example_runningTimeInRightBin[j])
    
        ex_mu = _mu[animal]['120']
        ex_mu_t = _mu_t[animal]['120']
        ex_mu_u = _mu_u[animal]['120']
        ex_sigma = _sigma[animal]['120']
        ex_sigma_t = _sigma_t[animal]['120']
        ex_sigma_u = _sigma_u[animal]['120']


        MU = np.zeros((N_bins, N_avg))
        SIGMA = np.zeros((N_bins, N_avg))
        for bin in range(N_bins):
            for avg in range(N_avg):
                MU[bin, avg] = ex_mu + bin*ex_mu_t + avg*ex_mu_u
                SIGMA[bin, avg] = ex_sigma + bin*ex_sigma_t + avg*ex_sigma_u

        m = []
        s = []
        for i in range(6):
            m.append((.9*MU[i][0] + 0.1*MU[i][1]))
            m.append((.9*MU[i][-1] + 0.1*MU[i][-2]))
            s.append((.9*SIGMA[i][0] + 0.1*SIGMA[i][1]))
            s.append((.9*SIGMA[i][-1] + 0.1*SIGMA[i][-2]))

        error[animal] = np.zeros((n_simul, 12))
        for _ in range(n_simul):
            res = [np.median(generate_running_time(m[i], s[i], len(data[i]), seed=_)) for i in range(len(blocks))]
            error[animal][_] = [np.sqrt((np.median(data[i]) - res[i])**2) for i in range(0, len(blocks))]
    
    return error


def LLratio_vs_complete(ablation_losses, keys, animalList, ax=None):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    LL_sums = [np.sum([ablation_losses[animal][key] for animal in animalList]) for key in keys]
    LL_complete = LL_sums[0]
    LL_reduced = LL_sums[1:]
    dParams = [np.sum(key) for key in keys][1:]

    p_vals = np.zeros(len(dParams))
    for i, (reducted_model_loss, df) in enumerate(zip(LL_reduced, dParams)):
        LR = -2*(reducted_model_loss - LL_complete)
        p_vals[i] = stats.chi2.sf(LR, df)

    sig, corrected_pvals, alphabonf = multipletests_bonferroni(p_vals)
    for i, p_val in enumerate(corrected_pvals):
        barplot_annotate_brackets(ax, 0, i+1, p_val, np.arange(1, 7), 
                                  [-LL_complete]*6, 
                                  dh=0.15+i*.1, barh=.025, maxasterix=None)

