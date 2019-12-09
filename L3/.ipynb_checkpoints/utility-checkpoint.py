# utility.py
import numpy as np
import matplotlib.pyplot as plt


def exp_normalize(log_dist):
    """
    Computes exp(log_dist) / sum_i(log_dist[i]) using the exp-normalize trick.
    
    Arguments:
        log_dist: np.ndarray, (N x K), log-distribution that will be normalized
        
    Returns:
        norm_dist: np.ndarray, (log_dist.shape), the normalized distribution
    """
    K = log_dist.shape[1]
    
    max_val = np.repeat(np.max(log_dist, axis=1)[:, np.newaxis], K, axis=1) # (N x K)
    
    norm_dist = np.exp(log_dist - max_val) # (N x K)

    sum_exp = norm_dist.sum(axis=1) # (N)
    norm_dist /= np.repeat(sum_exp[:, np.newaxis], K, axis=1)

    return norm_dist


def log_sum_exp(summands, axis=0):
    """
    Computes log(sum_i exp(summands[i])), using the log-sum-exp trick.
    
    Arguments:
        summands: np.ndarray, contains the summands
        axis: integer, the axis to sum over
        
    Returns:
        log_sum_exp: np.ndarray, the results of the summation
    """
    max_summand = np.max(summands)
    log_sum_exp = max_summand + np.log(np.exp(summands - max_summand).sum(axis=axis))
    
    return log_sum_exp


def random_choice_vectorized(p):
    """
    Samples from 
    
    Arguments:
        p: np.ndarray, (N x K), N posterior distributions
            over K components
            
    Returns:
        sample: np.ndarray, (N), sample[i] was sampled 
            from the i-th distribution over K elements
    """
    N, K = p.shape
    
    elements = np.arange(K)
    sample = np.zeros(N, dtype=int)
    
    cum_p = p.cumsum(axis=1)
    
    uniform_sample = np.random.rand(N, 1)
    
    sample = (uniform_sample >= cum_p).sum(axis=1)
    return sample


def prepare_plot():
    """
    Prepares the animated log-likelihood plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    ax.set_xlabel("Steps", fontweight='bold', fontsize=15)
    ax.set_ylabel("Log-likelihood", fontweight='bold', fontsize=15)
    line, = ax.plot([0], [0])
    
    lik = []
    
    return lik, fig, ax, line