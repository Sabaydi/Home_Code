# log_posteriors.py 

import numpy as np

from utility import exp_normalize, log_sum_exp

### Algorithm 1 ###

def posterior_zn(Pi, log_lik_zn_theta, log_lik_pi_theta):
    """
    Computes the posterior P(z_n | x_n, Pi, Theta) for all n
    and all k .
    
    Arguments:
        Pi: np.ndarray (K), mixing coefficients
        log_lik_zn_theta: np.ndarray, (N x K), log_lik_zn_theta[i, j] represents
            logP(x_i | z_i = j, Theta)
        log_lik_pi_theta: np.ndarray, (N), log_lik_pi_theta[i] represents
            logP(x_i | Pi, Theta)
    Returns:
        posteriors: np.ndarray, (N x K), posteriors[i, j] represents
            the posterior probability P(zn=j | xn, Pi, Theta)
    """
    K = Pi.shape[0]
    N = log_lik_pi_theta.shape[0]
    
    posteriors = np.repeat(np.log(Pi)[np.newaxis, :], N, axis=0)
    posteriors +=  log_lik_zn_theta 
    posteriors -= np.repeat(log_lik_pi_theta[:, np.newaxis], K, axis=1) # (N x K)

    return exp_normalize(posteriors)

### Algorithm 2 ###

def log_posterior_z_minus_n(Z, alpha, cluster_counts):
    """
    Computes the log-posterior logP(z_n = k | Z_minus_n) for all
    n and all k.
    
    Arguments:
        Z: np.ndarray (N), contains the cluster assignment of
            the documents
        alpha: np.ndarray (K), parameter for the prior distribution
            of Pi
        cluster_counts: np.ndarray (K), cluster_counts[i] is the number of
            documents in the i-th cluster
    Returns:
        log_posteriors: np.ndarray, (N x K), log_posteriors[i,j] represents
            logP(z_i = j | Z_minus_j)
    """
    N = Z.shape[0]
    
    alpha_counts = alpha + cluster_counts
    alpha_counts = np.repeat(alpha_counts[np.newaxis, :], N, axis=0) # (N x K)
    alpha_counts[np.arange(len(Z)), Z] -= 1
    
    log_posteriors = np.log(alpha_counts) - np.log(N - 1 + alpha.sum())
    
    return log_posteriors
    
    
def posterior_zn_2(log_lik_zn_theta, log_post_z_minus_n):
    """
    Computes the posterior P(z_n = k| x_n, Z_minus_n, Theta) for all
    n and all k.
    
    Arguments:
        log_lik_zn_theta: np.ndarray, (N x K), log_lik_zn_theta[i, j] represents
            logP(x_i | z_i = j, Theta)
        log_post_z_minus_n: np.ndarray, (N x K), log_post_z_minus_n[i, j] represents
            logP(z_i = j | Z_minus_j)
        
    Returns:
        posteriors: np.ndarray, (N x K), posteriors[i, j] represents
            P(z_i = j | x_i, Z_minus_i, Theta)
    """
    return exp_normalize(log_lik_zn_theta + log_post_z_minus_n)

### Algorithm 3 ###


def posterior_zn_3(log_post_z_minus_n, log_lik_zn_Xmn_Zmn):
    """
    Computes the posterior P(z_n = k| X, Z_minus_n) for a given
    n and all k.
    
    Arguments:
        log_post_z_minus_n: np.ndarray, (N x K), log_post_z_minus_n[i, j] represents
            logP(z_i = j | Z_minus_i)
        log_lik_zn_Xmn_Zmn: np.ndarray, (N x K), log_lik_zn_Xmn_Zmn[i, j] represents
            logP(x_i | z_i = j, X_minus_i, Z_minus_i)
        
    Returns:
        posteriors: np.ndarray, (N x K), posteriors[i, j] represents
            P(z_i = j | X, Z_minus_i)
    """
    N, K = log_post_z_minus_n.shape
    
    return exp_normalize(log_lik_zn_Xmn_Zmn + log_post_z_minus_n) # (N x K)
    
def altposterior_zn_3(log_post_z_minus_n, log_lik_zn_Xmn_Zmn):
    """
    Computes the posterior P(z_n = k| X, Z_minus_n) for a given
    n and all k.
    
    Arguments:
        log_post_z_minus_n: np.ndarray, (K), log_post_z_minus_n[i] represents
            logP(z_n = i | Z_minus_n)
        log_lik_zn_Xmn_Zmn: np.ndarray, (K), log_lik_zn_Xmn_Zmn[i] represents
            logP(x_n | z_n = i, X_minus_n, Z_minus_n)
        
    Returns:
        posteriors: np.ndarray, (K), posteriors[i] represents
            P(z_n = i | X, Z_minus_n)
    """
    log_dist = log_lik_zn_Xmn_Zmn + log_post_z_minus_n # (K)
    
    max_val = np.max(log_dist)
    
    norm_dist = np.exp(log_dist - max_val) # (K)

    sum_exp = norm_dist.sum() # (1)
    norm_dist /= sum_exp
    return norm_dist