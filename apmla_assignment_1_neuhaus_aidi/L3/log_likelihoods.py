# log_likelihoods.py

import numpy as np

from utility import exp_normalize, log_sum_exp

def log_likelihood_zn_theta(X_matrix, Theta):
    """
    Computes the log-likelihood logP(x_n | z_n = k, Theta) for
    every document x_n and every cluster k.
    
    Arguments:
        X_matrix: np.ndarray, (N x I), X_matrix[i, j] represents the 
            number of occurences of the j-th word of the dictionary in 
            the i-th document.
        Theta: np.ndarray, (K x I), likelihood parameters for all cluster
    
    Returns:
        log_likelihoods: np.ndarray, (N x K), log_likelihoods[i, j] represents
            logP(x_i | z_i = j, Theta)
    """
    N = X_matrix.shape[0]
    K = Theta.shape[0] 
    
    log_likelihoods = np.ones((N, K))   
    
    for n, document in enumerate(X_matrix):
        
        c_n = np.repeat(document[np.newaxis, :], K, axis=0) # K x I
        log_likelihoods[n, :] = (c_n * np.log(Theta)).sum(axis=1) # K
    
    return log_likelihoods


def log_likelihood_pi_theta(log_lik_zn_theta, Pi):
    """
    Computes the log-likelihood logP(x_n | Pi, Theta) for
    all documents.
    
    Arguments:
        log_lik_zn_theta: np.ndarray, (N x K), lik_zn_theta[i, j] represents
            logP(x_i | z_i = j, Theta) 
        Pi: np.ndarray (K), mixing coefficients
    
    Returns:
        log_lik_pi_theta: np.ndarray, (N), likelihoods[i] represents
            the log-likelihood logP(x_i | Pi, Theta)
    """
    summands = np.log(Pi) +  log_lik_zn_theta # (N x K)
    return log_sum_exp(summands, axis=1) 


def log_likelihood_zn_Xmn_Zmn(X_matrix, Z, gamma, K):
    """
    Computes the log-likelihood logP(x_n | z_n = k, X_-n, Z_-n) all n.
    
    Arguments:
        X_matrix: np.ndarray, (N x I), X_matrix[i, j] represents the 
            number of occurences of the j-th word of the dictionary in 
            the i-th document.
        Z: np.ndarray (N), contains the cluster assignments
        gamma: np.ndarray (I), parameter for the prior distribution
            of Theta
        K: integer, number of clusters 
        
    Returns:
        log_likelihoods: np.ndarray, (N x K), log_likelihoods[i]
            represents logP(x_i | z_i = j, X_-i, Z_-i) 
    """
    N, I = X_matrix.shape
    log_likelihoods = np.zeros((N, K))
    
    for k in range(K):
        
        # compute gamma_dash
        X_k = X_matrix[(Z == k), :]
        
        gamma_dash = np.repeat(X_k.sum(axis=0)[np.newaxis, :], N, axis=0) # nrows(X_k) x I 
        gamma_dash[(Z == k), :] -= X_k
        gamma_dash = gamma_dash + gamma
        
        # compute sum_m sum_i log(gamma_m + i)
        summands_i = np.copy(X_matrix)
        mask = (X_matrix > 0)
        summands_i -= (summands_i > 0)
        log_s_m = mask*np.log(gamma_dash + summands_i) # N x I
    
        while np.any(summands_i > 0):
            summands_i -= np.ones(X_matrix.shape)
            mask[summands_i < 0] = 0
            summands_i[summands_i < 0] = 0
            
            log_s_m += mask * np.log(gamma_dash + summands_i) # N x I
        log_s_m = np.sum(log_s_m, axis=1) # N
              
        # compute sum_j log(j + sum_m gamma_m)
        summands_j = np.copy(X_matrix).sum(axis=1) # N
        mask = (summands_j > 0)
        summands_j -= (summands_j > 0)
        gamma_sum = gamma_dash.sum(axis=1) # N
        log_w_n = mask*np.log(summands_j + gamma_sum)
    
        while np.any(summands_j > 0):
            summands_j -= np.ones(N)#(summands_j > 0)
            mask[summands_j < 0] = 0
            summands_j[summands_j < 0] = 0
            
            log_w_n += mask * np.log(summands_j + gamma_sum)
        
        # compute log-likelihoods
        log_likelihoods[:, k] = log_s_m - log_w_n

    return log_likelihoods