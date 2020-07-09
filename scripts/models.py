import numpy as np


def vectorborne(t, y, p, Nh):
    """
    """
    Ih, Iv = y
    alpha, beta_h, gamma_h, beta_v, lambda_v, mu_v = p
    Nv = lambda_v / mu_v + np.exp(-mu_v * t)
    # Nv = lambda_v / mu_v
    dihdt = alpha * beta_h * (Nh - Ih) * Iv - gamma_h * Ih
    divdt = alpha * beta_v * (Nv - Iv) * Ih - mu_v * Iv

    return [dihdt, divdt]


def vectorborneOld(t,y,params):
    ih, iv = y
    alpha_h, beta_h, alpha_v, lambda_v, mu_v, nh = params
    dihdt = alpha_h * (nh - ih) * iv - beta_h * ih
    divdt = alpha_v * ((np.exp(-mu_v * t) + lambda_v / mu_v) - iv) * ih - mu_v * iv
    return [dihdt,divdt]