#
# data_generation.py
#
# Project: Self-Normalization for CUSUM-based Change Detection in Locally
#          Stationary Time Series
# Date: 2025-08-20
# Author: Florian Heinrichs
#
# Script to generate data, data generating processes based on the paper:
# - Heinrichs, F., & Dette, H. (2021). A distribution free test for changes in
#   the trend function of locally stationary processes. Electronic Journal of
#   Statistics, 15(2), 3762-3797.

import numpy as np


def generate_data(n: int, n_samples: int, mean: str,
                  error: str, sigma: str, std: float = 1.0) -> np.ndarray:
    """
    Wrapper function to generate noisy data.

    :param n: Length of generated time series.
    :param n_samples: Number of generated time series.
    :param mean: Type of mean in ['1', '2', 'abrupt', 'const'].
    :param error: Type of error in ['iid', 'ar', 'ma'].
    :param sigma: Type of sigma in ['0', '1', '2', '3'].
    :param std: Standard deviation of i.i.d. innovations.
    :return: NumPy array containing noisy data.
    """
    if mean == '0':
        mean = mu_abrupt(n)
    elif mean == '1':
        mean = mu_abrupt(n, cps={0.5: 1})
    elif mean == '2':
        mean = mu_abrupt(n, cps={0.25: 1, 0.5: -0.5, 0.75: 1})
    else:
        raise ValueError("Mean type unknown.")

    if error == 'iid':
        error = generate_iid(n, n_samples, std=std)
    elif error == 'ar':
        error = generate_ar(n, n_samples, std=std)
    elif error == 'ma':
        error = generate_ma(n, n_samples, std=std)
    elif error == 'loc_stat':
        error = generate_loc_stat(n, n_samples, std=std)
    else:
        raise ValueError("Error type unknown.")

    if sigma == '0':
        sigma = np.ones((n, 1)) / 2
    elif sigma == '1':
        sigma = sigma_1(n)
    elif sigma == '2':
        sigma = sigma_2(n)
    else:
        raise ValueError("Sigma type unknown.")

    return mean + sigma * error


def mu_abrupt(n: int, cps: dict = None) -> np.ndarray:
    """
    Generates a step function with change point at "cp" (defined on the unit
    interval).

    :param n: Number of supporting points (observations).
    :param cps: Dictionary containing position and height of change points.
    :return: Function values at points np.arange(n) / n
    """
    if cps is None:
        cps = {}

    mu = np.zeros((n, 1))

    for cp, cp_height in cps.items():
        mu[int(cp * n):] += cp_height

    return mu


def generate_iid(n: int, n_samples: int, std: float = 1.0,
                 dist: str = 'normal') -> np.ndarray:
    """
    Generates i.i.d. errors according to the specified distribution.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param std: Standard deviation of random variables.
    :param dist: Distribution of noise. Defaults to 'normal'.
    :return: Errors at points np.arange(n) / n. Output has shape (n, n_samples).
    """
    rng = np.random.default_rng()

    if dist in ['normal', 'Gaussian']:
        errors = rng.normal(loc=0, scale=1, size=(n, n_samples)) * std
    elif dist == 'uniform':
        c = np.sqrt(3) * std
        errors = rng.uniform(low=-c, high=c, size=(n, n_samples))

    return errors


def generate_ar(n: int, n_samples: int, burn_in: int = 100,
                a: float = 0.5, std: float = 1.0,
                dist: str = 'normal') -> np.ndarray:
    """
    Generates AR(1) errors.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param burn_in: Time steps used for burn in of AR process.
    :param a: Autoregressive coefficient.
    :param std: Standard deviation of i.i.d. innovations.
    :param dist: Distribution of innovations. Defaults to 'normal'.
    :return: Errors at points np.arange(n) / n. Output has shape (n, n_samples).
    """
    epsilon = generate_iid(n + burn_in, n_samples, std=std, dist=dist)
    errors = np.zeros((n + burn_in, n_samples))

    errors[0] = epsilon[0]
    for i in range(1, n + burn_in):
        errors[i] = a * errors[i - 1] + epsilon[i]

    var_errors = 1 / (1 - a ** 2)
    errors = errors[burn_in:] / np.sqrt(var_errors)

    return errors


def generate_ma(n: int, n_samples: int, a: float = 0.5,
                std: float = 1.0) -> np.ndarray:
    """
    Generates MA(1) errors.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param a: MA coefficient.
    :param std: Standard deviation of i.i.d. innovations.
    :return: Errors at points np.arange(n) / n. Output has shape (n, n_samples).
    """
    epsilon = generate_iid(n + 1, n_samples, std=std)
    errors = epsilon[1:] + a * epsilon[:-1]

    var_errors = 1 + a ** 2
    errors = errors / np.sqrt(var_errors)

    return errors


def generate_loc_stat(n: int, n_samples: int, std: float = 1.0) -> np.ndarray:
    """
    Generates mixture of AR(1) errors with uniform and normal distribution.

    :param n: Number of supporting points (observations).
    :param n_samples: Number of (independent) trajectories.
    :param std: Standard deviation of i.i.d. innovations.
    :return: Errors at points np.arange(n) / n. Output has shape (n, n_samples).
    """
    eps_normal = generate_ar(n, n_samples, a=0.5, std=std, dist='normal')
    eps_uniform = generate_ar(n, n_samples, a=-0.5, std=std, dist='uniform')

    a = 1 / 2 - np.cos(
        (-np.cos(np.pi * np.linspace(0, 1, n)) + 1) / 2 * np.pi
    )[:, np.newaxis] / 2

    errors = np.sqrt(a) * eps_normal + np.sqrt(1 - a) * eps_uniform

    return errors


def sigma_1(n: int) -> np.ndarray:
    """
    Generates a monotonically increasing function (defined on the unit
    interval).

    :param n: Number of supporting points (observations).
    :return: Function values at points np.arange(n) / n
    """
    x = np.linspace(0, 1, n)
    sigma = 1 / 4 + x / 2

    return np.expand_dims(sigma, axis=1)


def sigma_2(n: int) -> np.ndarray:
    """
    Generates a step function (defined on the unit interval).

    :param n: Number of supporting points (observations).
    :return: Function values at points np.arange(n) / n
    """
    x = np.linspace(0, 1, n)
    sigma = x ** 10

    return np.expand_dims(sigma, axis=1)
