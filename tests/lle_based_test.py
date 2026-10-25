#
# tests/lle_based_test.py
#
# Project: Self-Normalization for Change Detection in Non-Stationary Time Series
# Date: 2026-04-14
# Author: Florian Heinrichs
#
# Test based on local linear estimation:
# Heinrichs, F., & Dette, H. (2021). A distribution free test for changes in
# the trend function of locally stationary processes. Electronic Journal of
# Statistics, 15(2), 3762-3797.

from time import time

import numpy as np

try:
    from .utils import bandwidth_cv, convolve
except ImportError:
    from utils import bandwidth_cv, convolve


def kernel_based_sn(X: np.ndarray,
                    alpha: float = 0.05,
                    n_trajectories: int = 1000) -> tuple:
    """
    Wrapper function to do all three kernel based tests from [1] and [2].

    :param X: Time series given as NumPy array of shape (n_time,) or
        (n_time, n_ts), where n_time is the length of the time series and n_ts
        the number of observed time series.
    :param alpha: Level of test.
    :param n_trajectories: Number of repetitions/trajectories, used for
        approximation of quantiles.
    :return: Tuple of test decision, p-value and computation time.
    """
    n = X.shape[0]
    expand_dims = len(X.shape) == 1

    if expand_dims:
        X = np.expand_dims(X, axis=1)

    min_bw, max_bw = 2, int(n // 4)

    quantile = estimate_quantile_self_norm(alpha, n_repetitions=n_trajectories)

    estimator = lambda *y: jackknife_estimator(*y)[0]
    bandwidths = bandwidth_cv(X, min_bw, max_bw, estimator, batch_axis=1)

    test_decisions = []
    times = []

    for bw, x in zip(bandwidths, X.transpose()):
        start = time()
        result = calculate_test_self_norm(x, bw, quantile, return_p=False)
        times.append(time() - start)
        test_decisions.append(result)

    if expand_dims:
        return test_decisions[0], times[0]
    else:
        return test_decisions, times


def sequential_lle(X: np.ndarray,
                   bw: int,
                   bn: int = 20,
                   n_lambda: int = 5) -> np.ndarray:
    """
    Calculate sequential local linear estimator, as used in [2] for
    self-normalization.

    :param X: Time series given as numpy array of size (n_time, n_ts).
    :param bw: Bandwidth of the estimator as int.
    :param bn: Block length.
    :param n_lambda: Number of support points of nu. Calculate sequential
        estimator at grid points 1/n_lambda, ..., 1.
    :return: Returns the sequential local linear estimators as numpy array of
        size (n_time, n_lambda, n_ts) or (n_time, n_lambda) if only one time
        series is provided.
    """
    n = X.shape[0]
    ln = int(n // bn)

    if bn % n_lambda > 0:
        raise ValueError(f"{bn=} and {n_lambda=} not compatible.")

    ones = np.ones(bn // n_lambda)
    zeros = np.zeros(bn // n_lambda)
    filter_arrays = np.stack([
        np.tile(
            np.concatenate((np.tile(ones, i), np.tile(zeros, n_lambda - i))), ln
        ) for i in range(1, n_lambda + 1)
    ])

    n_rem = int(n - bn * ln)
    if n_rem > 0:
        final = np.zeros((n_lambda, n_rem))
        final[-1] = 1
        filter_arrays = np.concatenate((filter_arrays, final), axis=1)

    filter_arrays = filter_arrays.astype(bool)
    estimates = np.stack([jackknife_estimator(X, bw, filter_array=filter_array)[0]
                          for filter_array in filter_arrays], axis=1)

    return estimates


def local_linear_estimator(X: np.ndarray,
                           bw: int,
                           filter_array: np.ndarray = None) -> tuple:
    """
    Use local linear regression to estimate mu and its Frechet derivative.

    :param X: Time series given as numpy array.
    :param bw: Bandwidth of the estimator as int.
    :param filter_array: Filter array to leave out certain observations (used
        for cross validation).
    :return: Returns the local linear estimators.
    """
    spatial_dims = len(X.shape) - 1
    n_time = X.shape[0]

    if filter_array is None:
        filter_array = np.ones_like(X, dtype=bool)

    X_filtered = X.copy()
    X_filtered[~filter_array] = 0

    X_support = np.ones_like(X_filtered)
    X_support[~filter_array] = 0

    kernel_support, kernel = get_kernel(bw)
    kernel = kernel.reshape((1,) * spatial_dims + (-1,))
    supp_kern = kernel_support * kernel
    supp2_kern = kernel_support ** 2 * kernel

    padding = ((bw, bw),) + spatial_dims * ((0, 0),)
    X_filtered = np.pad(X_filtered, padding, mode='edge')
    X_support = np.pad(X_support, padding, mode='edge')

    S0 = convolve(X_support, kernel)
    S1 = convolve(X_support, supp_kern[..., ::-1])
    S2 = convolve(X_support, supp2_kern[..., ::-1])

    R0 = convolve(X_filtered, kernel)
    R1 = convolve(X_filtered, supp_kern[..., ::-1])

    denominator = S0 * S2 - S1 ** 2
    mu_hat = (S2 * R0 - S1 * R1) / (denominator + 1e-10)
    mu_prime_hat = (S0 * R1 - S1 * R0) / (bw / n_time * denominator + 1e-10)

    return mu_hat, mu_prime_hat


def jackknife_estimator(X: np.ndarray,
                        bw: int,
                        filter_array: np.ndarray = None) -> tuple:
    """
    Function to calculate the Jackknife version of the local linear estimators.

    :param X: Time series given as numpy array.
    :param bw: Bandwidth of the estimator.
    :param filter_array: Filter array to leave out certain observations (used
        for cross validation).
    :return: Returns the Jackknife estimators given as numpy array.
    """
    mu_hat, mu_prime_hat = local_linear_estimator(
        X, bw, filter_array=filter_array
    )

    mu_hat2, mu_prime_hat2 = local_linear_estimator(
        X, int(bw / np.sqrt(2)), filter_array=filter_array
    )

    mu_tilde = 2 * mu_hat2 - mu_hat
    mu_prime_tilde = (np.sqrt(2) / (np.sqrt(2) - 1) * mu_prime_hat2
                      - mu_prime_hat / (np.sqrt(2) - 1))

    return mu_tilde, mu_prime_tilde


def get_kernel(bw: int,
               mode: str = 'quartic',
               version: str = 'regular') -> (np.ndarray, np.ndarray):
    """
    Define kernel for kernel based estimators.

    :param bw: Bandwidth of the estimator as int.
    :param mode: Mode of the kernel given as string. Currently only the
        'quartic', 'triweight', and 'tricube' kernels are supported.
    :param version: Version of the kernel, either 'regular' or 'jackknife'.
    :return: Returns the kernel and its support as numpy arrays.
    :raises: ValueError if unsupported mode is chosen.
    """
    if version == 'regular':
        support = np.arange(-bw, bw + 1) / bw

        if mode == 'quartic':
            kernel = 15 / 16 * (1 - support ** 2) ** 2
        elif mode == 'triweight':
            kernel = 35 / 32 * (1 - support ** 2) ** 3
        elif mode == 'tricube':
            kernel = 70 / 81 * (1 - np.abs(support) ** 3) ** 3
        elif mode == 'triangular':
            kernel = (1 - np.abs(support))
        else:
            raise ValueError(f"{mode=} unknown.")

    elif version == 'jackknife':
        bw2 = int(bw // np.sqrt(2))
        support, kern = get_kernel(bw)
        kern2 = np.sqrt(8) * get_kernel(bw2)[1]
        n_diff = (len(kern) - len(kern2)) // 2
        kernel = np.pad(kern2, (n_diff, n_diff)) - kern

    else:
        raise ValueError(f"{version=} unknown.")

    return support, kernel

def get_kernel_2nd_drv(bw: int,
                       mode: str = 'quartic',
                       version: str = 'regular') -> (np.ndarray, np.ndarray):
    """
    Get 2nd derivative of kernel for kernel based estimators.

    :param bw: Bandwidth of the estimator as int.
    :param mode: Mode of the kernel given as string. Currently only the
        'quartic' kernel is supported.
    :param version: Version of the kernel, either 'regular' or 'jackknife'.
    :return: Returns the kernel's 2nd derivative and its support as numpy arrays.
    :raises: ValueError if unsupported mode is chosen.
    """
    if version == 'regular':
        support = np.arange(-bw, bw + 1) / bw

        if mode == 'quartic':
            kernel = 15 / 4 * (3 * support ** 2 - 1)
        else:
            raise ValueError(f"{mode=} unknown.")
    elif version == 'jackknife':
        bw2 = int(bw // np.sqrt(2))
        support, kern = get_kernel_2nd_drv(bw)
        kern2 = np.sqrt(8) * get_kernel_2nd_drv(bw2)[1]
        n_diff = (len(kern) - len(kern2)) // 2
        kernel = np.pad(kern2, (n_diff, n_diff)) - kern

    else:
        raise ValueError(f"{version=} unknown.")


    return support, kernel


def calculate_test_self_norm(X: np.ndarray, bw: int, quantile: float,
                             return_p: bool = False) -> tuple | bool:
    """
    Calculate test statistic of test from [2].

    :param X: Time series given as NumPy array of size (n,).
    :param bw: Bandwidth of kernel estimator.
    :param quantile: Quantile of limiting distribution, obtained from
        estimate_quantile_self_norm().
    :param return_p: Specify whether p-value is returned.
    :return: Returns test decision and p-value.
    """
    mu_tilde = sequential_lle(X, bw)
    mean = np.mean(X, axis=0, keepdims=True)
    mu_tilde = mu_tilde - np.expand_dims(mean, axis=-1)

    d2_discrete = np.sqrt(np.mean(mu_tilde ** 2, axis=0))
    lambda_values = np.arange(1, 5) / 5

    integral_term = np.mean(
        lambda_values * np.abs(d2_discrete[:-1] ** 2 - d2_discrete[-1] ** 2),
        axis=0
    )

    test_statistic = d2_discrete[-1] ** 2 / integral_term
    test_decision = test_statistic > quantile

    if return_p:
        p_value = estimate_p_value_self_norm(test_statistic)
        return test_decision, p_value
    else:
        return test_decision



def estimate_quantile_self_norm(alpha: float,
                                n_lambda: int = 5,
                                n_grid: int = 1000,
                                n_repetitions: int = 1000) -> float:
    """
    Monte Carlo simulation for quantiles of the test statistic from [2].

    :param alpha: Level of test.
    :param n_lambda: Number of support points of nu. Calculate sequential
        estimator at grid points 1/n_lambda, ..., 1.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :param n_repetitions: Number of repetitions of Monte Carlo simulation.
    :return: Approximated quantile.
    """
    rng = np.random.default_rng()

    increments = (rng.standard_normal(size=(n_repetitions, n_grid))
                  / np.sqrt(n_grid))
    increments[:, 0] = 0
    W = np.cumsum(increments, axis=-1)

    lambda_val = np.arange(1, n_lambda) / n_lambda
    lambda_ind = (lambda_val * n_grid - 1).astype(int)
    B = W[:, -1] / np.mean(
        np.abs(W[:, lambda_ind] - lambda_val[None, :] * W[:, -1, None]), axis=-1
    )

    quantile = np.sort(B)[int(n_repetitions * (1 - alpha)) - 1]

    return quantile


def estimate_p_value_self_norm(test_statistic: np.ndarray,
                               n_lambda: int = 5,
                               n_grid: int = 1000,
                               n_repetitions: int = 1000) -> float:
    """
    Monte Carlo simulation for quantiles of the test statistic from [2].

    :param test_statistic: Test statistic as Numpy array.
    :param n_lambda: Number of support points of nu. Calculate sequential
        estimator at grid points 1/n_lambda, ..., 1.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :param n_repetitions: Number of repetitions of Monte Carlo simulation.
    :return: Approximated quantile.
    """
    rng = np.random.default_rng()

    increments = (rng.standard_normal(size=(n_repetitions, n_grid))
                  / np.sqrt(n_grid))
    increments[:, 0] = 0
    W = np.cumsum(increments, axis=-1)

    lambda_val = np.arange(1, n_lambda) / n_lambda
    lambda_ind = (lambda_val * n_grid - 1).astype(int)
    B = W[:, -1] / np.mean(
        np.abs(W[:, lambda_ind] - lambda_val[None, :] * W[:, -1, None]), axis=-1
    )

    p_value = np.mean(test_statistic < B)

    return p_value


if __name__ == '__main__':
    n = 1000
    mu1 = np.ones(n)
    mu2 = np.ones(n)
    mu2[n // 2:] = 0
    mu3 = np.sin(6 * np.pi * np.linspace(0, 1, n))

    for mu in [mu1, mu2, mu3]:
        eps = np.random.randn(n, 100)
        x = mu[:, None] + eps
        result = kernel_based_sn(x)

        print(f"Empirical rejection rate: {np.mean(result[0])}")
        print(f"Mean p-value: {np.mean(result[1])}")