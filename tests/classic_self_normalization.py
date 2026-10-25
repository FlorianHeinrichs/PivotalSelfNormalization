#
# tests/classic_self_normalization.py
#
# Project: Self-Normalization for Change Detection in Non-Stationary Time Series
# Date: 2026-10-08
# Author: Florian Heinrichs
#
# Test based on classic self-normalization:
# Shao, X., & Zhang, X. (2010). Testing for change points in time series.
# Journal of the American Statistical Association, 105(491), 1228-1240.


from time import time

import numpy as np


def self_normalization_test(X: np.ndarray, quantile: float) -> tuple:
    """
    Conduct change point test based on self-normalization in the AMOC ("At Most
    One Change") setting.

    :param X: Functional observations, given as NumPy array with shape
        (n_time, n_space), where n_time is the number of functional
        observations and n_space is the number of grid points.
        If multiple time series are provided simultaneously, the shape is
        (n_ts, n_time, n_space).
    :param X: Time series given as NumPy array of shape (n_time,), where n_time
        is the length of the time series.
    :param quantile: Quantile used for decision rule.
    :return: Tuple of test decision and computation time.
    """
    if X.ndim != 1:
        raise ValueError("X must be a one-dimensional time series.")

    start = time()
    n = len(X)
    T_n = np.cumsum(X) / n

    indices = np.arange(1, n + 1)
    numerator = np.sqrt(n) * (T_n - indices / n * T_n[-1])

    i, k = indices[:, None], indices[None, :]
    Ti, Tk = T_n[:, None], T_n[None, :]
    mask = i <= k
    first_sum = np.sum(((Ti - (i / k) * Tk) * mask) ** 2, axis=0)
    Ti, Tk = (T_n[-1] - T_n)[:, None], (T_n[-1] - T_n)[None, :]
    second_sum = np.zeros_like(T_n)
    second_sum[:-1] = np.sum(
        ((Ti - (n - i) / (n - k[:, :-1]) * Tk[:, :-1])
         * (~mask[:, :-1])) ** 2, axis=0
    )

    denominator = np.sqrt(first_sum + second_sum)

    test_statistic = np.max(np.abs(numerator) / denominator)
    test_decision = test_statistic > quantile
    end = time()

    return test_decision, end - start


def simulate_quantile(alpha: float, n_repetitions: int = 1000,
                      n_grid: int = 1000) -> float:
    """
    Simulate (1 - alpha) quantile of limiting distribution.

    :param alpha: Level of quantile.
    :param n_repetitions: Number of repetitions.
    :param n_grid: Number of grid points per Brownian motion.
    :return: (1 - alpha) quantile of limiting distribution.
    """
    realizations = simulate_ratio(n_repetitions, n_grid)
    quantile = np.sort(realizations)[int(n_repetitions * (1 - alpha))]

    return quantile


def simulate_ratio(n_repetitions: int = 1000,
                   n_grid: int = 1000) -> np.ndarray:
    """
    Simulate realizations of limiting distribution.

    :param n_repetitions: Number of repetitions.
    :param n_grid: Number of grid points per Brownian motion.
    :return: (1 - alpha) quantile of limiting distribution.
    """
    bm = simulate_brownian_motions(n_repetitions, n_grid)

    indices = np.arange(1, n_grid + 1) / n_grid
    numerator = bm - indices[None, :] * bm[:, -1:]

    i, k = indices[None, :, None], indices[None, None, :]
    bmi, bmk = bm[..., None], bm[:, None, :]
    mask = i <= k
    first_int = np.sum(((bmi - (i / k) * bmk) * mask) ** 2, axis=1) / n_grid
    bmi, bmk = (bm[:, -1:] - bm)[..., None], (bm[:, -1:] - bm)[:, None, :]
    second_int = np.zeros_like(bm)
    second_int[:, :-1] = np.sum(((bmi - (1 - i) / (1 - k[..., :-1]) * bmk[..., :-1])
                                 * (1 - mask[..., :-1])) ** 2, axis=1) / n_grid
    denominator = np.sqrt(first_int + second_int)

    realizations = np.max(np.abs(numerator) / denominator, axis=1)

    return realizations


def simulate_brownian_motions(n_repetitions: int = 1000,
                              n_grid: int = 1000) -> np.ndarray:
    """
    Simulate trajectories of (independent) Brownian motions.

    :param n_repetitions: Number of trajectories.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :return: Simulated trajectories as NumPy array.
    """
    rng = np.random.default_rng()

    increments = (rng.standard_normal(size=(n_repetitions, n_grid))
                  / np.sqrt(n_grid))
    increments[:, 0] = 0
    W = np.cumsum(increments, axis=-1)

    return W


if __name__ == '__main__':
    alpha = 0.05
    n = 1000
    mu1 = np.ones(n)
    mu2 = np.ones(n)
    mu2[n // 2:] = 0
    mu3 = np.sin(6 * np.pi * np.linspace(0, 1, n))
    quantile = simulate_quantile(alpha, n_repetitions=100000, n_grid=100)

    for mu in [mu1, mu2, mu3]:
        results = []
        for _ in range(100):
            eps = np.random.randn(n)
            x = mu + eps
            result = self_normalization_test(x, quantile=quantile)
            results.append(result)

        print(f"Empirical rejection rate: {np.mean([res[0] for res in results])}")
        print(f"Mean p-value: {np.mean([res[1] for res in results])}")