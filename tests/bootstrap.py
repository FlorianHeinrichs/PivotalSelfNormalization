#
# tests/bootstrap.py
#
# Project: Self-Normalization for Change Detection in Non-Stationary Time Series
# Date: 2026-04-14
# Author: Florian Heinrichs
#
# Test based on bootstrap procedure:
# Zhou, Z. (2013). Heteroscedasticity and autocorrelation robust structural
# change detection. Journal of the American Statistical Association, 108(502),
# 726-740.


from time import time

import numpy as np


def bootstrap_test(X: np.ndarray,
                   alpha: float = 0.05,
                   n_trajectories: int = 1000) -> tuple:
    """
    Conduct bootstrap-based test.

    :param X: Time series given as NumPy array of shape (n_time,), where n_time
        is the length of the time series.
    :param alpha: Level of test.
    :param n_trajectories: Number of bootstrap trajectories, used for
        approximation of quantiles.
    :return: Tuple of test decision, p-value and computation time.
    """
    bw = calculate_bandwidth(X)

    start = time()
    result = calculate_test_bootstrap(
        X, bw, alpha, n_trajectories=n_trajectories
    )
    end = time()
    test_decision = result[0]

    return test_decision, end - start


def calculate_bandwidth(X: np.ndarray, m_grid: np.ndarray = None) -> np.ndarray:
    """
    Function to select the nuisance parameter m, that is, the block length of
    the bootstrap approximations.

    :param X: Time series given as NumPy array of shape (n_time,), where n_time
    is the length of the time series.
    :param m_grid: Grid of potential block lengths.
    :return:
    """
    n = len(X)
    partial_sum = np.cumsum(X)

    if m_grid is None:
        m_grid = np.arange(1, int(n ** (2 / 3)) + 1)

    gamma = []
    m_max = np.max(m_grid)

    for m in m_grid:
        if m > 1:
            partial_sum_m = partial_sum[(m-1):] - partial_sum[:-(m-1)]
        else:
            partial_sum_m = X

        gamma_m = (np.cumsum((partial_sum_m - m / n * partial_sum[-1]) ** 2)
                   / (m * (n - m + 1)))
        gamma.append(gamma_m[:n - m_max])

    gamma = np.array(gamma)

    standard_error = np.lib.stride_tricks.sliding_window_view(
        gamma, window_shape=(7,), axis=0).std(axis=2)

    maximal_se = np.max(standard_error, axis=1)
    m_optimal = m_grid[np.argmin(maximal_se) + 3]

    return m_optimal


def calculate_test_bootstrap(X: np.ndarray,
                             bw: int,
                             alpha: float = 0.05,
                             n_trajectories: int = 1000) -> tuple:
    """
    Calculate test statistic of test from [2].

    :param X: Time series given as NumPy array of size (n,).
    :param bw: Bandwidth of kernel estimator.
    :param alpha: Level of test.
    :param n_trajectories: Number of bootstrap trajectories, used for
        approximation of quantiles.
    :return: Returns test decision and p-value.
    """
    n = len(X)
    partial_sum = np.cumsum(X)
    cusum_stat = np.max(
        np.abs(partial_sum - np.arange(1, n + 1) / n * partial_sum[-1])
    )

    random_multipliers = np.random.randn(n_trajectories, n - bw + 1)

    if bw > 1:
        partial_sum_bw = partial_sum[(bw - 1):] - partial_sum[:-(bw - 1)]
    else:
        partial_sum_bw = X

    phi = np.cumsum((partial_sum_bw - bw / n * partial_sum[-1])[None, :]
                    * random_multipliers, axis=-1) / np.sqrt(bw * (n - bw + 1))
    phi_max = np.max(np.abs(phi[:, bw:] - np.arange(bw + 1, n - bw + 2)
                            / (n - bw + 1) * phi[:, -1:]), axis=-1)
    phi_ordered = np.sort(phi_max)

    critical_value = phi_ordered[int(np.floor(n_trajectories * (1 - alpha)))]

    test_decision = cusum_stat / np.sqrt(n) > critical_value
    p_value = 1 - np.sum(phi_max <= (cusum_stat / np.sqrt(n))) / n_trajectories

    return test_decision, p_value


if __name__ == '__main__':
    n = 1000
    mu1 = np.ones(n)
    mu2 = np.ones(n)
    mu2[n // 2:] = 0
    mu3 = np.sin(6 * np.pi * np.linspace(0, 1, n))

    for mu in [mu1, mu2, mu3]:
        results = []
        for _ in range(100):
            eps = np.random.randn(n)
            x = mu + eps
            result = bootstrap_test(x)
            results.append(result)

        print(f"Empirical rejection rate: {np.mean([res[0] for res in results])}")
        print(f"Mean p-value: {np.mean([res[1] for res in results])}")