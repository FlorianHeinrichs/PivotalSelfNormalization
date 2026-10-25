#
# tests/best_approx_test.py
#
# Project: Self-Normalization for Change Detection in Non-Stationary Time Series
# Date: 2026-06-09
# Author: Florian Heinrichs

from time import time

import numpy as np
from scipy.optimize import linprog


def best_approx_test(X: np.ndarray,
                     weights: np.ndarray,
                     quantile: float,
                     p_max: int = 0,
                     k_max: int = 10) -> tuple:
    """
    Apply self-normalized tests to X.

    :param X: Time series given as NumPy array of shape (n_time,), where n_time
        is the length of the time series.
    :param weights: Weights for calculation of test statistic and quantiles.
    :param quantile: Quantiles of tests, given as dictionary (for multiple tests
        or a single value for one test). If the quantile is not provided, it is
        approximated when the test is applied.
    :param p_max: Number of null hypotheses to test.
    :param k_max: Number of alternatives to consider for each null.
    :return: First non-rejected null hypothesis as int (None if none), and run time.
    """
    start = time()
    n = len(X)
    partial_sum = np.cumsum(X) / n
    segments = optimize_all_segments(partial_sum, version='linear')

    statistics = np.sqrt(n) * calculate_statistics(segments, n, k_max + p_max + 1)

    for p in range(p_max + 1):
        test_stat = np.max(statistics[p] / statistics[p + 1: p + k_max + 1] * weights)
        test_decision = test_stat > quantile[p]

        if not test_decision:
            not_rejected = p
            break
    else:
        not_rejected = None

    end = time()

    return not_rejected, end - start


def calculate_statistics(segments: dict, n: int, k_max: int) -> float:
    """
    Calculate N_k(S_n), for k = 0, ..., k_max based on pre-calculated segments
    using dynamic programming. More specifically, let dp[i][j] be the min-max
    error for covering the first i points of S_n with j segments. Then dp[i][1]
    is the best single-segment fit for the first i points. In general:

        dp[i][j] = min_{m < i} max(dp[m][j-1], error([m, i])),

    where error([m, i]) is the minimax error for the segment [m, i]. The final
    statistics are N_k(S_n) = dp[n][k+1].

    :param segments: Pre-calculated segments.
    :param k_max: Maximal order of statistic.
    :return: Value of statistics.
    """
    # Initialize dynamic programming table
    dp = np.full((n, k_max + 1), np.inf)

    # Base case: 1 segment
    for i in range(1, n + 1):
        dp[i - 1, 0] = segments.get((0, i), np.inf)

    # Fill DP table
    for j in range(1, k_max + 1):  # j + 1 segments
        for i in range(2, n + 1):  # point i
            for m in range(1, i):  # point m
                error = max(dp[m - 1, j - 1], segments.get((m, i), np.inf))
                if error < dp[i - 1, j]:
                    dp[i - 1, j] = error

    return dp[-1]


def optimize_all_segments(x: np.ndarray, version: str = 'linear') -> dict:
    """
    For all segments [i/n, j/n], for i < j, find the best affine linear
    approximation.

    :param x: Function values as numpy array of shape (n,).
    :param version: 'linear' or 'monotone' approximation.
    :return: Dictionary containing pairs (i, j) as keys and the min-max
        deviation as values.
    """
    n = len(x)
    t = np.linspace(0, 1, n + 1)[1:]

    if version == 'linear':
        segments = {(i, j): optimize_segment_lin(x[i: j], t[i: j])[2]
                    for i in range(n) for j in range(i + 1, n + 1)}
    else:
        segments = {(i, j): optimize_segment_mon(x[i: j], t[i: j])
                    for i in range(n) for j in range(i + 1, n + 1)}

    return segments


def optimize_segment_lin(x: np.ndarray, t: np.ndarray) -> tuple:
    """
    Solve inf_{a, b} sup_{t} |x(t) - a*t - b| for discrete x and t.

    Minimize: d
    Subject to: |x(t_i) - a * t_i - b| <= d
    Rewritten as:
      a * t_i + b - d <= x(t_i)
      - a * t_i - b - d <= - x(t_i)

    :param x: Function values as numpy array of shape (n,).
    :param t: Time grid as numpy array of shape (n,).
    :return: Tuple (a, b, max_error), where (a, b) are the best affine
        parameters and max_error is the minimum maximum deviation.
    """
    if len(x) != len(t):
        raise ValueError('x and t must be the same length')

    n = len(x)

    c = np.array([0.0, 0.0, 1.0])  # (a, b, d)
    A_i = np.stack((t, np.ones(n), -np.ones(n)), axis=-1)
    A_ii = np.stack((-t, -np.ones(n), -np.ones(n)), axis=-1)

    A_ub = np.concatenate((A_i, A_ii))
    b_ub = np.concatenate((x, -x))

    bounds = [(None, None), (None, None), (0.0, None)]  # a, b in \R, d \ge 0
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    if not result.success:
        raise RuntimeError(f"Linear program did not converge: {result.message}")

    a, b, d = result.x

    return a, b, d


def optimize_segment_mon(x: np.ndarray, t: np.ndarray) -> float:
    """
    Solve inf_{f} sup_{t} |x(t) - f(t)| for discrete x and t, where f ranges
    over all monotone functions.

    :param x: Function values as numpy array of shape (n,).
    :param t: Time grid as numpy array of shape (n,).
    :return: Minimum maximum deviation.
    """
    if len(x) != len(t):
        raise ValueError('x and t must be the same length')

    cumin = np.minimum.accumulate(x, axis=-1)
    cumax = np.maximum.accumulate(x, axis=-1)
    sup_up = np.max(x - cumin, axis=-1)
    sup_down = np.max(cumax - x, axis=-1)

    d = 0.5 * min(sup_up, sup_down)

    return d


def approximate_quantiles(weights: np.ndarray,
                          p_max: int = 0,
                          k_max: int = 10,
                          n_repetitions: int = 1000,
                          n_grid: int = 1000,
                          alpha: float = 0.05) -> float:
    """
    Approximate quantiles of the various test statistics.

    :param weights: Weights for calculation of test statistic and quantiles.
    :param p_max: Number of null hypotheses to test.
    :param k_max: Number of alternatives to consider for each null.
    :param n_repetitions: Number of simulated trajectories.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :param alpha: Level of the quantile.
    :return: Quantile as float.
    """
    ratio = _simulate_ratio(weights, p_max=p_max, k_max=k_max,
                            n_repetitions=n_repetitions, n_grid=n_grid)
    quantile = np.sort(ratio, axis=0)[int(n_repetitions * (1 - alpha)) - 1]

    return quantile


def _simulate_ratio(weights: np.ndarray,
                    p_max: int = 0,
                    k_max: int = 10,
                    n_repetitions: int = 1000,
                    n_grid: int = 1000) -> np.ndarray:
    """
    Simulate limiting distribution, which is the ratio of two suprema.

    :param weights: Weights for calculation of test statistic and quantiles.
    :param p_max: Number of null hypotheses to test.
    :param k_max: Number of alternatives to consider for each null.
    :param n_repetitions: Number of simulated ratios.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :return: Simulated limiting distribution.
    """
    W = simulate_brownian_motions(n_repetitions=n_repetitions,
                                  n_grid=n_grid)

    num = np.max(np.abs(W), axis=-1)
    ratios = []

    for i, numerator in enumerate(num):
        segments = optimize_all_segments(W[i], version='monotone')
        denominators = calculate_statistics(segments, n_grid, k_max + 2 * p_max)

        ratio = [np.max(
            numerator / denominators[2 * p: 2 * p + k_max] * weights
        ) for p in range(p_max + 1)]
        ratios.append(ratio)

    return np.array(ratios)


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
    n = 1000
    epsilon = 0.001
    k_max = 10
    p_max = 3

    weights = np.arange(1, k_max + 1) ** (-1 / 2 - epsilon)
    quantile = approximate_quantiles(weights, p_max=p_max,
                                     k_max=k_max, alpha=0.05)

    mu1 = np.ones(n)
    mu2 = np.ones(n)
    mu2[n // 2:] = 0
    mu3 = np.sin(6 * np.pi * np.linspace(0, 1, n))

    for mu in [mu1, mu2, mu3]:
        eps = np.random.randn(n, 100)
        x = mu[:, None] + eps
        result = best_approx_test(x, weights, quantile, p_max=p_max, k_max=k_max)
        print(result)
