#
# tests/best_approx_alternatives.py
#
# Project: Self-Normalization for Change Detection in Non-Stationary Time Series
# Date: 2026-04-21
# Author: Florian Heinrichs

from time import time

import numpy as np
from scipy.optimize import linprog, OptimizeResult

from best_approx_test import simulate_brownian_motions


def best_approx_test(X: np.ndarray,
                     alpha: float = 0.05,
                     version: str = 'all',
                     versions: list = None,
                     quantile: float | dict = None) -> dict:
    """
    Apply self-normalized tests to X.

    :param X: Time series given as NumPy array of shape (n_time,), where n_time
        is the length of the time series.
    :param alpha: Level of test.
    :param version: Version of the test to use, either of: 'bivariate' or
        'estimator'. Defaults to 'all', so that all tests are applied.
    :param quantile: Quantiles of tests, given as dictionary (for multiple tests
        or a single value for one test). If the quantile is not provided, it is
        approximated when the test is applied.
    :return:
    """
    start_num = time()
    num = compute_numerator(X)
    end_num = time()

    if version == 'all' and versions is None:
        versions = ['bivariate', 'estimator']
    elif versions is None:
        versions = [version]

    results = {}

    for version in versions:
        start_denom = time()
        denom = compute_denominator(X, version)
        test_statistic = num / denom
        end_denom = time()

        if isinstance(quantile, dict) and version in quantile:
            quantile_test = quantile.get(version)
        else:
            quantile_test = approximate_quantiles(alpha=alpha, version=version)

        test_decision = test_statistic > quantile_test
        total_time = end_denom - start_denom + end_num - start_num
        results[version] = {'test_decision': test_decision, 'time': total_time}

    return results


def compute_numerator(X: np.ndarray) -> float:
    """
    Calculate numerator of the proposed test statistic.

    Build the linear‑program:

        minimize   d
        subject to -d <= S_k - t_k c <= d     (k = 0,...,n)

    Two inequalities per k:
    (i)    (-t_k, -1) * (c, d)^T <= -S_k
    (ii)   ( t_k, -1) * (c, d)^T <=  S_k

    Unknown vector:  x = (c, d).

    :param X: Time series given as NumPy array of shape (n_time,), where n_time
    is the length of the time series.
    :return: Value of numerator.
    """
    n = len(X)

    partial_sum = np.concatenate(([0.0], np.cumsum(X))) / n
    t = np.linspace(0, 1, n + 1)

    res = _solve_lin_program(partial_sum, t)

    c_star, d_star = res.x
    numerator = d_star

    return numerator


def compute_denominator(X: np.ndarray, version: str = 'bivariate') -> float:
    """
    Calculate denominator D of the proposed test statistic.

    :param X: Time series given as NumPy array of shape (n_time,), where n_time
        is the length of the time series.
    :param version: Version of denominator (either 'bivariate' or 'estimator').
        If classic, build the linear‑program (+ minimization over s):

        minimize   d
        subject to -d <= S_k - t_k c_1 - (t_k - s)_+ c_2 <= d     (k = 0,...,n)

    Two inequalities per k:
    (i)    (-t_k, -(t_k - s)_+, -1) * (c_1, c_2, d)^T <= -S_k
    (ii)   ( t_k,  (t_k - s)_+, -1) * (c_1, c_2, d)^T <=  S_k

    Unknown vector:  x = (c_1, c_2, d).

    :return: Value of numerator.
    """
    n = len(X)

    if version == 'bivariate':
        S = _calculate_double_cusum_process(X)
        H = S[:, -1] - np.arange(1, n + 1) / n * S[-1, -1]
        denominator = np.max(np.abs(H))
    elif version == 'estimator':
        denominator = _lrv_estimator(X) / np.sqrt(n)
    else:
        raise NotImplementedError(f"Denominator {version} not implemented.")

    return denominator


def approximate_quantiles(n_repetitions: int = 1000,
                          n_grid: int = 1000,
                          alpha: float = 0.05,
                          version: str = 'classic') -> float:
    """
    Approximate quantiles of the various test statistics.

    :param n_repetitions: Number of simulated trajectories.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :param alpha: Level of the quantile.
    :param version: Indicating version of the test statistic (either 'classic'
        or 'bivariate').
    :return: Quantile as float.
    """
    ratio = _simulate_ratio(n_repetitions=n_repetitions, n_grid=n_grid,
                            version=version)
    quantile = np.sort(ratio)[int(n_repetitions * (1 - alpha)) - 1]

    return quantile

def _solve_lin_program(X: np.ndarray,
                       t: np.ndarray,
                       tp: np.ndarray = None) -> OptimizeResult:
    """
    Auxiliary function to set up and solve linear program.

    :param X: NumPy array containing a discretized function (usual the partial
        sum process).
    :param t: Discrete time of function.
    :param tp: Truncated time (used for denominator).
    :return:
    """
    if tp is None:
        A_i = np.stack((-t, -np.ones_like(t)), axis=-1)
        A_ii = np.stack((t, -np.ones_like(t)), axis=-1)
        c_obj = np.array([0.0, 1.0])
        bounds = [(None, None), (0.0, None)]
    else:
        A_i = np.stack((-t, -tp, -np.ones_like(t)), axis=-1)
        A_ii = np.stack(( t,  tp, -np.ones_like(t)), axis=-1)
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0.0,   None)]

    A_ub = np.concatenate((A_i, A_ii))
    b_ub = np.concatenate((-X, X))
    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    if not res.success:
        raise RuntimeError('LP did not converge: ' + res.message)
    else:
        return res


def _calculate_double_cusum_process(X: np.ndarray, bn: int = None) -> np.ndarray:
    """
    Calculate the double cusum process S_n(t, s) from X.

    :param X: Observed time series as NumPy array of size (n_time,).
    :param bn: Block length.
    :return: Double CUSUM process S_n(t, s) as NumPy array.
    """
    n = len(X)

    if bn is None:
        bn = int(n ** (1 / 3) + 1e-5)

    ln = int(n // bn)
    x1 = np.arange(n)
    x2 = ((np.arange(ln) * bn)[None, :] + np.arange(bn)[:, None]).flatten()
    x2 = np.concatenate((x2, np.arange(ln * bn, n)), axis=0)

    M = np.zeros((n, n))
    M[x1, x2] = X[x2]
    S = M.cumsum(axis=0).cumsum(axis=1) / n

    return S


def _lrv_estimator(X: np.ndarray, bn: int = None) -> np.ndarray:
    """
    Calculate block-difference estimator of the long-run variance.

    :param X: Observed time series as NumPy array of size (n_time,).
    :param bn: Block length.
    :return: Estimated (global) long-run variance
    """
    n = len(X)

    if bn is None:
        bn = int(n ** (1 / 3) + 1e-5)

    ln = n // bn
    X = X[: ln * bn]
    blocks = X.reshape(ln, bn)
    m = bn // 2

    left_sum = np.sum(blocks[:, :m], axis=-1)
    right_sum = np.sum(blocks[:, m:], axis=-1)

    diff = left_sum - right_sum
    estimate = np.sum(diff ** 2) / n

    return estimate


def _simulate_ratio(n_repetitions: int = 1000,
                    n_grid: int = 1000,
                    version: str = 'bivariate') -> np.ndarray:
    """
    Simulate limiting distribution, which is the ratio of two suprema.

    :param n_repetitions: Number of simulated ratios.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :param version: Indicating version of the test statistic (either 'classic'
        or 'bivariate').
    :return: Simulated limiting distribution.
    """
    if version == 'bivariate':
        n_grid = 101
        W = simulate_brownian_motions(n_repetitions=2 * n_repetitions,
                                       n_grid=n_grid)
        W1, W2 = W[:n_repetitions], W[n_repetitions:]

        num = np.max(np.abs(W1), axis=-1)
        denom = np.max(np.abs(
            W2 - np.arange(1, n_grid + 1)[None, :] / n_grid * W2[:, -1:]
        ), axis=-1)
    elif version == 'estimator':
        W = simulate_brownian_motions(n_repetitions=n_repetitions,
                                       n_grid=n_grid)
        num = np.max(np.abs(W), axis=-1)
        denom = 1
    else:
        raise ValueError(f"{version=} is not a valid version.")

    ratio = num / denom

    return ratio


if __name__ == '__main__':
    n = 500
    x = np.random.randn(n)

    results = best_approx_test(x)
    print(results)