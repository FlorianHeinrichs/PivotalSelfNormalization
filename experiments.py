#
# experiments.py
#
# Project: Self-Normalization for Change Detection in Non-Stationary Time Series
# Date: 2026-06-16
# Author: Florian Heinrichs
#
# Script containing main experiments for test comparison.

from datetime import datetime
import json

import numpy as np

from data_generation import generate_data
from tests.bootstrap import bootstrap_test
from tests.classic_self_normalization import self_normalization_test
from tests.classic_self_normalization import simulate_quantile as simulate_quantile_sn
from tests.lle_based_test import kernel_based_sn
from tests.best_approx_test import (best_approx_test,
                                    approximate_quantiles)
from tests.best_approx_alternatives import best_approx_test as baseline_test
from tests.best_approx_alternatives import approximate_quantiles as baseline_quantiles
from simulate_quantiles_parallel import calculate_quantile


def experiment(n: int, n_samples: int, mean: str, error: str, sigma: str,
               alpha: float = 0.05, weights: np.ndarray = None, p_max: int = 3,
               k_max: int = 10, quantile: np.ndarray = None,
               quantile_sn: float = None) -> dict:
    """
    Main function for running the experiments.

    :param n: Length of generated time series.
    :param n_samples: Number of generated time series.
    :param mean: Type of mean in ['1', '2', 'abrupt', 'const'].
    :param error: Type of error in ['iid', 'ar', 'ma'].
    :param sigma: Type of sigma in ['0', '1', '2', '3'].
    :param alpha: Level of test.
    :param weights: Weights used for the calculation of test statistics.
    :param p_max: Number of null hypotheses to test.
    :param k_max: Number of alternatives to consider for each null.
    :return: Dictionaries with test decisions for all tests and their
        respective run times.
    """
    if weights is None:
        epsilon = 0.001
        k = np.arange(1, k_max + 1)
        weights = np.log(1.5 + k) * k ** (-1 / 2 - epsilon)
        weights = weights / weights[0]

    # Generate Data
    data = generate_data(n, n_samples, mean, error, sigma)

    # Quantile Estimation
    if quantile is None:
        quantile = approximate_quantiles(
            weights, p_max=p_max, k_max=k_max, alpha=alpha, # n_grid=n
        )

    quantiles = {'proposed': quantile}
    quantiles.update({version: baseline_quantiles(alpha=alpha, version=version)
                      for version in ['bivariate', 'estimator']})

    # Do tests
    all_tests = ['proposed', 'bivariate', 'estimator', 'bootstrap', 'classic_sn']
    results = {version: {'test_decision': [], 'time': []}
               for version in all_tests}

    result_kernel = kernel_based_sn(data, alpha=alpha)
    results['kernel_based'] = {'test_decision': result_kernel[0],
                               'time': result_kernel[1]}

    for x in data.transpose():
        result_bootstrap = bootstrap_test(x)
        results['bootstrap']['test_decision'].append(result_bootstrap[0])
        results['bootstrap']['time'].append(result_bootstrap[1])

        if quantile_sn is not None:
            result_classic_sn = self_normalization_test(x, quantile=quantile_sn)
            results['classic_sn']['test_decision'].append(result_classic_sn[0])
            results['classic_sn']['time'].append(result_classic_sn[1])

        result_proposed = best_approx_test(x, weights, quantiles['proposed'],
                                           p_max=p_max, k_max=k_max)
        results['proposed']['test_decision'].append(result_proposed[0])
        results['proposed']['time'].append(result_proposed[1])

        result_baseline = baseline_test(
            x, versions=['bivariate', 'estimator'], quantile=quantiles
        )

        for version, result in result_baseline.items():
            for metric, res in result.items():
                results[version][metric].append(res)

    return results


def convert_to_serializable(data):
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, (list, np.ndarray)):
        return np.array(data).tolist()
    elif isinstance(data, tuple):
        return [convert_to_serializable(value) for value in data]
    else:
        return data


def get_config(n_samples) -> list:
    """
    Get list of configurations for experiments.

    :param n_samples: Number of generated time series.
    :return: List of configurations, where each entry of the list is a tuple
        of arguments of experiment().
    """
    config = [
        (n, n_samples, mean, error, sigma)
        for n in [100, 200, 500] for mean in ['0', '1', '2']
        for error in ['iid', 'loc_stat'] for sigma in ['0', '1', '2']
    ]

    return config


if __name__ == '__main__':
    alpha = 0.05
    n_samples = 1000
    config = get_config(n_samples)
    results = {}

    filepath = f"path_to_results.json"  # log file
    fp_quantile = f"path_to_quantiles.json"  # output of simulate_quantiles
    quantile = calculate_quantile(fp_quantile)
    quantile_sn = simulate_quantile_sn(alpha, n_repetitions=100000, n_grid=100)

    for args in config:
        arg_str = f"n{args[0]}_mu{args[2]}_{args[3]}_sigma{args[4]}"
        print(f"Starting experiment: {arg_str}")
        results[arg_str] = experiment(*args, alpha=alpha, quantile=quantile,
                                      quantile_sn=quantile_sn)

        with open(filepath, 'w') as file:
            result_tmp = convert_to_serializable(results)
            json.dump(result_tmp, file, indent=4)
