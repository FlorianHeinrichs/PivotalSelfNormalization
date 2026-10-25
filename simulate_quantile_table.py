#
# tests/simulate_quantile_table.py
#
# Project: Self-Normalization for Change Detection in Non-Stationary Time Series
# Date: 2026-10-09
# Author: Florian Heinrichs

from datetime import datetime
import multiprocessing
import os
from typing import Callable

import numpy as np

from tests.best_approx_test import (optimize_all_segments,
                                    calculate_statistics,
                                    simulate_brownian_motions)


def _simulate_ratio(weights_dict: dict,
                    k_list: list | np.ndarray,
                    p_max: int = 2,
                    n_grid: int = 1000) -> dict:
    """
    Simulate limiting distribution, which is the ratio of two suprema.

    :param weights_dict: Weights for calculation of statistic with identifiers
        as keys and weights as values.
    :param p_max: Maximal number of null hypotheses to test.
    :param k_list: List of maximal k values.
    :param n_grid: Number of grid points used for the approximation of the
        Brownian motion.
    :return: Simulated limiting distribution.
    """
    k_max = np.max(k_list)
    W = simulate_brownian_motions(n_repetitions=1, n_grid=n_grid)

    num = np.max(np.abs(W), axis=-1)

    segments = optimize_all_segments(W[0], version='monotone')
    denom = calculate_statistics(segments, n_grid, k_max + 2 * p_max)

    results = {}

    for weight_idx, weights in weights_dict.items():
        ratio = [
            [np.max(num / denom[2 * p: 2 * p + k] * weights[:k])
             for p in range(p_max + 1)] for k in k_list
        ]
        argmax = [
            [np.argmax(num / denom[2 * p: 2 * p + k] * weights[:k])
             for p in range(p_max + 1)] for k in k_list
        ]

        result = np.concatenate([ratio, argmax], axis=-1)
        results[weight_idx] = result

    return results


def write_result_to_json(results: dict, folder: str, lock=None):
    with lock:
        for weight_idx, result in results.items():
            filepath = os.path.join(folder, f"{weight_idx}.npy")
            result = np.expand_dims(result, axis=0)
            if os.path.exists(filepath):
                old = np.load(filepath)
                new = np.concatenate([old, result], axis=0)
            else:
                new = result

            np.save(filepath, new)


def worker(sim_ratio: Callable, weights_dict: dict, k_list: list, folder: str,
           lock: multiprocessing.Lock):
    results = sim_ratio(weights_dict, k_list)
    write_result_to_json(results, folder, lock=lock)


def run_in_parallel(sim_ratio: Callable, weights_dict: dict, k_list: list,
                    folder: str, n_repetitions: int = 10_000, n_cpu: int = 4):
    n_processes = min(multiprocessing.cpu_count() - 1, n_cpu)
    processes = []

    with multiprocessing.Manager() as manager:
        lock = manager.Lock()

        for idx in range(n_repetitions):
            print(f"Simulate ratio {idx}")
            p = multiprocessing.Process(
                target=worker, args=(sim_ratio, weights_dict, k_list, folder, lock)
            )
            processes.append(p)
            p.start()

            if len(processes) >= n_processes:
                for proc in processes:
                    proc.join()
                processes = []

        for p in processes:
            p.join()


def calculate_quantile(folder: str, simple: bool = True) -> dict:
    alpha = np.array([0.001, 0.01, 0.05, 0.1])
    results = {}

    for weight_idx in ['log', 'sqrt']:
        filepath = folder + f"/{weight_idx}.npy"
        result = np.load(filepath)

        ratios = result[..., :3]

        quantiles = np.quantile(ratios, 1-alpha, axis=0,
                                method='averaged_inverted_cdf')
        if simple:
            results[weight_idx] = quantiles
        else:
            arg_max = result[..., 3:]
            k_mean = np.mean(arg_max, axis=0)
            k_std = np.std(arg_max, axis=0)
            results[weight_idx] = quantiles, k_mean, k_std

    return results


if __name__ == '__main__':
    n_grid = 1000
    k_list = [10, 20, 30, 40, 50]
    k_max = np.max(k_list)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = "estimated_quantiles"
    os.makedirs(folder, exist_ok=True)

    epsilon = 0.001
    k = np.arange(1, k_max + 1)
    weights = np.log(1.5 + k) * k ** (-1 / 2 - epsilon)
    weights0 = weights / weights[0]
    weights1 = k ** (-1 / 2 - epsilon)
    weights_dict = {'log': weights0, 'sqrt': weights1}

    run_in_parallel(_simulate_ratio, weights_dict, k_list, folder)

    folder = f"estimated_quantiles"
    results = calculate_quantile(folder)
    for k, v in results.items():
        print(k)
        print(np.round(np.concatenate([v[:, ::1, j] for j in range(3)], axis=1), 2))