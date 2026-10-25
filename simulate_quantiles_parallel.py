#
# tests/simulate_quantiles_parallel.py
#
# Project: Self-Normalization for Change Detection in Non-Stationary Time Series
# Date: 2026-06-09
# Author: Florian Heinrichs

from datetime import datetime
import json
import multiprocessing
import os
from typing import Callable

import numpy as np

from experiments import convert_to_serializable
from tests.best_approx_test import _simulate_ratio


def simulate_ratio(weights: np.ndarray = None, p_max: int = 3, k_max: int = 10,
                   n_grid: int = 1_000, n_repetitions: int = 1) -> np.ndarray:
    if weights is None:
        epsilon = 0.001
        k = np.arange(1, k_max + 1)
        weights = np.log(1.5 + k) * k ** (-1 / 2 - epsilon)
        weights = weights / weights[0]

    ratio = _simulate_ratio(weights, p_max=p_max, k_max=k_max,
                            n_repetitions=n_repetitions, n_grid=n_grid)

    return ratio


def write_result_to_json(idx: int, ratios: np.ndarray, filepath: str, lock=None):
    with lock:
        if os.path.exists(filepath):
            with open(filepath, "r") as file:
                all_results = json.load(file)
        else:
            all_results = {}

        all_results[idx] = ratios
        all_results = convert_to_serializable(all_results)

        with open(filepath, mode='w') as file:
            json.dump(all_results, file, indent=4)


def worker(idx: int, sim_ratio: Callable, filepath: str, n_grid: int,
           lock: multiprocessing.Lock):
    ratios = sim_ratio(n_grid=n_grid)
    write_result_to_json(idx, ratios, filepath, lock=lock)


def run_in_parallel(sim_ratio: Callable, filepath: str, n_grid: int = 1_000,
                    n_repetitions: int = 10_000):
    n_processes = min(multiprocessing.cpu_count() - 1, 2)
    processes = []

    with multiprocessing.Manager() as manager:
        lock = manager.Lock()

        for idx in range(n_repetitions):
            print(f"Simulate ratio {idx}")
            p = multiprocessing.Process(
                target=worker, args=(idx, sim_ratio, filepath, n_grid, lock)
            )
            processes.append(p)
            p.start()

            if len(processes) >= n_processes:
                for proc in processes:
                    proc.join()
                processes = []

        for p in processes:
            p.join()


def calculate_quantile(filepath: str, alpha: float = 0.05) -> np.ndarray:
    with open(filepath) as f:
        results = json.load(f)

    ratios = np.array(list(results.values()))[:, 0, :]
    n_repetitions = len(ratios)
    quantile = np.sort(ratios, axis=0)[int(n_repetitions * (1 - alpha)) - 1]

    return quantile



if __name__ == "__main__":
    filepath = "estimated_quantiles.json"

    # Simulate and store ratios
    run_in_parallel(simulate_ratio, filepath)

    # Load ratios and calculate quantile
    print(calculate_quantile(filepath))