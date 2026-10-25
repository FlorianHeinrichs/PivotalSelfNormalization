#
# tests/utils.py
#
# Project: Self-Normalization for CUSUM-based Change Detection in Locally
#          Stationary Time Series
# Date: 2025-08-15
# Author: Florian Heinrichs
#
# Utility functions used for different tests.

from typing import Callable

import numpy as np
from scipy import signal


def bandwidth_cv(X: np.ndarray,
                 min_bw: int,
                 max_bw: int,
                 estimator: Callable,
                 num_folds: int = 5,
                 step_size: int = 1,
                 batch_axis: int = -1,
                 return_mses: bool = False) -> np.ndarray | tuple:
    """
    Function to tune the bandwidth of kernel estimators.

    :param X: Functional time series given as numpy array.
    :param min_bw: Minimal bandwidth of the estimator.
    :param max_bw: Maximal bandwidth of the estimator.
    :param estimator: Kernel estimator whose bandwidth to tune.
    :param num_folds: Number of folds used for cross validation. Defaults to 5.
    :param step_size: Step size of bandwidth. Defaults to 1.
    :param batch_axis: The first axis of the NumPy array is the time axis along
        which the time series is smoothened, and the bandwidth is selected. The
        shape of the time series is (n_time,) + space_shape, and a single
        bandwidth, that is optimal across all points in space, is returned. If
        batch_axis is provided, the bandwidth is tuned for each entry along this
        axis separately.
    :param return_mses: Indicates whether MSEs are returned too.
    :return: Returns the optimal bandwidth as int.
    """
    indices_shuffle = np.arange(X.shape[0] // num_folds * num_folds)
    np.random.shuffle(indices_shuffle)
    folds = np.split(indices_shuffle, num_folds)
    indices = np.arange(X.shape[0])

    non_batch_axes = tuple(a for a in range(X.ndim) if a != batch_axis)
    n_samples = 1 if batch_axis == -1 else X.shape[batch_axis]

    best_bw, best_mse = - np.ones(n_samples, dtype=int), - np.ones(n_samples)
    mses = []

    for bw in range(min_bw, max_bw + 1, step_size):
        mse = np.zeros(n_samples)
        for fold in folds:
            filter_array = ~np.isin(indices, fold)
            estimate = estimator(X, bw, filter_array)

            mse += np.nanmean(
                (X[~filter_array] - estimate[~filter_array]) ** 2,
                axis=non_batch_axes
            )

        mses.append(mse)

        better_bw = np.where((mse < best_mse) | (best_mse == -1)
                             | np.isnan(best_mse))
        best_bw[better_bw], best_mse[better_bw] = bw, mse[better_bw]

    return (best_bw, mses) if return_mses else best_bw


def convolve(X: np.ndarray, kernel: np.ndarray, mode: str = 'valid') -> np.ndarray:
    """
    Convolve X with kernel across time axis.

    :param X: NumPy array of shape (n_time,) + space_shape.
    :param kernel: NumPy array of shape (bw,) with bw < n_time.
    :param mode: Mode of convolution, defaults to 'valid'.
    :return: Convolution of X with kernel across time axis.
    """
    convolution = np.moveaxis(
        signal.convolve(np.moveaxis(X, 0, -1), kernel, mode=mode), -1, 0
    )
    return convolution