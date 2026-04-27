#!/usr/bin/env python
"""
This file includes functions used for the preprocessing of measurement data that is to be analyzed using bayesloop. The
time series data is formatted into data segments that can be fed to the inference algorithm piece-by-piece.

Example: The auto-regressive process of first order needs two subsequent data points at each time step. Therefore, the
    data is separated into overlapping data segments containing two data points.
"""

from __future__ import division, print_function
import numpy as np


def movingWindow(rawData, n):
    """
    Generates an array consisting of overlapping sub-sequences of raw data.

    Args:
        rawData(ndarray): Array containing time series data
        n(int): integer (> 0) stating the number of data points in each data segment that is passed to the algorithm

    Returns:
        ndarray: Array of data segments, each containing n overlapping data points
    """
    data = np.array([rawData[i:i+n] for i in range(rawData.shape[0] - (n-1))])
    return data
