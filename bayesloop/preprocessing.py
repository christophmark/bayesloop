#!/usr/bin/env python
"""
This file includes functions used for the preprocessing of measurement data that is to be analyzed using bayesloop. The
time series data is formatted into data segments that can be fed to the inference algorithm piece-by-piece.

Example: The auto-regressive process of first order needs two subsequent data points at each time step. Therefore, the
data is separated into overlapping data segments containing two data points.
"""

import numpy as np


def movingWindow(rawData, n):
    """
    Generates an array consisting of overlapping subsequences of raw data.

    Args:
        rawData: Numpy array containing time series data
        n: integer (> 0) stating the number of data points in each data segment that is passed to the algorithm

    Returns:
        Numpy array of data segments, each containing n overlapping data points
    """
    data = np.array([rawData[i:i+n] for i in range(rawData.shape[0] - (n-1))])
    return data
