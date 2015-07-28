# -*- coding: utf-8 -*-
"""
preprocessing.py includes functions needed to reshape data.
"""
import numpy as np

def movingWindow(rawData, n):
    """
    Generates an array consisting of overlapping subsequences of raw data.

    :return: array of subsequences
    """
    data = np.array([rawData[i:i+n] for i in range(rawData.shape[0] - (n-1))])
    return data
