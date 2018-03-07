#!/usr/bin/env python

from __future__ import print_function, division
import bayesloop as bl
import numpy as np


class TestFileIO:
    def test_save_load(self):
        S = bl.HyperStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))
        S.setTM(bl.tm.Static())
        S.fit()

        bl.save('study.bl', S)
        S = bl.load('study.bl')
