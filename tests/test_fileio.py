#!/usr/bin/env python

from __future__ import print_function, division
import bayesloop as bl
import numpy as np


class TestFileIO:
    def test_save_load(self, tmp_path):
        S = bl.HyperStudy()
        S.load_data(np.array([1, 2, 3, 4, 5]))
        S.set_observation_model(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))
        S.set_transition_model(bl.tm.Static())
        S.fit()

        path = tmp_path / 'study.bl'
        bl.save(path, S)
        S = bl.load(path)
