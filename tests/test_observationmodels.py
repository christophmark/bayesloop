#!/usr/bin/env python

from __future__ import print_function, division
import bayesloop as bl
import numpy as np
import scipy.stats
import sympy.stats
from sympy import Symbol


class TestSymPy:
    def test_sympy_1p(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        rate = Symbol('rate', positive=True)
        poisson = sympy.stats.Poisson('poisson', rate)
        L = bl.om.SymPy(poisson, 'rate', bl.oint(0, 7, 100))

        S.setOM(L)
        S.setTM(bl.tm.Static())
        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.447907381964875, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_sympy_2p(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        mu = Symbol('mu')
        std = Symbol('std', positive=True)
        normal = sympy.stats.Normal('norm', mu, std)

        L = bl.om.SymPy(normal, 'mu', bl.cint(0, 7, 200), 'std', bl.oint(0, 1, 200), prior=lambda x, y: 1.)

        S.setOM(L)
        S.setTM(bl.tm.Static())
        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -13.663836264357226, decimal=5,
                                       err_msg='Erroneous log-evidence value.')


class TestSciPy:
    def test_scipy_1p(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.SciPy(scipy.stats.poisson, 'mu', bl.oint(0, 7, 100), fixedParameters={'loc': 0})

        S.setOM(L)
        S.setTM(bl.tm.Static())
        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.238278174965238, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_scipy_2p(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.SciPy(scipy.stats.norm, 'loc', bl.cint(0, 7, 200), 'scale', bl.oint(0, 1, 200))

        S.setOM(L)
        S.setTM(bl.tm.Static())
        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -13.663836264357225, decimal=5,
                                       err_msg='Erroneous log-evidence value.')
