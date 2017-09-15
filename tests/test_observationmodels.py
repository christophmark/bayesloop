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


class TestNumPy:
    def test_numpy_1p(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([[1, 0.5], [2, 0.5], [3, 0.5], [4, 1.], [5, 1.]]))

        def likelihood(data, mu):
            x, std = data

            pdf = np.exp((x - mu) ** 2. / (2 * std ** 2.)) / np.sqrt(2 * np.pi * std ** 2.)
            return pdf

        L = bl.om.NumPy(likelihood, 'mu', bl.oint(0, 7, 100))

        S.setOM(L)
        S.setTM(bl.tm.Static())
        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, 148.92056578058387, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_scipy_2p(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        def likelihood(data, mu, std):
            x = data

            pdf = np.exp((x - mu) ** 2. / (2 * std ** 2.)) / np.sqrt(2 * np.pi * std ** 2.)
            return pdf

        L = bl.om.NumPy(likelihood, 'mu', bl.oint(0, 7, 100), 'std', bl.oint(1, 2, 100))

        S.setOM(L)
        S.setTM(bl.tm.Static())
        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, 29.792823521784587, decimal=5,
                                       err_msg='Erroneous log-evidence value.')


class TestBuiltin:
    def test_bernoulli(self):
        S = bl.Study()
        S.loadData(np.array([1, 0, 1, 0, 0]))

        L = bl.om.Bernoulli('p', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.logEvidence, -4.3494298741972859, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_poisson(self):
        S = bl.Study()
        S.loadData(np.array([1, 0, 1, 0, 0]))

        L = bl.om.Poisson('rate', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.logEvidence, -4.433708287229158, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_gaussian(self):
        S = bl.Study()
        S.loadData(np.array([1, 0, 1, 0, 0]))

        L = bl.om.Gaussian('mu', bl.oint(0, 1, 100), 'std', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.logEvidence, -12.430583625665736, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_gaussianmean(self):
        S = bl.Study()
        S.loadData(np.array([[1, 0.5], [0, 0.4], [1, 0.3], [0, 0.2], [0, 0.1]]))

        L = bl.om.GaussianMean('mu', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.logEvidence, -6.3333705075036226, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_whitenoise(self):
        S = bl.Study()
        S.loadData(np.array([1, 0, 1, 0, 0]))

        L = bl.om.WhiteNoise('std', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.logEvidence, -6.8161638661444073, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_ar1(self):
        S = bl.Study()
        S.loadData(np.array([1, 0, 1, 0, 0]))

        L = bl.om.AR1('rho', bl.oint(-1, 1, 100), 'sigma', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.logEvidence, -4.3291291450463421, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_scaledar1(self):
        S = bl.Study()
        S.loadData(np.array([1, 0, 1, 0, 0]))

        L = bl.om.ScaledAR1('rho', bl.oint(-1, 1, 100), 'sigma', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.logEvidence, -4.4178639067800738, decimal=5,
                                       err_msg='Erroneous log-evidence value.')
