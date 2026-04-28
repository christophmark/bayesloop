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
        S.load_data(np.array([1, 2, 3, 4, 5]))

        rate = Symbol('rate', positive=True)
        poisson = sympy.stats.Poisson('poisson', rate)
        L = bl.om.SymPy(poisson, 'rate', bl.oint(0, 7, 100))

        S.set_observation_model(L)
        S.set_transition_model(bl.tm.Static())
        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -10.238278174965238, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_sympy_2p(self):
        # carry out fit
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        mu = Symbol('mu')
        std = Symbol('std', positive=True)
        normal = sympy.stats.Normal('norm', mu, std)

        L = bl.om.SymPy(normal, 'mu', bl.cint(0, 7, 200), 'std', bl.oint(0, 1, 200), prior=lambda x, y: 1.)

        S.set_observation_model(L)
        S.set_transition_model(bl.tm.Static())
        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -13.663836264357226, decimal=5,
                                       err_msg='Erroneous log-evidence value.')


class TestSciPy:
    def test_scipy_1p(self):
        # carry out fit
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.SciPy(scipy.stats.poisson, 'mu', bl.oint(0, 7, 100), fixed_parameters={'loc': 0})

        S.set_observation_model(L)
        S.set_transition_model(bl.tm.Static())
        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -10.238278174965238, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_scipy_2p(self):
        # carry out fit
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.SciPy(scipy.stats.norm, 'loc', bl.cint(0, 7, 200), 'scale', bl.oint(0, 1, 200))

        S.set_observation_model(L)
        S.set_transition_model(bl.tm.Static())
        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -13.663836264357225, decimal=5,
                                       err_msg='Erroneous log-evidence value.')


class TestNumPy:
    def test_numpy_1p(self):
        # carry out fit
        S = bl.Study()
        S.load_data(np.array([[1, 0.5], [2, 0.5], [3, 0.5], [4, 1.], [5, 1.]]))

        def likelihood(data, mu):
            x, std = data

            pdf = np.exp((x - mu) ** 2. / (2 * std ** 2.)) / np.sqrt(2 * np.pi * std ** 2.)
            return pdf

        L = bl.om.NumPy(likelihood, 'mu', bl.oint(0, 7, 100))

        S.set_observation_model(L)
        S.set_transition_model(bl.tm.Static())
        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, 148.92056578058387, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_scipy_2p(self):
        # carry out fit
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        def likelihood(data, mu, std):
            x = data

            pdf = np.exp((x - mu) ** 2. / (2 * std ** 2.)) / np.sqrt(2 * np.pi * std ** 2.)
            return pdf

        L = bl.om.NumPy(likelihood, 'mu', bl.oint(0, 7, 100), 'std', bl.oint(1, 2, 100))

        S.set_observation_model(L)
        S.set_transition_model(bl.tm.Static())
        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, 29.792823521784587, decimal=5,
                                       err_msg='Erroneous log-evidence value.')


class TestBuiltin:
    def test_prepared_grid_likelihoods_match_formulas(self):
        x = np.array([1.25])

        p_grid = [bl.oint(0, 1, 17)]
        bernoulli = bl.om.Bernoulli('p', p_grid[0])
        np.testing.assert_allclose(bernoulli.pdf(p_grid, np.array([1.])), p_grid[0])
        np.testing.assert_allclose(bernoulli.pdf(p_grid, np.array([0.])), 1. - p_grid[0])

        rate_grid = [bl.oint(0, 8, 31)]
        poisson = bl.om.Poisson('rate', rate_grid[0])
        expected = (rate_grid[0] ** 3.) * np.exp(-rate_grid[0]) / 6.
        np.testing.assert_allclose(poisson.pdf(rate_grid, np.array([3.])), expected)

        mean, std = np.meshgrid(bl.cint(-2, 2, 13), bl.oint(0.2, 3, 11), indexing='ij')
        gaussian_grid = [mean, std]
        gaussian = bl.om.Gaussian('mean', None, 'std', None)
        expected = np.exp(-((x[0] - mean) ** 2.) / (2. * std ** 2.) - .5 * np.log(2. * np.pi * std ** 2.))
        np.testing.assert_allclose(gaussian.pdf(gaussian_grid, x), expected)

        laplace = bl.om.Laplace('mean', None, 'scale', None)
        expected = np.exp(-np.abs(x[0] - mean) / std) / (2. * std)
        np.testing.assert_allclose(laplace.pdf(gaussian_grid, x), expected)

        noise_grid = [bl.oint(0.2, 3, 17)]
        white_noise = bl.om.WhiteNoise('std', None)
        expected = np.exp(-(x[0] ** 2.) / (2. * noise_grid[0] ** 2.) -
                          .5 * np.log(2. * np.pi * noise_grid[0] ** 2.))
        np.testing.assert_allclose(white_noise.pdf(noise_grid, x), expected)

        rho, sigma = np.meshgrid(bl.oint(-.9, .9, 11), bl.oint(.2, 2, 13), indexing='ij')
        ar_grid = [rho, sigma]
        data_segment = np.array([1.1, .2])
        ar1 = bl.om.AR1('rho', None, 'sigma', None)
        expected = np.exp(-((data_segment[1] - rho * data_segment[0]) ** 2.) / (2. * sigma ** 2.) -
                          .5 * np.log(2. * np.pi * sigma ** 2.))
        np.testing.assert_allclose(ar1.pdf(ar_grid, data_segment), expected)

        scaled_ar1 = bl.om.ScaledAR1('rho', None, 'sigma', None)
        scaled_sigma = sigma * np.sqrt(1. - rho ** 2.)
        expected = np.exp(-((data_segment[1] - rho * data_segment[0]) ** 2.) / (2. * scaled_sigma ** 2.) -
                          .5 * np.log(2. * np.pi * scaled_sigma ** 2.))
        np.testing.assert_allclose(scaled_ar1.pdf(ar_grid, data_segment), expected)

    def test_prepared_grid_cache_updates_for_new_grid(self):
        L = bl.om.Poisson('rate', None)
        first_grid = [bl.oint(0, 4, 12)]
        second_grid = [bl.oint(0, 8, 12)]

        first = L.pdf(first_grid, np.array([2.]))
        second = L.pdf(second_grid, np.array([2.]))

        np.testing.assert_allclose(first, (first_grid[0] ** 2.) * np.exp(-first_grid[0]) / 2.)
        np.testing.assert_allclose(second, (second_grid[0] ** 2.) * np.exp(-second_grid[0]) / 2.)

    def test_bernoulli(self):
        S = bl.Study()
        S.load_data(np.array([1, 0, 1, 0, 0]))

        L = bl.om.Bernoulli('p', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.log_evidence, -4.3494298741972859, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_poisson(self):
        S = bl.Study()
        S.load_data(np.array([1, 0, 1, 0, 0]))

        L = bl.om.Poisson('rate', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.log_evidence, -4.433708287229158, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_gaussian(self):
        S = bl.Study()
        S.load_data(np.array([1, 0, 1, 0, 0]))

        L = bl.om.Gaussian('mu', bl.oint(0, 1, 100), 'std', bl.oint(0, 1, 100), prior=lambda m, s: 1/s**3)
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.log_evidence, -12.430583625665736, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_laplace(self):
        S = bl.Study()
        S.load_data(np.array([1, 0, 1, 0, 0]))

        L = bl.om.Laplace('mu', None, 'b', None)
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.log_evidence, -10.658573159, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_gaussianmean(self):
        S = bl.Study()
        S.load_data(np.array([[1, 0.5], [0, 0.4], [1, 0.3], [0, 0.2], [0, 0.1]]))

        L = bl.om.GaussianMean('mu', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.log_evidence, -6.3333705075036226, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_whitenoise(self):
        S = bl.Study()
        S.load_data(np.array([1, 0, 1, 0, 0]))

        L = bl.om.WhiteNoise('std', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.log_evidence, -6.8161638661444073, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_ar1(self):
        S = bl.Study()
        S.load_data(np.array([1, 0, 1, 0, 0]))

        L = bl.om.AR1('rho', bl.oint(-1, 1, 100), 'sigma', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.log_evidence, -4.3291291450463421, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_scaledar1(self):
        S = bl.Study()
        S.load_data(np.array([1, 0, 1, 0, 0]))

        L = bl.om.ScaledAR1('rho', bl.oint(-1, 1, 100), 'sigma', bl.oint(0, 1, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()
        np.testing.assert_almost_equal(S.log_evidence, -4.4178639067800738, decimal=5,
                                       err_msg='Erroneous log-evidence value.')
