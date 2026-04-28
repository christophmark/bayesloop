#!/usr/bin/env python

from __future__ import print_function, division
import bayesloop as bl
import numpy as np
from scipy.ndimage import gaussian_filter1d


class TestBuiltin:
    def test_gaussianrandomwalk_matches_scipy_filter(self):
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Gaussian('mu', bl.oint(0, 6, 20), 'sigma', bl.oint(0, 2, 20))
        T = bl.tm.GaussianRandomWalk('sigma', 0.2, target='mu')
        S.set(L, T)

        posterior = np.arange(400, dtype=float).reshape(20, 20)
        posterior /= np.sum(posterior)

        expected = gaussian_filter1d(posterior, 0.2 / S.lattice_constant[0], axis=0)
        np.testing.assert_allclose(T.compute_forward_prior(posterior, 0), expected)

    def test_gaussianrandomwalk_reuses_multiple_cached_kernels(self):
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.GaussianRandomWalk('sigma', 0.1, target='rate')
        S.set(L, T)

        posterior = np.ones(S.grid_size)
        posterior /= np.sum(posterior)

        T.hyper_parameter_values[0] = 0.1
        T.compute_forward_prior(posterior, 0)
        first_kernel_key = (float(0.1 / S.lattice_constant[0]), 0)
        first_kernel = T.kernel_cache[first_kernel_key]

        T.hyper_parameter_values[0] = 0.2
        T.compute_forward_prior(posterior, 0)
        T.hyper_parameter_values[0] = 0.1
        T.compute_forward_prior(posterior, 0)

        assert len(T.kernel_cache) == 2
        assert T.kernel_cache[first_kernel_key] is first_kernel

    def test_static(self):
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -10.372209708143769, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_deterministic(self):
        S = bl.HyperStudy()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        def linear(t, a=[1, 2]):
            return 0.5 + 0.2*a*t

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.Deterministic(linear, target='rate')
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -9.4050089375418136, decimal=3,
                                       err_msg='Erroneous log-evidence value.')

    def test_gaussianrandomwalk(self):
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.GaussianRandomWalk('sigma', 0.2, target='rate')
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -10.323144246611964, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_bivariaterandomwalk(self):
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Gaussian('mu', bl.oint(0, 6, 20), 'sigma', bl.oint(0, 2, 20))
        T = bl.tm.BivariateRandomWalk('sigma1', 1., 'sigma2', 0.1, 'rho', 0.5)
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -7.330706514472251, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_alphastablerandomwalk(self):
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.AlphaStableRandomWalk('c', 0.2, 'alpha', 1.5, target='rate')
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -10.122384638661309, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_changepoint(self):
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.ChangePoint('t_change', 2)
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -12.894336092378385, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_regimeswitch(self):
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.RegimeSwitch('p_min', -3)
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -10.372866559561402, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_independent(self):
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.Independent()
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -11.087360077190617, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_notequal(self):
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.NotEqual('p_min', -3)
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -10.569099863134156, decimal=5,
                                       err_msg='Erroneous log-evidence value.')


class TestNested:
    def test_nested(self):
        S = bl.Study()
        S.load_data(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.SerialTransitionModel(
            bl.tm.Static(),
            bl.tm.ChangePoint('t_change', 1),
            bl.tm.CombinedTransitionModel(
                bl.tm.GaussianRandomWalk('sigma', 0.2, target='rate'),
                bl.tm.RegimeSwitch('p_min', -3)
            ),
            bl.tm.BreakPoint('t_break', 3),
            bl.tm.Independent()
        )
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -13.269918024215237, decimal=5,
                                       err_msg='Erroneous log-evidence value.')
