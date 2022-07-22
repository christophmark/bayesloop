#!/usr/bin/env python

from __future__ import print_function, division
import bayesloop as bl
import numpy as np
import sympy.stats as stats


class TestTwoParameterModel:
    def test_fit_1cp_1bp_2hp(self):
        # carry out fit
        S = bl.ChangepointStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))

        T = bl.tm.SerialTransitionModel(
            bl.tm.Static(),
            bl.tm.ChangePoint('ChangePoint', [0, 1]),
            bl.tm.CombinedTransitionModel(
                bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 2), target='mean'),
                bl.tm.RegimeSwitch('log10pMin', [-3, -1])
            ),
            bl.tm.BreakPoint('BreakPoint', 'all'),
            bl.tm.Static()
        )

        S.setTM(T)
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.012437, 0.030168, 0.01761 , 0.001731, 0.001731],
                                   rtol=1e-03, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [0.968022, 1.956517, 3.476958, 4.161028, 4.161028],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -15.072007461556161, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test hyper-parameter distribution
        x, p = S.getHyperParameterDistribution('sigma')
        np.testing.assert_allclose(np.array([x, p]),
                                   [[0., 0.2], [0.4963324, 0.5036676]],
                                   rtol=1e-05, err_msg='Erroneous values in hyper-parameter distribution.')

        # test duration distribution
        d, p = S.getDurationDistribution(['ChangePoint', 'BreakPoint'])
        np.testing.assert_allclose(np.array([d, p]),
                                   [[1., 2., 3.], [0.01039273, 0.49395867, 0.49564861]],
                                   rtol=1e-05, err_msg='Erroneous values in duration distribution.')

    def test_fit_hyperpriors(self):
        # carry out fit
        S = bl.ChangepointStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))

        T = bl.tm.SerialTransitionModel(
            bl.tm.Static(),
            bl.tm.ChangePoint('ChangePoint', [0, 1], prior=np.array([0.3, 0.7])),
            bl.tm.CombinedTransitionModel(
                bl.tm.GaussianRandomWalk('sigma', bl.oint(0, 0.2, 2), target='mean', prior=lambda s: 1./s),
                bl.tm.RegimeSwitch('log10pMin', [-3, -1])
            ),
            bl.tm.BreakPoint('BreakPoint', 'all', prior=stats.Normal('Normal', 3., 1.)),
            bl.tm.Static()
        )

        S.setTM(T)
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.03372851, 0.05087598, 0.02024129, 0.00020918, 0.00020918],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [0.9894398, 1.92805399, 3.33966456, 4.28759449, 4.28759449],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -15.709534690217343, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test hyper-parameter distribution
        x, p = S.getHyperParameterDistribution('sigma')
        np.testing.assert_allclose(np.array([x, p]),
                                   [[0.06666667, 0.13333333], [0.66515107, 0.33484893]],
                                   rtol=1e-05, err_msg='Erroneous values in hyper-parameter distribution.')

        # test duration distribution
        d, p = S.getDurationDistribution(['ChangePoint', 'BreakPoint'])
        np.testing.assert_allclose(np.array([d, p]),
                                   [[1., 2., 3.], [0.00373717, 0.40402616, 0.59223667]],
                                   rtol=1e-05, err_msg='Erroneous values in duration distribution.')
