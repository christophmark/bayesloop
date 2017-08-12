#!/usr/bin/env python

from __future__ import print_function, division
import bayesloop as bl
import numpy as np
import sympy.stats as stats


class TestTwoParameterModel:
    def test_fit_0hp(self):
        # carry out fit (this test is designed to fall back on the fit method of the Study class)
        S = bl.HyperStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20)))
        S.setTM(bl.tm.Static())
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.00707902, 0.00707902, 0.00707902, 0.00707902, 0.00707902],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [3., 3., 3., 3., 3.],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -16.1946904707, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_fit_1hp(self):
        # carry out fit
        S = bl.HyperStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20)))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 2), target='mean'))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.01042107, 0.00766233, 0.00618352, 0.00554651, 0.00548637],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [2.88534505, 2.93135361, 3., 3.06864639, 3.11465495],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -16.0629517262, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test hyper-parameter distribution
        x, p = S.getHyperParameterDistribution('sigma')
        print(np.array([x, p]))
        np.testing.assert_allclose(np.array([x, p]),
                                   [[0., 0.2], [0.43828499, 0.56171501]],
                                   rtol=1e-05, err_msg='Erroneous values in hyper-parameter distribution.')

    def test_fit_2hp(self):
        # carry out fit
        S = bl.HyperStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20)))

        T = bl.tm.CombinedTransitionModel(bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 2), target='mean'),
                                          bl.tm.RegimeSwitch('log10pMin', [-3, -1]))

        S.setTM(T)
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [5.80970506e-03, 1.12927905e-01, 4.44501254e-02, 1.00250119e-02, 1.72751309e-05],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [0.96492471, 2.09944204, 2.82451616, 3.72702495, 5.0219119],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.7601875492, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test hyper-parameter distribution
        x, p = S.getHyperParameterDistribution('sigma')
        np.testing.assert_allclose(np.array([x, p]),
                                   [[0., 0.2], [0.48943645, 0.51056355]],
                                   rtol=1e-05, err_msg='Erroneous values in hyper-parameter distribution.')

        # test joint hyper-parameter distribution
        x, y, p = S.getJointHyperParameterDistribution(['log10pMin', 'sigma'])
        np.testing.assert_allclose(np.array([x, y]),
                                   [[-3., -1.], [0., 0.2]],
                                   rtol=1e-05, err_msg='Erroneous parameter values in joint hyper-parameter '
                                                       'distribution.')

        np.testing.assert_allclose(p,
                                   [[0.00701834, 0.0075608], [0.48241812, 0.50300274]],
                                   rtol=1e-05, err_msg='Erroneous probability values in joint hyper-parameter '
                                                       'distribution.')

    def test_fit_hyperprior_array(self):
        # carry out fit
        S = bl.HyperStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20)))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 2), target='mean', prior=np.array([0.2, 0.8])))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.01205759, 0.00794796, 0.00574501, 0.00479608, 0.00470649],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [2.82920111, 2.89773902, 3., 3.10226098, 3.17079889],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -15.9915077133, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test hyper-parameter distribution
        x, p = S.getHyperParameterDistribution('sigma')
        np.testing.assert_allclose(np.array([x, p]),
                                   [[0., 0.2], [0.16322581, 0.83677419]],
                                   rtol=1e-05, err_msg='Erroneous values in hyper-parameter distribution.')

    def test_fit_hyperprior_function(self):
        # carry out fit
        S = bl.HyperStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20)))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', bl.cint(0.1, 0.3, 2), target='mean', prior=lambda s: 1./s))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.04071021, 0.00783661, 0.00527211, 0.00484169, 0.00480379],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [2.68460027, 2.81872578, 3., 3.18127422, 3.31539973],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -15.9898700147, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test hyper-parameter distribution
        x, p = S.getHyperParameterDistribution('sigma')
        np.testing.assert_allclose(np.array([x, p]),
                                   [[0.1, 0.3], [0.61609973, 0.38390027]],
                                   rtol=1e-05, err_msg='Erroneous values in hyper-parameter distribution.')

    def test_fit_hyperprior_sympy(self):
        # carry out fit
        S = bl.HyperStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20)))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 2), target='mean', prior=stats.Exponential('e', 1.)))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.01012545, 0.00761074, 0.00626273, 0.00568207, 0.00562725],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [2.89548676, 2.93742566, 3., 3.06257434, 3.10451324],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -17.0866290887, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test hyper-parameter distribution
        x, p = S.getHyperParameterDistribution('sigma')
        np.testing.assert_allclose(np.array([x, p]),
                                   [[0., 0.2], [0.487971, 0.512029]],
                                   rtol=1e-05, err_msg='Erroneous values in hyper-parameter distribution.')
