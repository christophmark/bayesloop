#!/usr/bin/env python

from __future__ import print_function, division
import bayesloop as bl
import numpy as np
import sympy.stats as stats


class TestOneParameterModel:
    def test_fit_0hp(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Poisson('rate'))
        S.setTM(bl.tm.Static())
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('rate', density=False)[1][:, 250],
                                   [0.00034, 0.00034, 0.00034, 0.00034, 0.00034],
                                   rtol=1e-3, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [3.09761, 3.09761, 3.09761, 3.09761, 3.09761],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.4463425036, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_fit_1hp(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Poisson('rate'))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', 0.1, target='rate'))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('rate', density=False)[1][:, 250],
                                   [0.000417, 0.000386, 0.000356, 0.000336, 0.000332],
                                   rtol=1e-03, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [3.073534, 3.08179 , 3.093091, 3.104016, 3.111173],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.4337420351, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_fit_2hp(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Poisson('rate'))

        T = bl.tm.CombinedTransitionModel(bl.tm.GaussianRandomWalk('sigma', 0.1, target='rate'),
                                          bl.tm.RegimeSwitch('log10pMin', -3))

        S.setTM(T)
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('rate', density=False)[1][:, 250],
                                   [0.000412, 0.000376, 0.000353, 0.000336, 0.000332],
                                   rtol=1e-02, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [2.942708, 3.002756, 3.071995, 3.103038, 3.111179],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.4342948181, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_fit_prior_array(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Poisson('rate', bl.oint(0, 6, 1000), prior=np.ones(1000)))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', 0.1, target='rate'))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('rate', density=False)[1][:, 250],
                                   [0.000221, 0.000202, 0.000184, 0.000172, 0.000172],
                                   rtol=1e-02, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [3.174159, 3.180812, 3.190743, 3.200642, 3.20722 ],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.0866227472, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_fit_prior_function(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Poisson('rate', bl.oint(0, 6, 1000), prior=lambda x: 1./x))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', 0.1, target='rate'))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('rate', density=False)[1][:, 250],
                                   [0.000437, 0.000401, 0.000366, 0.000342, 0.000337],
                                   rtol=1e-02, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [2.967834, 2.977838, 2.990624, 3.002654, 3.010419],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -11.3966589329, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_fit_prior_sympy(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Poisson('rate', bl.oint(0, 6, 1000), prior=stats.Exponential('expon', 1.)))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', 0.1, target='rate'))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('rate', density=False)[1][:, 250],
                                   [0.000881, 0.00081 , 0.00074 , 0.00069 , 0.000674],
                                   rtol=1e-03, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [2.627709, 2.643611, 2.661415, 2.677185, 2.687023],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -11.1819034242, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_optimize(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Poisson('rate', bl.oint(0, 6, 1000), prior=stats.Exponential('expon', 1.)))

        T = bl.tm.CombinedTransitionModel(bl.tm.GaussianRandomWalk('sigma', 2.1, target='rate'),
                                          bl.tm.RegimeSwitch('log10pMin', -3))

        S.setTM(T)
        S.optimize()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('rate', density=False)[1][:, 250],
                                   [1.820641e-03, 2.083830e-03, 7.730833e-04, 1.977125e-04, 9.441302e-05],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [1.015955, 2.291846, 3.36402 , 4.113622, 4.390356],
                                   rtol=1e-03, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -9.47362827569, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test optimized hyper-parameter values
        np.testing.assert_almost_equal(S.getHyperParameterValue('sigma'), 2.11216289063, decimal=5,
                                       err_msg='Erroneous log-evidence value.')
        np.testing.assert_almost_equal(S.getHyperParameterValue('log10pMin'), -3.0, decimal=3,
                                       err_msg='Erroneous log-evidence value.')


class TestTwoParameterModel:
    def test_fit_0hp(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))
        S.setTM(bl.tm.Static())
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.013349, 0.013349, 0.013349, 0.013349, 0.013349],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [3., 3., 3., 3., 3.],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -16.1946904707, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_fit_1hp(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', 0.1, target='mean'))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.013547, 0.013428, 0.013315, 0.013241, 0.013232],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [2.995242, 2.997088, 3.      , 3.002912, 3.004758],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -16.1865343702, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_fit_2hp(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))

        T = bl.tm.CombinedTransitionModel(bl.tm.GaussianRandomWalk('sigma', 0.1, target='mean'),
                                          bl.tm.RegimeSwitch('log10pMin', -3))

        S.setTM(T)
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.018848, 0.149165, 0.025588, 0.006414, 0.005426],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [1.005987, 2.710129, 3.306985, 3.497192, 3.527645],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -14.3305753098, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_fit_prior_array(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=np.ones((20, 20))))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', 0.1, target='mean'))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.02045 , 0.020327, 0.020208, 0.020128, 0.020115],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [2.99656 , 2.997916, 3.      , 3.002084, 3.00344 ],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.9827282104, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_fit_prior_function(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1./s))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', 0.1, target='mean'))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.018242, 0.018119, 0.018001, 0.017921, 0.01791 ],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [2.996202, 2.997693, 3.      , 3.002307, 3.003798],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -11.9842221343, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_fit_prior_sympy(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20),
                               prior=[stats.Uniform('u', 0, 6), stats.Exponential('e', 2.)]))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', 0.1, target='mean'))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.014305, 0.014183, 0.014066, 0.01399 , 0.01398 ],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [2.995526, 2.997271, 3.      , 3.002729, 3.004474],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -12.4324853153, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_optimize(self):
        # carry out fit
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))

        T = bl.tm.CombinedTransitionModel(bl.tm.GaussianRandomWalk('sigma', 1.07, target='mean'),
                                          bl.tm.RegimeSwitch('log10pMin', -3.90))

        S.setTM(T)
        S.optimize()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [9.903855e-03, 1.887901e-02, 8.257234e-05, 5.142727e-06, 2.950377e-06],
                                   rtol=1e-02, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [0.979099, 1.951689, 3.000075, 4.048376, 5.020886],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -8.010466752050611, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test optimized hyper-parameter values
        np.testing.assert_almost_equal(S.getHyperParameterValue('sigma'), 1.065854087589326, decimal=5,
                                       err_msg='Erroneous log-evidence value.')
        np.testing.assert_almost_equal(S.getHyperParameterValue('log10pMin'), -4.039735868499399, decimal=5,
                                       err_msg='Erroneous log-evidence value.')
