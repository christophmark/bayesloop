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
                                   [0.00046598, 0.00046598, 0.00046598, 0.00046598, 0.00046598],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [2.99835794, 2.99835794, 2.99835794, 2.99835794, 2.99835794],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

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
                                   [0.00055206, 0.00051683, 0.00048233, 0.00045903, 0.00045629],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [2.97583695, 2.98402851, 2.99501907, 3.00527058, 3.01126784],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

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
                                   [0.00052993, 0.00050302, 0.00047929, 0.00045883, 0.00045629],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [2.87337277, 2.9265807, 2.98217424, 3.00439557, 3.01127175],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

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
                                   [0.00048679, 0.00045124, 0.00041767, 0.00039717, 0.00039976],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [2.81716591, 2.82337653, 2.83204058, 2.83944083, 2.84187612],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

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
                                   [0.0007529, 0.00070742, 0.00066273, 0.00063262, 0.00062968],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [2.77252114, 2.78251864, 2.79475018, 2.80541289, 2.81072838],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

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
                                   [0.0020607, 0.0019692, 0.00187339, 0.0018053, 0.00179584],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [2.25427356, 2.26949283, 2.28527551, 2.29704214, 2.30024139],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

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
                                   [0.00181567, 0.00213315, 0.00091028, 0.00041154, 0.00090885],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('rate'),
                                   [1.01204314, 2.25763551, 3.24176817, 3.74634864, 3.12632199],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

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
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', 0.1, target='mean'))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.00722368, 0.00712209, 0.00702789, 0.00696926, 0.00696322],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [2.99313985, 2.99573566, 3., 3.00426434, 3.00686015],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

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
                                   [0.02976422, 0.15404218, 0.10859567, 0.02553673, 0.00054109],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [1.08288559, 2.24388932, 2.38033179, 2.98934128, 4.64547841],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

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
                                   [0.04317995, 0.04296549, 0.04275526, 0.04262151, 0.04262491],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [2.66415455, 2.66519273, 2.66664847, 2.66788051, 2.66828383],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

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
                                   [0.01591204, 0.01579036, 0.01567361, 0.01559665, 0.01558591],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [2.99576496, 2.99741879, 3., 3.00258121, 3.00423504],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

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
                                   [0.00909976, 0.0089861, 0.00887967, 0.00881235, 0.00880499],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [2.9942575, 2.99646768, 3., 3.00353232, 3.0057425],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

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
                                   [4.525729e-04, 1.677903e-03, 2.945258e-07, 1.498415e-08, 1.102384e-09],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [0.95899404, 1.93816557, 2.99999968, 4.06183394, 5.04100612],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -8.010466752050611, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test optimized hyper-parameter values
        np.testing.assert_almost_equal(S.getHyperParameterValue('sigma'), 1.065854087589326, decimal=5,
                                       err_msg='Erroneous log-evidence value.')
        np.testing.assert_almost_equal(S.getHyperParameterValue('log10pMin'), -4.039735868499399, decimal=5,
                                       err_msg='Erroneous log-evidence value.')
