# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='bayesloop',
    version='0.1',
    description='Framework for fitting hierarchical time series models based on Bayesian inference.',
    url='https://bitbucket.org/chrismark/bayesloop',
    author='Christoph Mark',
    author_email='christoph.mark@fau.de',
    license='The MIT License (MIT)',
    packages=['bayesloop'],
    install_requires=['numpy>=1.9.2', 'scipy>=0.15.1']
    )
