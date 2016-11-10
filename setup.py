#!/usr/bin/env python

from setuptools import setup

setup(
    name='bayesloop',
    version='1.0',
    description='Probabilistic programming framework that enables objective model selection for time-varying parameter models.',
    url='https://github.com/christophmark/bayesloop',
    author='Christoph Mark',
    author_email='christoph.mark@fau.de',
    license='The MIT License (MIT)',
    packages=['bayesloop'],
    install_requires=['numpy>=1.11.0', 'scipy>=0.17.1', 'sympy>=1.0', 'matplotlib>=1.5.1', 'tqdm>=4.7.6']
    )
