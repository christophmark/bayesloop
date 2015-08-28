# bayesloop
Time series analysis today is an important cornerstone of quantitative science in many disciplines, including natural and life sciences as well as economics and social sciences. Regarding diverse phenomena like tumor cell migration, brain activity and stock trading, a similarity of these complex systems becomes apparent: the observable data we measure – cell migration paths, neuron spike rates and stock prices – are the result of a multitude of underlying processes that act over a broad range of spatial and temporal scales. It is thus to expect that the statistical properties of these systems are not constant, but themselves show stochastic or deterministic dynamics of their own.

*bayesloop* is a python module for fitting time series models with time-varying parameters and model selection based on [Bayesian inference](https://cocosci.berkeley.edu/tom/papers/tutorial.pdf). Instead of relying on [MCMC methods](http://www.cs.ubc.ca/~arnaud/andrieu_defreitas_doucet_jordan_intromontecarlomachinelearning.pdf), *bayesloop* uses a grid-based approach to handle probability distributions, allowing for an efficient approximation of the [marginal likelihood (evidence)](http://alumni.media.mit.edu/~tpminka/statlearn/demo/). The marginal likelihood represents a powerful tool to objectively compare different models and/or optimize the hyper-parameters of a hierarchical model. To avoid the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) when analyzing time series models with time-varying parameters, *bayesloop* employs a sequential inference algorithm that is based on the [forward-backward-algorithm](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm) used in [Hidden Markov models](http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf). Here, the relevant parameter spaces are kept low-dimensional by processing time series data step by step. The module is easily extensible and covers a large class of time series models.

The algorithm on which *bayesloop* is based has been successfully employed in cancer research, studying the migration paths of invasive tumor cells, see this [article](http://www.nature.com/articles/ncomms8516).

## Features
* objective model selection based on Bayesian evidence (marginal likelihood)
* optimization of hyper-parameters in hierarchical time series models
* provides parameter distributions instead of point estimates
* suitable for online analysis as well as retrospective analysis
* straight-forward handling of missing data points

## Installation
To install the latest version of *bayesloop*, download the zipped version or clone the repository and install *bayesloop* using `python setup.py install`.

Another option is to install *bayesloop* from the master branch using pip:
```
pip install git+https://github.com/christophmark/bayesloop
```

## Dependencies
*bayesloop* is tested on Python 2.7 and depends on NumPy, SciPy, matplotlib and Seaborn.

## License
[The MIT License (MIT)](https://github.com/christophmark/bayesloop/blob/master/LICENSE)

