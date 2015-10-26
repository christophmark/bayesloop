# bayesloop
<sup>[project page](http://christophmark.github.io/bayesloop/)</sup><br>
Time series analysis today is an important cornerstone of quantitative science in many disciplines, including natural and life sciences as well as economics and social sciences. Regarding diverse phenomena like tumor cell migration, brain activity and stock trading, a similarity of these complex systems becomes apparent: the observable data we measure – cell migration paths, neuron spike rates and stock prices – are the result of a multitude of underlying processes that act over a broad range of spatial and temporal scales. It is thus to expect that the statistical properties of these systems are not constant, but themselves show stochastic or deterministic dynamics of their own. Time series models used to understand the dynamics of complex systems therefore have to account for temporal changes of the models' parameters.

*bayesloop* is a python module that focuses on fitting time series models with time-varying parameters and model selection based on [Bayesian inference](https://cocosci.berkeley.edu/tom/papers/tutorial.pdf). Instead of relying on [MCMC methods](http://www.cs.ubc.ca/~arnaud/andrieu_defreitas_doucet_jordan_intromontecarlomachinelearning.pdf), *bayesloop* uses a grid-based approach to evaluate probability distributions, allowing for an efficient approximation of the [marginal likelihood (evidence)](http://alumni.media.mit.edu/~tpminka/statlearn/demo/). The marginal likelihood represents a powerful tool to objectively compare different models and/or optimize the hyper-parameters of hierarchical models. To avoid the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) when analyzing time series models with time-varying parameters, *bayesloop* employs a sequential inference algorithm that is based on the [forward-backward-algorithm](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm) used in [Hidden Markov models](http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf). Here, the relevant parameter spaces are kept low-dimensional by processing time series data step by step. The module covers a large class of time series models and is easily extensible.

The underlying algorithm of *bayesloop* has been successfully employed in cancer research, studying the migration paths of invasive tumor cells, see this [article](http://www.nature.com/articles/ncomms8516).

## Features
* objective model selection based on Bayesian model evidence
* optimization of hyper-parameters in hierarchical time series models
* computes distributions of hyper-parameters
* computes parameter distributions for each time step
* analysis of change-points and structural breaks
* suitable for online analysis as well as retrospective analysis
* straight-forward handling of missing data points

Note that the module is currently in beta state, so features may change in the future.

## Getting started
For a comprehensive introduction and overview of the main features that *bayesloop* provides, see the [tutorial](http://nbviewer.ipython.org/github/christophmark/bayesloop/blob/master/docs/bayesloop_tutorial.ipynb). While you may read this tutorial directly in your browser, you can also download it and interactively manipulate/execute the code therein (see [here](https://ipython.org/ipython-doc/3/notebook/notebook.html#starting-the-notebook-server) for further information).

The following code provides a minimal example of an analysis carried out using *bayesloop*. The data here consists of the number of coal mining disasters in the UK per year from 1851 to 1962 (see this [article](http://www.dima.unige.it/~riccomag/Teaching/ProcessiStocastici/coal-mining-disaster-original%20paper.pdf) for further information).
```
import bayesloop as bl
import matplotlib.pyplot as plt

S = bl.Study()  # start new data study
S.loadExampleData()  # load data array

# observed number of disasters is modeled by Poisson distribution
M = bl.observationModels.Poisson()
S.setObservationModel(M)

# set boundaries for annual disaster-rate
S.setBoundaries([[0, 6]])

# disaster rate itself may change gradually over time
K = bl.transitionModels.GaussianRandomWalk(sigma=0.2)
S.setTransitionModel(K)

S.fit()  # Bayesian inference

# plot data together with inferred parameter evolution
plt.figure(figsize=(10, 5))
plt.xlim([1850, 1962])
plt.bar(range(1851, 1962), S.rawData,
        align='center', facecolor='r', alpha=.5)
S.plotParameterEvolution(xLower=1851, xUpper=1962)
plt.show()
```
This analysis indicates a significant improvement of safety conditions between 1880 and 1900. Check out the [tutorial](http://nbviewer.ipython.org/github/christophmark/bayesloop/blob/master/docs/bayesloop_tutorial.ipynb) for further insights!

## Installation
To install the latest version of *bayesloop*, download the [zipped version](https://github.com/christophmark/bayesloop/zipball/master) or clone the repository and install *bayesloop* using `python setup.py install`.

Another option is to install *bayesloop* from the master branch using pip (requires git):
```
pip install git+https://github.com/christophmark/bayesloop
```

## Dependencies
*bayesloop* is tested on Python 2.7 and depends on NumPy, SciPy, matplotlib and Seaborn.

## Optional dependencies
*bayesloop* uses [dill](https://pypi.python.org/pypi/dill), an extension to Python's [pickle](https://docs.python.org/2/library/pickle.html) module, to save/load on-going data studies to/from file. It can be installed via pip:
```
pip install dill
```

*bayesloop* further supports multiprocessing for computationally expensive analyses, based on the [pathos](https://github.com/uqfoundation/pathos) module. The latest version can be obtained directly from GitHub using pip (requires git):
```
pip install git+https://github.com/uqfoundation/pathos
```
Note: Windows users need to install a C compiler *before* installing pathos. One possible solution for 64bit systems is to install [Microsoft Visual C++ 2008 SP1 Redistributable Package (x64)](http://www.microsoft.com/en-us/download/confirmation.aspx?id=2092) and [Microsoft Visual C++ Compiler for Python 2.7](http://www.microsoft.com/en-us/download/details.aspx?id=44266).

## License
[The MIT License (MIT)](https://github.com/christophmark/bayesloop/blob/master/LICENSE)
