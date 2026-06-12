.. _migration:

****************************
Migrating from bayesloop 1.x
****************************

*bayesloop* 2.0 renames the entire API from camelCase to snake_case (e.g. ``S.loadData(...)`` becomes ``S.load_data(...)``). Existing 1.x scripts can still be run without modification by importing the opt-in compatibility layer once, right after importing *bayesloop*:

.. code-block:: python

    import bayesloop as bl
    import bayesloop.v1compat  # activates the 1.x API

    S = bl.HyperStudy()
    S.loadExampleData()  # 1.x method names work again

The compatibility layer restores:

- all camelCase method, attribute and property names (``loadData``, ``getParameterDistribution``, ``logEvidence``, ...), including the 1.x shorthand aliases (``setOM``, ``setTM``, ``getPD``, ...)
- camelCase keyword arguments (``forwardOnly``, ``nJobs``, ``storeHistory``, ...)
- the 1.x module names ``bayesloop.observationModels``, ``bayesloop.transitionModels`` and ``bayesloop.fileIO``
- custom observation/transition models that implement 1.x hooks (``computeForwardPrior``, ``estimateParameterValues``, ...)
- loading study files saved with *bayesloop* 1.x via ``bl.load(...)`` (attribute names are migrated on load)

Every use of a 1.x name emits a ``DeprecationWarning`` that points to its snake_case replacement, so the layer doubles as a migration guide.

Two things are **not** covered by the compatibility layer:

- The probability ``Parser`` and ``Study.eval()`` were removed in 2.0.
- The default parameter names of some transition models changed (``'tChange'`` → ``'t_change'``, ``'tBreak'`` → ``'t_break'``, ``'log10pMin'`` → ``'log10p_min'``) — scripts that rely on these default names should pass the names explicitly.
