=====================
earth-movers-distance
=====================

This package contains a CUDA-accelerated implementation of the earth mover's distance or Wasserstien metric in 2, 3, and 4 dimensions. The implementation is taken from Li, Wuchen et al (2018) [#f1]_, but is extended to dimensions higher than two. The CUDA acceleration is provided by ``cupy``. Code samples for each dimension can be found in ``tests/test_emd.py``.

Sample Usage (3D example):

.. code-block:: python

  from earth_movers_distance import distance
  import numpy as np

  N = 64
  spacing = np.linspace(-10, 10, N)
  x,y,z = np.meshgrid(spacing, spacing, spacing)

  sigma = 1
  mu = np.array([1., -1., 3.])
  source = np.exp(-.5*(x**2+y**2+z**2)/sigma)
  dest = np.exp(-.5*((x-mu[0])**2+(y-mu[1])**2+(z-mu[2])**2)/sigma)
  source/=source.sum()
  dest/=dest.sum()
  computed_distance = distance(source, dest, maxiter=200000, tau=3, mu=1e-7, dx=spacing[1]-spacing[0])
  actual_distance = (mu**2).sum()**.5
  print(f"Computed distance: {computed_distance} - actual: {actual_distance}")

The above example computes the earth mover's distance or Wasserstein metric between two Gaussians with identical covariance and differing means. In this case, the Wasserstein metric is simply equal to the Euclidean distance between the means.

Notes
==========

The default settings for the parameters ``tau`` and ``mu`` are taken from the original paper, but may need to be tuned for a given problem. Currently, ``dx`` is constant across all dimensions and grid points. Both input arrays should be normalized to sum to one, as done in the example.

Dependencies
============

The only dependency is``cupy``, which enables GPU-accelerated array operations. By default the CUDA 12 variant of ``cupy`` is installed, but you can change this by modifying ``setup.cfg``. It's possible to use AMD GPUs with ``cupy``, but this feature is still experimental (see https://docs.cupy.dev/en/stable/install.html).

Installation
============

Simply clone the repository and install with ``pip``:

.. code-block:: bash

  git clone https://github.com/UnityHPC/earth-movers-distance.git
  cd earth-movers-distance
  pip install .

.. rubric:: References

.. [#f1] Li, Wuchen, Ernest K. Ryu, Stanley Osher, Wotao Yin, and Wilfrid Gangbo. "A parallel method for earth mover’s distance." Journal of Scientific Computing 75, no. 1 (2018): 182-197.
