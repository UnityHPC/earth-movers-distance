#from cuda.core import experimental as pycuda
from earth_movers_distance import cpp
from importlib import resources
import cupy as cp
import numpy as np
from typing import List

kernel_names = ["mUpdate_l1", "mUpdate_l2", "PhiUpdate"]

def get_source_path(name):
    with resources.path(cpp, name) as path:
        return path.as_posix()

def read_source(name):
    return resources.files(cpp).joinpath(name).read_text()

def load_kernels():
    module = cp.RawModule(code='extern "C"{'+read_source("kernels2d.cu")+'}')

    return {k: module.get_function(k) for k in kernel_names}


def l2_update(phi: cp.ndarray, m: cp.ndarray, rhodiff: cp.ndarray, tau=3, mu=3e-6, dx = None):
    """Do an L2 update."""

    if dx is None: dx = 1./phi.shape[0]
    dim = len(phi.shape)
    grad = cp.stack(cp.gradient(phi, dx), axis=-1)
    m_new = m + mu * grad
    norm = cp.maximum(mu, cp.sqrt(cp.sum(m_new**2, axis=-1)))
    shrink_factor = 1 - mu / norm
    m_new *= shrink_factor[..., None]
    m_temp = 2*m_new - m
    divergence = cp.gradient(m_temp[..., 0], dx, axis=0)
    for axis in range(1, dim):
        divergence += cp.gradient(m_temp[..., axis], dx, axis=axis)
    phi += tau * (divergence + rhodiff)
    m[:] = m_new[:]

def l2_distance(source: np.ndarray, dest: np.ndarray, maxiter=100, **kwargs):
    """Compute L2 earth mover's distance between two N-dimensional arrays."""

    rhodiff = cp.array(dest-source)
    phi = cp.zeros_like(rhodiff)
    m = cp.zeros(phi.shape + (len(phi.shape),))

    for i in range(maxiter):
        l2_update(phi, m, rhodiff, **kwargs)
        if i %10 == 0:
            print(f"Iteration: {i}, L2 distance", cp.sum(cp.sqrt(cp.sum(m**2,axis=-1))))


"""
def load_kernels():
    dev = pycuda.Device()
    dev.set_current()
    capability = dev.compute_capability
    compilation_options = (f"-arch=sm_{capability.major}{capability.minor}",)
    source = read_source("kernels2d.cu")
    prog = pycuda.Program(source, "c++")
    kernel_names = ["mUpdate_l1", "mUpdate_l2", "PhiUpdate"]
    obj_code = prog.compile("cubin",  options=compilation_options, name_expressions=kernel_names)
    for k in kernel_names:
        print(k, obj_code.get_kernel(k))
    return {k: obj_code.get_kernel(k) for k in kernel_names}"""

if __name__ == "__main__":
    spacing = np.linspace(0,1,128)
    x, y = np.meshgrid(spacing, spacing)
    source = np.exp(3*((.5-x)**2+3*(.5-y)**2))
    dest = np.exp(3*((.3-x)**2+3*(.3-y)**2))
    source /= source.sum()
    dest /= dest.sum()
    
    l2_distance(source, dest, maxiter=2000, dx=1./len(spacing))
