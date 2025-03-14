import cupy as cp
import numpy as np

MAX_DIM = 4

def l2_update(phi: cp.ndarray, m: cp.ndarray, m_temp: cp.ndarray, rhodiff: cp.ndarray, tau=3, mu=3e-6, dx = None):
    """Do an L2 update."""

    if dx is None: dx = 1./phi.shape[0]
    dim = len(phi.shape)
    m_temp[:] = -m
    m[0, :-1] += mu * (phi[1:] - phi[:-1]) / dx
    if dim > 1:
        m[1, :, :-1] += mu * (phi[:, 1:] - phi[:, :-1]) / dx
    if dim > 2:
        m[2, :, :, :-1] += mu * (phi[:, :, 1:] - phi[:, :, :-1]) / dx
    if dim > 3:
        m[3, :, :, :, :-1] += mu * (phi[:, :, :, 1:] - phi[:, :, :, :-1]) / dx
    
    norm = cp.sqrt(cp.sum(m**2, axis=0))
    shrink_factor = 1 - mu / cp.maximum(norm, mu)
    #shrink_factor[norm < mu] = 0
    m *= shrink_factor[None, ...]
    m_temp += 2*m
    divergence = m_temp.sum(axis=0)
    divergence[1:] -= m_temp[0, :-1]
    if dim > 1: divergence[:, 1:] -= m_temp[1, :, :-1]
    if dim > 2: divergence[:, :, 1:] -= m_temp[2, :, :, :-1]
    if dim > 3: divergence[:, :, :, 1:] -= m_temp[3, :, :, :, :-1]
    
    phi += tau * (divergence/dx + rhodiff)

def l2_distance(source: np.ndarray, dest: np.ndarray, maxiter=100, tau=3, mu=None, **kwargs):
    """Compute L2 earth mover's distance between two N-dimensional arrays."""

    if len(source.shape) > MAX_DIM:
        raise ValueError(f"Dimensions of greater than {MAX_DIM} are not supported!")
    elif source.shape != dest.shape:
        raise ValueError(f"Dimension mismatch between source and destination! Source shape is '{source.shape}', dest shape is '{dest.shape}'.")
    rhodiff = cp.array(dest-source)
    if mu is None: mu = 1./(tau*16*rhodiff.size)
    phi = cp.zeros_like(rhodiff)
    m = cp.zeros((len(phi.shape),) + phi.shape)
    m_temp = cp.zeros_like(m)
    for i in range(maxiter):
        l2_update(phi, m, m_temp, rhodiff, **kwargs)
        if i %1000 == 0:
            print(f"Iteration: {i}, L2 distance", cp.sum(cp.sqrt(cp.sum(m**2,axis=0))))

if __name__ == "__main__":
    N = 256 
    spacing = np.linspace(-5,5,N)
    x, y = np.meshgrid(spacing, spacing)
    source = np.exp(-((.8-x)**2+(.8-y)**2)/2) / (2*np.pi) ** .5
    dest = np.exp(-((.6-x)**2+(.6-y)**2)/2) / (2*np.pi) ** .5
    #source = np.zeros_like(x)
    #source[((x-.3)**2+(y-.3)**2) < .01] = 1
    #dest = np.zeros_like(x)
    #dest[((x-.7)**2+(y-.7)**2) < .01] = 1
    dx = spacing[1]-spacing[0]
    tau = 3
    mu = 1./(16*tau*(N-1)**2)
    source /= source.sum()
    dest /= dest.sum()
    l2_distance(source, dest, maxiter=40000, dx=dx, tau=tau)
