from earth_movers_distance import distance
import numpy as np

def test_4d():
    N = 32 
    spacing = np.linspace(-10, 10, N)
    x,y,z,w = np.meshgrid(spacing, spacing, spacing, spacing)

    sigma = 1
    mu = np.array([1., -1., 3., 5**.5])
    source = np.exp(-.5*(x**2+y**2+z**2+w**2)/sigma)
    dest = np.exp(-.5*((x-mu[0])**2+(y-mu[1])**2+(z-mu[2])**2+(w-mu[3])**2)/sigma)
    source/=source.sum()
    dest/=dest.sum()
    computed_distance = distance(source, dest, maxiter=200000, tau=3, mu=3e-6, dx=spacing[1]-spacing[0])
    actual_distance = (mu**2).sum()**.5
    print(f"Computed distance: {computed_distance} - actual: {actual_distance}")
    assert abs(computed_distance-actual_distance) < .04

def test_3d():
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
    assert abs(computed_distance-actual_distance) < .04

def test_2d():
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
    computed_distance = distance(source, dest, maxiter=200000, mu=mu, dx=dx, tau=tau)
    actual_distance = .2 * 2**.5
    print(f"Computed distance: {computed_distance} - actual: {actual_distance}")
    assert abs(computed_distance-actual_distance) < .04

if __name__ == "__main__":
    #test_4d()
    #test_3d()
    test_2d()
