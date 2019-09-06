"""
Tuesday, 3 September 2019
@author: moreira
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def init_tumor(phi, radius, position):
    """
     initializes the field phi with a circle
     :param phi: order parameter
     :param radius: tumor initial radius
     :param position: where the position is tumor will be located
    """
    for i in range(0, phi.size):
        s = np.asarray(np.unravel_index(i, phi.shape))  # vector position
        distance = np.sqrt(np.sum(np.power(s - position, 2)))
        if distance <= radius * np.sin(np.random.rand()):
            phi.itemset(i, 1.0)

    return phi


def init_chemical_field(sigma, S, radius, position):
    """
      initialize the chemical field sigma and the nutrient sources
    :param sigma: nutrient chemical field
    :param S: source of the nutrient
    :param radius: the radius of the region where the sources will be placed
    :param position: the center of the region
    :return:
    """
    for i in range(0, sigma.size):
        s = np.asarray(np.unravel_index(i, sigma.shape))
        #distance = np.sqrt(np.sum(np.power(s - position, 2)))
        #if distance >= radius* np.sin(np.random.rand()) + 10:
        S.itemset(i, 2.75 )
        sigma.itemset(i, 2.75 )
    return sigma, S


def integrate(phi, sigma, S, tstep, dt, h, **kwargs):
    """
      integrate the order parameter phi and the chemical field sigma on time
    :param phi: order parameter
    :param sigma: nutrient
    :param S: source of nutrient
    :param tstep: total number of timesteps
    :param dt: time increment
    :param kwargs: dictionary of parameters lambda, tau, chi, A, epsilon, delta, gamma
    :return:
    """
    # five point stencil - https://en.wikipedia.org/wiki/Five-point_stencil
    stencil = (1.0 / (12.0 * h * h)) * np.array(
        [[0, 0, -1, 0, 0],
         [0, 0, 16, 0, 0],
         [-1, 16, -60, 16, -1],
         [0, 0, 16, 0, 0],
         [0, 0, -1, 0, 0]])

    nstep = 0
    nprint = 2000
    while nstep <= tstep:
        phio = np.copy(phi)
        phi = phi + dt * (kwargs['lambda_'] * convolve(phi, stencil, mode='nearest') +
                          kwargs['tau'] * phi * (1.0 - phi) * (phi - 0.5) +
                          kwargs['chi'] * sigma * phi - kwargs['A'] * phi)

        sigma = sigma + dt * (kwargs['epsilon'] * convolve(sigma, stencil, mode='nearest') +
                              S - kwargs['delta'] * phio - kwargs['gamma'] * sigma)

        if (nprint >= 2000):
            print(nstep)
            nprint = 0
            plt.imshow(phi)
            plt.show()
        nstep += 1
        nprint += 1

    return phi

if __name__ == '__main__':
    L = np.array([100, 100])  # system length
    phi = np.zeros(L)  # allocating the array for the order paramter
    sigma = np.zeros(L)  # allocating the array for the chemical field
    S = np.zeros(L)

    phi = init_tumor(phi, radius=20, position=L/2)  # initializing the tumor field
    sigma, S = init_chemical_field(sigma, S, radius=20, position=L / 2)  # initializing the nutrient field and sources
    plt.imshow(sigma)
    plt.show()
    plt.imshow(S)
    plt.show()
    phi_end = integrate(phi, sigma, S, tstep=40000, dt=0.001, h=1., lambda_=1., epsilon=10.,
              A=4., gamma=0.1, tau=5., chi=0.2, delta=0.1)  # time integration
