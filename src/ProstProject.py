"""
   Copyright (C) 2019 ProstProject

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
   02111-1307, USA.

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
        if distance <= radius :
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
        S.itemset(i, 0.1)
        sigma.itemset(i, 0.1)
    return sigma, S


def select_stencil(ndim, order, h):
    """
        give as input the number of dimensions, derivative accuracy and the spacing
        and it returns the stencil for calculating the laplace operator.
    :param dim: number of spatial dimensions
    :param order: derivatives accuracy
    :param h: lattice parameter/spacing
    :return:
    """

    #   order -> accuracy -> number of points: 0, 1, 2, 3 -> 2, 4, 6, 8 -> 3, 5, 7, 9
    accuracy = np.array([3, 5, 7, 9])
    stencil = np.zeros((accuracy[order],)*ndim)
    pattern = [np.array([1, -2, 1]),
               (1.0 / 12.0) * np.array([-1, 16, -30, 16, -1]),
               np.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]),
               np.array([-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560])
               ]

    if ndim==1:
        stencil[:] += pattern[order]
    elif ndim==2:
        stencil[np.int(accuracy[order] / ndim), :] += pattern[order]
        stencil[:, np.int(accuracy[order] / ndim)] += pattern[order]
    elif ndim==3:
        stencil[np.int(accuracy[order] / ndim), :, :] += pattern[order]
        stencil[:, np.int(accuracy[order] / ndim), :] += pattern[order]
        stencil[:, :, np.int(accuracy[order] / ndim)] += pattern[order]


    return stencil / (h * h)

def plot(phi, sigma=None):
    """
        select different plot functions from matplotlib accordingly with
        the dimensionality of phi.
    :param phi:
    :param sigma:
    :return:
    """

    if phi.ndim == 1:
        plt.plot(phi, label='Tumor')
        plt.plot(sigma, label='Nutrients')
        plt.legend()
    elif phi.ndim == 2:
        plt.imshow(phi)

    plt.show()

    return

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
    stencil = select_stencil(phi.ndim, 0, h)
    print(stencil)
    nstep = 0
    nprint = 100000
    while nstep <= tstep:
        phio = np.copy(phi)
        phi = phi + dt * (kwargs['lambda_'] * convolve(phi, stencil, mode='wrap') +
                          kwargs['tau'] * phi * (1.0 - phi) * (phi - 0.5) +
                          kwargs['chi'] * sigma * phi - kwargs['A'] * phi)

        sigma = sigma + dt * (kwargs['epsilon'] * convolve(sigma, stencil, mode='wrap') +
                              S - kwargs['delta'] * phio - kwargs['gamma'] * sigma)

        if (nprint >= 10000):
            print(nstep)
            nprint = 0
            plot(phi)
        nstep += 1
        nprint += 1

    return phi



if __name__ == '__main__':
    L = np.array([100,100])  # system length
    phi = np.zeros(L)  # allocating the array for the order parameter
    sigma = np.zeros(L)  # allocating the array for the chemical field
    S = np.zeros(L)

    phi = init_tumor(phi, radius=20, position=L/2)  # initializing the tumor field
    sigma, S = init_chemical_field(sigma, S, radius=20, position=L/2)  # initializing the nutrient field and sources

    plot(phi)

    phi_end = integrate(phi, sigma, S, tstep=100000, dt=0.001, h=0.05, lambda_=0.01, epsilon=0.10,
              A=1.9, gamma=0.1, tau=100., chi=2.0, delta=1.0)
