#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
        s = np.asarray(np.unravel_index(i, phi.shape)) # vector position
        distance = np.sqrt(np.sum(np.power(s - position,2)))
        if distance <= radius*np.sin(np.random.rand()):
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
        distance = np.sqrt(np.sum(np.power(s - position,2)))
        if  distance <= radius :
            S.itemset(i, 0.1 * np.random.rand())
            sigma.itemset(i, 0.1)
    return sigma, S

def integrate(phi, sigma, S, tstep, dt, **kwargs):
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
    stencil = (1.0 / (12.0 * dL * dL)) * np.array(
        [[0, 0, -1, 0, 0],
         [0, 0, 16, 0, 0],
         [-1, 16, -60, 16, -1],
         [0, 0, 16, 0, 0],
         [0, 0, -1, 0, 0]])
    
    nstep = 0
    nprint = 2000
    while (nstep <= tstep):

        phi = phi + dt * (kwargs['lambda_'] * convolve(phi, stencil, mode='nearest') +
                          kwargs['tau'] * phi * (1.0 - phi) * (phi - 0.5) +
                          kwargs['chi'] * sigma - kwargs['A'] * phi)

        sigma = sigma + dt * (kwargs['epsilon'] * convolve(sigma, stencil, mode='nearest') +
                              S - kwargs['delta'] * phi - kwargs['gamma'] * sigma)


        if(nprint>=2000):
            print(nstep)
            nprint=0

            plt.imshow(phi)
            plt.show()
        nstep += 1
        nprint += 1

        
if __name__ = '__main__':
    L = np.array([100, 100])
    phi = np.zeros(L)
    sigma = np.zeros(L)

    phi = init_tumor(phi, 20, L/2)
    sigma, S = init_chemical_field(sigma, S, 20, L/2)
    integrate(phi, sigma, S, tstep=10000, dt=0.01, lambda_=0.3, epsilon=lambda_*100,
          A=0.1, gamma=0.0, tau=5., chi=0.0, delta=0.)


