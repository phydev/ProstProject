#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    implementation of calculation of Laplace operator by loop
    and subsequent calculation of phi and sigma(nutrients)
"""

import numpy as np

def check_boundary(x,x0,x1):
    if x < x0:
        x = x0
    elif x >= x1:
        x = x1 - 1
    return x

def laplace(f,h):
    sx=f.shape[0]
    sy=f.shape[1]
    lap= np.zeros((sx,sy))
    #boundary conditions
    
    for x in np.arange(sx):
        for y in np.arange(sy):
            xl = check_boundary(x-h,0,sx)
            yl = check_boundary(y-h,0,sy)
            xu = check_boundary(x+h,0,sx)
            yu = check_boundary(y+h,0,sy)
            
            lap[x,y]=(f[xu,y]+f[xl,y]+f[x,yu]+f[x,yl]-4.*f[x,y])/h/h
    return lap
    
def potential(f):
    return f*(1.-f)*(f-.5)

def der_order(f,lamb,tau,chi,nut,A,h=1):
    return lamb * laplace(f,h) + (1./tau) * potential(f) + chi * f * nut - A * f

def der_nut(nut,phi,s,epsilon,delta,gamma,h=1):
    return epsilon * laplace(nut,h) + s - delta * phi - gamma * nut

def euler(f,dt,der):
    fnew = f + dt * der
    return fnew