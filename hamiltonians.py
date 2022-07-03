# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 17:19:46 2022

@author: gabri
"""
import numpy as np

def index(i, j, alpha, L_x, L_y):
  """Return the index of basis vector given the site (i,j)
  and spin index alpha for i in {1, ..., L_x} and
  j in {1, ..., L_y}
  
  .. math ::
     (c_{11}, c_{12}, ..., c_{1L_x}, c_{21}, ..., c_{L_xL_y})^T
     
  """
  if (i>L_x or j>L_y):
      raise Exception("Site index should not be greater than samplesize.")
  return alpha + 4*( L_y*(i-1) + j - 1)

def Hamiltonian_A1u(t, mu, L_x, L_y, Delta):
    r"""Return the matrix for A1u model with:

    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
       
       H = \frac{1}{2} \sum_n^{L_x} \sum_m^{L_y} (-\mu \vec{c}^\dagger_{n,m} \vec{c}_{n,m}) +
           \frac{1}{2} \sum_n^{L_x-1} \sum_m^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
           \frac{1}{2} \sum_n^{L_x} \sum_m^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
    """
    M = np.zeros((4*L_x*L_y, 4*L_x*L_y))
    for alpha in range(4):
      for i in range(1, L_x+1):
        for j in range(1, L_y+1):
          M[index(i, j, alpha, L_x, L_y), index(i, j, alpha, L_x, L_y)] = -mu
      for i in range(1, L_x):
        for j in range(1, L_y+1):      
          M[index(i, j, alpha, L_x, L_y), index(i+1, j, alpha, L_x, L_y)] = -t
      for i in range(1, L_x+1):
        for j in range(1, L_y):      
          M[index(i, j, alpha, L_x, L_y), index(i, j+1, alpha, L_x, L_y)] = -t
