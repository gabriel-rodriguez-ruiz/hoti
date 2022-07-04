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

def index_semi_infinite(i, alpha, L_x):
  """Return the index of basis vector given the site i
  and spin index alpha for i in {1, ..., L_x}
  
  .. math ::
     (c_{1}, c_{2}, ..., c_{L_x})^T
     
  """
  if i>L_x:
      raise Exception("Site index should not be greater than samplesize.")
  return alpha + 4*(i - 1)


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
    M = np.zeros((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(2):
            M[index(i, j, alpha, L_x, L_y), index(i, j, alpha, L_x, L_y)] = -mu/4   # para no duplicar al sumar la traspuesta 
            M[index(i, j, alpha+2, L_x, L_y), index(i, j, alpha+2, L_x, L_y)] = mu/4   # para no duplicar al sumar la traspuesta 
    for i in range(1, L_x):
      for j in range(1, L_y+1):      
        M[index(i, j, 0, L_x, L_y), index(i+1, j, 0, L_x, L_y)] = -t/2
        M[index(i, j, 1, L_x, L_y), index(i+1, j, 1, L_x, L_y)] = -t/2
        M[index(i, j, 2, L_x, L_y), index(i+1, j, 2, L_x, L_y)] = t/2
        M[index(i, j, 3, L_x, L_y), index(i+1, j, 3, L_x, L_y)] = t/2
        M[index(i, j, 0, L_x, L_y), index(i+1, j, 3, L_x, L_y)] = -1j*Delta/4
        M[index(i, j, 1, L_x, L_y), index(i+1, j, 2, L_x, L_y)] = -1j*Delta/4
        M[index(i, j, 2, L_x, L_y), index(i+1, j, 1, L_x, L_y)] = -1j*Delta/4
        M[index(i, j, 3, L_x, L_y), index(i+1, j, 0, L_x, L_y)] = -1j*Delta/4
    for i in range(1, L_x+1):
      for j in range(1, L_y):      
        M[index(i, j, 0, L_x, L_y), index(i, j+1, 0, L_x, L_y)] = -t/2
        M[index(i, j, 1, L_x, L_y), index(i, j+1, 1, L_x, L_y)] = -t/2
        M[index(i, j, 2, L_x, L_y), index(i, j+1, 2, L_x, L_y)] = t/2
        M[index(i, j, 3, L_x, L_y), index(i, j+1, 3, L_x, L_y)] = t/2
        M[index(i, j, 0, L_x, L_y), index(i, j+1, 3, L_x, L_y)] = -Delta/4
        M[index(i, j, 1, L_x, L_y), index(i, j+1, 2, L_x, L_y)] = Delta/4
        M[index(i, j, 2, L_x, L_y), index(i, j+1, 1, L_x, L_y)] = -Delta/4
        M[index(i, j, 3, L_x, L_y), index(i, j+1, 0, L_x, L_y)] = Delta/4
    return M + M.conj().T

def Hamiltonian_A1u_semi_infinite(k, t, mu, L_x, Delta):
    r"""Returns the H_k matrix for A1u model with:

    .. math::
        H_{A1u} = \frac{1}{2}\sum_k H_k
        
        H_k = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta sin(k_y)\tau_x\sigma_y \right] +
            \sum_n^{L-1}\vec{c}^\dagger_n(-t\tau_z\sigma_0 + \frac{\Delta}{2i}\tau_x\sigma_x)\vec{c}_{n+1}
            + H.c.
            
       \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    """
    M = np.zeros((4*L_x, 4*L_x), dtype=complex)
    for i in range(1, L_x+1):
        M[index_semi_infinite(i, 0, L_x), index_semi_infinite(i, 0, L_x)] = -t/2*np.cos(k) - mu/4   # para no duplicar al sumar la traspuesta 
        M[index_semi_infinite(i, 1, L_x), index_semi_infinite(i, 1, L_x)] = -t/2*np.cos(k) - mu/4
        M[index_semi_infinite(i, 2, L_x), index_semi_infinite(i, 2, L_x)] = t/2*np.cos(k) + mu/4
        M[index_semi_infinite(i, 3, L_x), index_semi_infinite(i, 3, L_x)] = t/2*np.cos(k) + mu/4
        M[index_semi_infinite(i, 0, L_x), index_semi_infinite(i, 3, L_x)] = -1j*Delta/2*np.sin(k)
        M[index_semi_infinite(i, 1, L_x), index_semi_infinite(i, 2, L_x)] = 1j*Delta/2*np.sin(k)
        M[index_semi_infinite(i, 2, L_x), index_semi_infinite(i, 1, L_x)] = -1j*Delta/2*np.sin(k)
        M[index_semi_infinite(i, 3, L_x), index_semi_infinite(i, 0, L_x)] = 1j*Delta/2*np.sin(k)
    for i in range(1, L_x):
        M[index_semi_infinite(i, 0, L_x), index_semi_infinite(i+1, 0, L_x)] = -t/2
        M[index_semi_infinite(i, 1, L_x), index_semi_infinite(i+1, 1, L_x)] = -t/2   
        M[index_semi_infinite(i, 2, L_x), index_semi_infinite(i+1, 2, L_x)] = t/2
        M[index_semi_infinite(i, 3, L_x), index_semi_infinite(i+1, 3, L_x)] = t/2  
        M[index_semi_infinite(i, 0, L_x), index_semi_infinite(i+1, 3, L_x)] = -1j*Delta/2  
        M[index_semi_infinite(i, 1, L_x), index_semi_infinite(i+1, 2, L_x)] = -1j*Delta/2  
        M[index_semi_infinite(i, 2, L_x), index_semi_infinite(i+1, 1, L_x)] = -1j*Delta/2  
        M[index_semi_infinite(i, 3, L_x), index_semi_infinite(i+1, 0, L_x)] = -1j*Delta/2  
    return M + M.conj().T

def Zeeman(theta, Delta_Z, L_x, L_y):
    """ Return the Zeeman Hamiltonian matrix in 2D. 
    """
    M = np.zeros((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        M[index(i, j, 0, L_x, L_y), index(i, j, 3, L_x, L_y)] = Delta_Z/2*(np.cos(theta)-1j*np.sin(theta))
        M[index(i, j, 1, L_x, L_y), index(i, j, 2, L_x, L_y)] = Delta_Z/2*(np.cos(theta)+1j*np.sin(theta))
        M[index(i, j, 2, L_x, L_y), index(i, j, 1, L_x, L_y)] = Delta_Z/2*(np.cos(theta)-1j*np.sin(theta))
        M[index(i, j, 3, L_x, L_y), index(i, j, 0, L_x, L_y)] = Delta_Z/2*(np.cos(theta)+1j*np.sin(theta))
    return M + M.conj().T