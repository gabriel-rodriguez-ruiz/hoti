# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 17:19:46 2022

@author: gabri
"""
import numpy as np

# Pauli matrices
sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
tau_0 = np.eye(2)
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

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
       
       H = \frac{1}{2} \sum_n^{L_x} \sum_m^{L_y} (-\mu \vec{c}^\dagger_{n,m} \tau_z\sigma_0  \vec{c}_{n,m}) +
           \frac{1}{2} \sum_n^{L_x-1} \sum_m^{L_y} \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} + H.c. \right) +
           \frac{1}{2} \sum_n^{L_x} \sum_m^{L_y-1} \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1} + H.c. \right) 
    """
    M = np.zeros((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    onsite = -mu/4 * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite[alpha, beta]   
    hopping_x = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_x)
    for i in range(1, L_x):
      for j in range(1, L_y+1):    
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i+1, j, beta, L_x, L_y)] = hopping_x[alpha, beta]
    hopping_y = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_y)
    for i in range(1, L_x+1):
      for j in range(1, L_y): 
        for alpha in range(4):
          for beta in range(4):
            M[index(i, j, alpha, L_x, L_y), index(i, j+1, beta, L_x, L_y)] = hopping_y[alpha, beta]
    return M + M.conj().T

def Hamiltonian_A1u_semi_infinite(k, t, mu, L_x, Delta):
    r"""Returns the H_k matrix for A1u model with:

    .. math::
        H_{A1u} = \frac{1}{2}\sum_k H_k
        
        H_k = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 +
            \Delta sin(k_y)\tau_x\sigma_y \right] \vec{c}_n +
            \sum_n^{L-1}\vec{c}^\dagger_n(-t\tau_z\sigma_0 + \frac{\Delta}{2i}\tau_x\sigma_x)\vec{c}_{n+1}
            + H.c.
        
        \xi_k = -2t\cos(k)-\mu    
        
       \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    """
    M = np.zeros((4*L_x, 4*L_x), dtype=complex)
    onsite = (-mu/4 - t/2*np.cos(k)) * np.kron(tau_z, sigma_0)   # para no duplicar al sumar la traspuesta
    for i in range(1, L_x+1):
        for alpha in range(4):
            for beta in range(4):
                M[index_semi_infinite(i, alpha, L_x), index_semi_infinite(i, beta, L_x)] = onsite[alpha, beta] 
    hopping = -t/2 * np.kron(tau_z, sigma_0) - 1j*Delta/4 * np.kron(tau_x, sigma_x)
    for i in range(1, L_x):
        for alpha in range(4):
            for beta in range(4):
                M[index_semi_infinite(i, alpha, L_x), index_semi_infinite(i+1, beta, L_x)] = hopping[alpha, beta]
    return M + M.conj().T

def Zeeman(theta, Delta_Z, L_x, L_y):
    r""" Return the Zeeman Hamiltonian matrix in 2D.
    
    .. math::
        H_Z = \frac{\Delta_Z}{2} \sum_n^{L_x} \sum_m^{L_y} \vec{c}^\dagger_{n,m}
        \tau_0(\cos(\theta)\sigma_x + \sin(\theta)\sigma_y)\vec{c}_{n,m}
    
        \vec{c}_{n,m} = (c_{n,m,\uparrow},
                         c_{n,m,\downarrow},
                         c^\dagger_{n,m,\downarrow},
                         -c^\dagger_{n,m,\uparrow})^T
    """
    M = np.zeros((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
    onsite = Delta_Z/2*( np.cos(theta)*np.kron(tau_0, sigma_x) +
                        np.sin(theta)*np.kron(tau_0, sigma_y) )
    for i in range(1, L_x+1):
      for j in range(1, L_y+1):
          for alpha in range(4):
              for beta in range(4):
                  M[index(i, j, alpha, L_x, L_y), index(i, j, beta, L_x, L_y)] = onsite[alpha, beta]
    return M