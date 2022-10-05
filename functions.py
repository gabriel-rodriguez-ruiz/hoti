#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:26:05 2022

@author: usuario
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

# Spin operators
S_x = np.kron(tau_z, sigma_x)
S_y = np.kron(tau_z, sigma_y)
S_z = np.kron(tau_z, sigma_z)

def spectrum(system, k_values, **params):
    """Returns an array whose rows are the eigenvalues of the system for
    a definite k. System should be a function that returns an array.
    """
    eigenvalues = []
    for k in k_values:
        params["k"] = k
        H = system(**params)
        energies = np.linalg.eigvalsh(H)
        energies = list(energies)
        eigenvalues.append(energies)
    eigenvalues = np.array(eigenvalues)
    return eigenvalues

def mean_spin(state):
    """Returns a 1D-array of length 3 with the spin mean value of the state.
    State should be a vector of length 4.
    """
    spin_mean_value = np.concatenate([state.T.conj()@S_x@state,
                                      state.T.conj()@S_y@state,
                                      state.T.conj()@S_z@state])
    return spin_mean_value

def mean_spin_xy(state):
    N_x, N_y, N_z = np.shape(state)     #N_z is always 4
    spin_mean_value = np.zeros((N_x, N_y, 3))
    for i in range(N_x):
        for j in range(N_y):
            for k in range(3):
                spin_mean_value[i, j, k] = mean_spin(np.reshape(state[i,j,:], (4,1)))[k][0]
    return spin_mean_value