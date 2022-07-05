#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:48:52 2022

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_A1u, Zeeman

L_x = 20
L_y = 30
t = 1
Delta = 1
mu = -2
Delta_Z = 0
theta = np.pi/4

eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian_A1u(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta) +
                                           Zeeman(theta=theta, Delta_Z=Delta_Z, L_x=L_x, L_y=L_y))
zero_modes = eigenvectors[:, 2*(L_x*L_y-1):2*(L_x*L_y+1)]      #4 modes with zero energy

creation_up = []  
creation_down = []
destruction_down = []
destruction_up = []
for i in range(4):      #each list has 4 elements corresponding to the 4 degenerated energies
    destruction_up.append((zero_modes[0::4, i]).reshape((L_x, L_y)))
    destruction_down.append((zero_modes[1::4, i]).reshape((L_x, L_y)))
    creation_down.append((zero_modes[2::4, i]).reshape((L_x, L_y)))
    creation_up.append((zero_modes[3::4, i]).reshape((L_x, L_y)))

probability_density = np.zeros((L_x,L_y, 4))
for i in range(4):      #each list has 4 elements corresponding to the 4 degenerated energies
    probability_density[:,:,i] = np.abs(creation_up[i])**2 + np.abs(creation_down[i])**2 + np.abs(destruction_down[i])**2 + np.abs(destruction_up[i])**2
plt.imshow(probability_density[:,:,0])
plt.colorbar()

#plt.figure()
#plt.plot(probability_density[10,:,0])