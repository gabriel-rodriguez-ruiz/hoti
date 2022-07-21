#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:59:07 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_A1u_S

L_x = 20
L_y = 100
t = 1
Delta = 1
mu = -2
Phi = np.pi/4   #superconducting phase
t_J = t/2    #t/2

params = dict(t=t, mu=mu, Delta=Delta, t_J=t_J,
              Phi=np.round(Phi, 3))

eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian_A1u_S(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta, t_J=t_J, Phi=Phi))
zero_modes = eigenvectors[:, 2*(L_x*L_y-1):2*(L_x*L_y+1)]      #4 (2) modes with zero energy (with Zeeman)

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
for i in range(4):      #each list has 4 elements corresponding to the 4 degenerated energies, if Zeeman is on only index 1 and 2 are degenerate
    probability_density[:,:,i] = np.abs(creation_up[i])**2 + np.abs(creation_down[i])**2 + np.abs(destruction_down[i])**2 + np.abs(destruction_up[i])**2

fig, ax = plt.subplots(num="Zeeman", clear=True)
image = ax.imshow(probability_density[:,:,2].T, cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
#plt.plot(probability_density[10,:,0])

#%% Energies

plt.figure("Energies", clear=True)
plt.scatter(np.arange(0, len(eigenvalues), 1), eigenvalues)
plt.xlim([2*(L_x*L_y-5), 2*(L_x*L_y+5)])
plt.ylim([-0.1, 0.1])
plt.ylabel("Energy")
plt.xlabel("Index")