#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:31:17 2022

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_ZKM, Zeeman

L_x = 60
L_y = 60
t = 1
mu = -2*t   # mu=t*Delta_0/Delta_1
Delta_Z = 0  # 0.2
theta = np.pi/2
phi = 0
Delta_0 = -0.4*t*4
Delta_1 = 0.2*t*4
Lambda = 0.5*t

params = dict(t=t, mu=mu, Delta_0=Delta_0,
              Delta_Z=Delta_Z, theta=theta,
              Delta_1=Delta_1, Lambda=Lambda,
              phi=phi)

eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian_ZKM(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta_0=Delta_0, Delta_1=Delta_1, Lambda=Lambda) +
                                           Zeeman(theta=theta, Delta_Z=Delta_Z, L_x=L_x, L_y=L_y, phi=phi))
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

#%%
fig, ax = plt.subplots(num="Zeeman", clear=True)
image = ax.imshow(probability_density[:,:,2].T, cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
#plt.plot(probability_density[0,:,0])

#%% Energies

plt.figure("Energies", clear=True)
plt.scatter(np.arange(0, len(eigenvalues), 1), eigenvalues)
plt.xlim([2*(L_x*L_y-5), 2*(L_x*L_y+5)])
plt.ylim([-0.1, 0.1])