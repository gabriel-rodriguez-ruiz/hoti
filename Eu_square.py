# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 14:16:11 2022

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_Eu, Zeeman

L_x = 30
L_y = 30
t = 1
Delta = 1
mu = -2
Delta_Z = 0.2   #0.2
theta = 0
phi = 0

params = dict(t=t, mu=mu, Delta=Delta,
              Delta_Z=Delta_Z, theta=theta, phi=phi)

eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian_Eu(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta) +
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