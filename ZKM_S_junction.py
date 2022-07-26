#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:59:07 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_ZKM_S

L_x = 40
L_y = 40
t = 1
mu = -2*t   # mu=t*Delta_0/Delta_1
Delta_0 = -0.4*t*4
Delta_1 = 0.2*t*4
Lambda = 0.5*t
Phi = np.pi/6
t_J = 1

params = dict(t=t, mu=mu, Delta_0=Delta_0,
              Delta_1=Delta_1, Lambda=Lambda,
              t_J=t_J, Phi=np.round(Phi, 2))

eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian_ZKM_S(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta_0=Delta_0, Delta_1=Delta_1, Lambda=Lambda, t_J=t_J, Phi=Phi))
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
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.text(5,25, rf'$t_J={params["t_J"]}; \Phi={params["Phi"]}$')
#plt.plot(probability_density[10,:,0])
plt.tight_layout()
#%% Energies
ax2 = fig.add_axes([0.3, 0.3, 0.25, 0.25])
ax2.scatter(np.arange(0, len(eigenvalues), 1), eigenvalues)
ax2.set_xlim([2*(L_x*L_y-5), 2*(L_x*L_y+5)])
ax2.set_ylim([-0.05, 0.05])