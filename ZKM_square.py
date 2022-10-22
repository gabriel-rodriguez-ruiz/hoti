#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:31:17 2022

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_ZKM, Zeeman

L_x = 30
L_y = 30
t = 1
mu = -2*t   # mu=t*Delta_0/Delta_1
Delta_Z = 0  # 0.2
theta = np.pi/2
phi = np.pi/4
Delta_0 = -0.4*t*4
Delta_1 = 0.2*t*4
Lambda = 0.5*t

params = dict(t=t, mu=mu, Delta_0=Delta_0,
              Delta_Z=Delta_Z, theta=np.round(theta,2),
              Delta_1=Delta_1, Lambda=Lambda,
              phi=np.round(phi,2))

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
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.text(5,25, rf'$\Delta_Z={params["Delta_Z"]}; \theta={params["theta"]}; \varphi={params["phi"]}$')
#plt.plot(probability_density[10,:,0])
plt.tight_layout()
#%% Energies
ax2 = fig.add_axes([0.3, 0.3, 0.25, 0.25])
ax2.scatter(np.arange(0, len(eigenvalues), 1), eigenvalues)
ax2.set_xlim([2*(L_x*L_y-5), 2*(L_x*L_y+5)])
ax2.set_ylim([-0.1, 0.1])

#%% Spin determination
from functions import mean_spin, mean_spin_xy

zero_plus_state = np.stack((destruction_up[2], destruction_down[2], creation_down[2], creation_up[2]), axis=2) #positive energy eigenvector splitted in components
corner_state = zero_plus_state[L_x-2, L_y//2, :].reshape(4,1)  #positive energy point state localized at the junction
corner_state_normalized = corner_state/np.linalg.norm(corner_state[:2]) #normalization with only particle part
zero_plus_state_normalized = zero_plus_state/np.linalg.norm(zero_plus_state[:,:,:2], axis=2, keepdims=True)

# Spin mean value
spin_mean_value = mean_spin(corner_state_normalized)

spin = mean_spin_xy(zero_plus_state_normalized)
# fig, ax = plt.subplots()
# image = ax.imshow(spin[:,:,2].T, cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
# plt.colorbar(image)
#image.set_clim(np.min(spin[:,:,1].T), np.max(spin[:,:,1].T))

# Meshgrid
x, y = np.meshgrid(np.linspace(0, L_x-1, L_x), 
                    np.linspace(L_y-1, 0, L_y))
                    #np.linspace(0, L_y-1, L_y))


  
# Directional vectors
u = spin[:, :, 0].T   #x component
v = spin[:, :, 1].T   #y component

# Plotting Vector Field with QUIVER
ax.quiver(x, y, u, v, color='r')
ax.set_title('Spin Field in the plane')

#%% Spin in z
fig, ax = plt.subplots()
ax.set_title("Spin in z for ZKM model")
image = ax.imshow(spin[:,:,2].T, cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
plt.tight_layout()