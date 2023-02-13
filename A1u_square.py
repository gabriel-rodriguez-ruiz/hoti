#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:48:52 2022

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_A1u
from functions import probability_density

L_x = 30
L_y = 30
t = 1
Delta = 1
mu = -2

Delta_Z = 0.2   #0.2

H = Hamiltonian_A1u(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta=Delta)
probability_density_2D, eigenvalues, eigenvectors = probability_density(H, L_x, L_y, index=2)

#%%
fig, ax = plt.subplots(num="Zeeman", clear=True)
image = ax.imshow(probability_density_2D, cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
#plt.plot(probability_density[10,:,0])
plt.tight_layout()

#%% Energies
"""
ax2 = fig.add_axes([0.3, 0.3, 0.25, 0.25])
ax2.scatter(np.arange(0, 4*L_x*L_y, 1), eigenvalues)
ax2.set_xlim([2*(L_x*L_y-5), 2*(L_x*L_y+5)])
ax2.set_ylim([-0.1, 0.1])
"""

#%% Spin determination
from functions import mean_spin, mean_spin_xy, get_components

index = 2 #which zero mode
zero_modes = eigenvectors[:, 2*(L_x*L_y-1):2*(L_x*L_y+1)]      #4 (2) modes with zero energy (with Zeeman)
creation_up, creation_down, destruction_down, destruction_up = get_components(zero_modes[:,index], L_x, L_y)
zero_plus_state = np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2) #positive energy eigenvector splitted in components
corner_state = zero_plus_state[L_x-2, L_y//2, :].reshape(4,1)  #positive energy point state localized at the junction
corner_state_normalized = corner_state/np.linalg.norm(corner_state[:2]) #normalization with only particle part
zero_plus_state_normalized = zero_plus_state/np.linalg.norm(zero_plus_state[:,:,:2], axis=2, keepdims=True)

# Spin mean value
spin_mean_value = mean_spin(corner_state)

spin = mean_spin_xy(zero_plus_state)
# fig, ax = plt.subplots()
# image = ax.imshow(spin[:,:,2].T, cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
# plt.colorbar(image)
#image.set_clim(np.min(spin[:,:,1].T), np.max(spin[:,:,1].T))

# Meshgrid
x, y = np.meshgrid(np.linspace(0, L_x-1, L_x), 
                    #np.linspace(L_y-1, 0, L_y))
                    np.linspace(0, L_y-1, L_y))


  
# Directional vectors
u = spin[:, :, 0]   #x component
v = spin[:, :, 1]   #y component

# Plotting Vector Field with QUIVER
ax.quiver(x, y, u, v, color='r')
ax.set_title('Spin Field in the plane')

#%% Spin in z
fig, ax = plt.subplots()
image = ax.imshow(spin[:,:,2], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
