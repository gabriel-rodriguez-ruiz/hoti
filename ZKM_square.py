#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:31:17 2022

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_ZKM, Zeeman
from functions import probability_density

L_x = 20
L_y = 40
t = 1
mu = -2*t   # mu=t*Delta_0/Delta_1
Delta_Z = 0  # 0.2
theta = np.pi/2
phi = 0
Delta_0 = -0.4*t
Delta_1 = 0.2*t
Lambda = 0.5*t
index = 1

#Aligia
# Delta_0 = 4  #-0.4*t
# Delta_1 = 2.2  #0.2*t
# Lambda = 7

H = (Hamiltonian_ZKM(t=t, mu=mu, L_x=L_x, L_y=L_y, Delta_0=Delta_0, Delta_1=Delta_1, Lambda=Lambda) +
        Zeeman(theta=theta, Delta_Z=Delta_Z, L_x=L_x, L_y=L_y, phi=phi))
probability_density_2D, eigenvalues, eigenvectors = probability_density(H, L_x, L_y, index=index)

#%%
fig, ax = plt.subplots()
image = ax.imshow(probability_density_2D, cmap="Blues", origin="lower")
#ax.set_title(f"{params}")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()
#%% Energies
ax2 = fig.add_axes([0.3, 0.3, 0.25, 0.25])
ax2.scatter(np.arange(0, len(eigenvalues), 1), eigenvalues)
ax2.set_xlim([2*(L_x*L_y-5), 2*(L_x*L_y+5)])
ax2.set_ylim([-0.1, 0.1])

#%% Spin determination
from functions import mean_spin_xy, get_components

zero_modes = eigenvectors[:, 2*(L_x*L_y-1):2*(L_x*L_y+1)]      #4 (2) modes with zero energy (with Zeeman)
creation_up, creation_down, destruction_down, destruction_up = get_components(zero_modes[:,index], L_x, L_y)


zero_plus_state = np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2) #positive energy eigenvector splitted in components
# corner_state = zero_plus_state[L_x-2, L_y//2, :].reshape(4,1)  #positive energy point state localized at the junction
# corner_state_normalized = corner_state/np.linalg.norm(corner_state[:2]) #normalization with only particle part
#zero_plus_state_normalized = zero_plus_state/np.linalg.norm(zero_plus_state[:,:,:2], axis=2, keepdims=True)

# Spin mean value
# spin_mean_value = mean_spin(corner_state_normalized)

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
ax.quiver(x, y, u, v, color='r',angles='uv', scale_units='xy', scale=1)
ax.set_title('Spin Field in the plane')

#%% Spin in z
fig, ax = plt.subplots()
ax.set_title("Spin in z")
image = ax.imshow(spin[:,:,2], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
