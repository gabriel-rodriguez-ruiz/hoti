#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:55:54 2022

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_ZKM_semi_infinite
from functions import mean_spin

L_x = 200
t = 1
Delta_0 = -0.4*t
Delta_1 = 0.2*t
mu = -2*t    # topological phase if -3<mu<-1
k = 0.1*np.pi
Lambda = 0.5*t
index = 0

#Aligia
# t = 1
# mu = 2*t
# Delta_0 = 4*t
# Delta_1 = 2.2*t
# Lambda = 7*t

H = Hamiltonian_ZKM_semi_infinite(k, t, mu, L_x, Delta_0, Delta_1, Lambda)
eigenvalues, eigenvectors = np.linalg.eigh(H)
zero_modes = eigenvectors[:,(2*L_x-2):(2*L_x+2)]

right_minus = (zero_modes[:,0] + zero_modes[:,1])/np.sqrt(2)
left_minus = (zero_modes[:,0] - zero_modes[:,1])/np.sqrt(2)
right_plus = (zero_modes[:,2] + zero_modes[:,3])/np.sqrt(2)
left_plus = (zero_modes[:,2] - zero_modes[:,3])/np.sqrt(2)

destruction_up = right_plus[::4]
destruction_down = right_plus[1::4]
creation_down = right_plus[2::4]
creation_up = right_plus[3::4]

k_values = np.linspace(0, 0.5*np.pi)
theta_r_k = []
theta_l_k = []
phi_r_k = []
phi_l_k = []
delta_l_k = []
delta_r_k = []

for k_value in k_values:
    H = Hamiltonian_ZKM_semi_infinite(k_value, t, mu, L_x, Delta_0, Delta_1, Lambda)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    zero_modes = eigenvectors[:,(2*L_x-2):(2*L_x+2)]
    right_minus = (zero_modes[:,0] + zero_modes[:,1])/np.sqrt(2)
    left_minus = (zero_modes[:,0] - zero_modes[:,1])/np.sqrt(2)
    right_plus = (zero_modes[:,2] + zero_modes[:,3])/np.sqrt(2)
    left_plus = (zero_modes[:,2] - zero_modes[:,3])/np.sqrt(2)

    theta_l_k.append(2*np.arctan(np.abs(left_minus[1])/np.abs(left_minus[0])))
    theta_r_k.append(2*np.arctan(np.abs(right_minus[-3])/np.abs(right_minus[-4])))
    phi_l_k.append(np.angle(-left_minus[1]/left_minus[0]))
    phi_r_k.append(np.angle(-right_minus[-3]/right_minus[-4]))
    delta_l_k.append(-1/2*np.angle(left_plus[0]/left_minus[1]))
    delta_r_k.append(-1/2*np.angle(right_plus[-4]/right_minus[-3]))

probability_density = np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2

fig, ax = plt.subplots()
ax.plot(k_values, phi_l_k, ".", label=r"$\varphi_{l,k}$")
ax.plot(k_values, phi_r_k, ".", label=r"$\varphi_{r,k}$")
ax.plot(k_values, delta_l_k, ".", label=r"$\delta_{l,k}$")
ax.plot(k_values, delta_r_k, ".", label=r"$\delta_{r,k}$")
ax.plot(k_values, theta_l_k, "-", label=r"$\theta_k$")
ax.legend()
ax.text(0,3,f"$\phi_l=${phi_l_k[1].round(2)}")
ax.set_xlabel("k")

versor = [np.cos(phi_l_k[1]), np.sin(phi_l_k[1]),0]
ax.arrow(0,0,versor[0]/10, versor[1]/10)
spin = mean_spin((left_minus[0:4]/(np.linalg.norm(left_minus[0:4]))/np.sqrt(2)).reshape((4,1)))

spin = []
k_value = 0.1*np.pi
for i in range(L_x):
    spin.append(mean_spin(left_minus[4*i:4*(i+1)].reshape((4,1))))

spin_x = [spin[i][0] for i in range(L_x)]
spin_y = [spin[i][1] for i in range(L_x)]
spin_z = [spin[i][2] for i in range(L_x)]

x, y = np.meshgrid(1, np.linspace(0, L_x-1, L_x))


  
# Directional vectors
u = spin_x   #x component
v = spin_y   #y component

# Plotting Vector Field with QUIVER
fig, ax = plt.subplots()
ax.quiver(x, y, u, v, color='r',angles='uv')
ax.set_title('Spin Field in the line')
