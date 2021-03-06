#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:24:41 2022

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_A1u_semi_infinite_with_Zeeman
from functions import spectrum

L_x = 50
t = 1
Delta = 1
mu = -2    # topological phase if 0<mu<4
Delta_Z = 0.2   #0.2
theta = 0
phi = np.pi/2
k = np.linspace(0, np.pi, 200)

params = dict(t=t, mu=mu, Delta=Delta, L_x=L_x,
              Delta_Z=Delta_Z, theta=np.round(theta,2), phi=np.round(phi,2))

spectrum_A1u = spectrum(Hamiltonian_A1u_semi_infinite_with_Zeeman, k, **params)
fig, ax = plt.subplots(num="Espectro", clear=True)
ax.plot(k, spectrum_A1u, linewidth=0.1, color="m")
ax.set_xlabel(r"$k_y$")
ax.set_ylabel("E")
#ax.set_title(f'$\Delta_Z={params["Delta_Z"]}$')
ax.text(0, 3, rf'$\Delta_Z={params["Delta_Z"]}; \theta={params["theta"]}; \varphi={params["phi"]}$')
plt.tight_layout()