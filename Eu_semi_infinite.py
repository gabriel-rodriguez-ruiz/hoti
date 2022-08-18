#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:24:41 2022

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_Eu_semi_infinite_with_Zeeman
from functions import spectrum

L_x = 50
t = 1
Delta = 1
mu = -2    # topological phase if 0<mu<4
Delta_Z = 0.1   #0.2
theta = np.pi/2
phi = np.pi/2
k = np.linspace(0, np.pi, 200)

params = dict(t=t, mu=mu, Delta=Delta, L_x=L_x,
              Delta_Z=Delta_Z, theta=theta, phi=phi)

spectrum_Eu = spectrum(Hamiltonian_Eu_semi_infinite_with_Zeeman, k, **params)
fig, ax = plt.subplots(num="Espectro", clear=True)
ax.plot(k, spectrum_Eu, linewidth=0.1, color="m")
ax.set_xlabel(r"$k_y$")
ax.set_ylabel("E")
ax.set_title(f"{params}")
