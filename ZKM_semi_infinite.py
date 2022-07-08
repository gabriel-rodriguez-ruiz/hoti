#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:53:35 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_ZKM_semi_infinite
from functions import spectrum

L_x = 200
t = 1
Delta_0 = -0.4*t
Delta_1 = 0.2*t
mu = -2*t    # topological phase if -3<mu<-1
Delta_Z = 0   #0.2
theta = 0
k = np.linspace(0, np.pi, 200)
Lambda = 0.5*t

params = dict(t=t, mu=mu, Delta_0=Delta_0, Delta_1=Delta_1, L_x=L_x,
             Lambda=Lambda)

spectrum_A1u = spectrum(Hamiltonian_ZKM_semi_infinite, k, **params)
fig, ax = plt.subplots(num="Espectro", clear=True)
ax.plot(k, spectrum_A1u, linewidth=0.1, color="m")
ax.set_xlabel(r"$k_y$")
ax.set_ylabel("E")
ax.set_title(f"{params}")
