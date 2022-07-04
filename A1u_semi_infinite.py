#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:24:41 2022

@author: usuario
"""
import numpy as np
import matplotlib.pyplot as plt
from hamiltonians import Hamiltonian_A1u_semi_infinite
from functions import spectrum

L_x = 50
t = 1
Delta = 1
mu = -2    # topological phase if 0<mu<4t
k = np.linspace(0, np.pi)

params = dict(t=t, mu=mu, Delta=Delta, L_x=L_x)

spectrum_A1u = spectrum(Hamiltonian_A1u_semi_infinite, k, **params)
fig, ax = plt.subplots()
ax.plot(k, spectrum_A1u, linewidth=0.1, color="m")
ax.set_xlabel("k")
ax.set_xlabel("E")
