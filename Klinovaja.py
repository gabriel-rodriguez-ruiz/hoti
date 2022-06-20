# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:11:08 2022

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt

def spectrum(k, Delta, Gamma, alpha, mu=0):
    r""" Returns a list with the four energies of the bulk spectrum.
    
    .. math::
        E^2 = (k^2-\mu)^2 + \Delta^2 + \Gamma^2 +
        \alpha^2 k^2 \pm 2\sqrt{(k^2-\mu)^2(\Gamma^2+\alpha^2k^2) +
                              \Delta^2\Gamma^2}
        
        \hbar^2/2m = 1
    """
    m = 1
    E_k = k**2/(2*m)
    E_square_plus = ((E_k-mu)**2 + Delta**2 + Gamma**2 + (alpha*k)**2 +
                2*np.sqrt((E_k-mu)**2*(Gamma**2 + (alpha*k)**2)+
                          Delta**2*Gamma**2)
                )
    E_square_minus = ((E_k-mu)**2 + Delta**2 + Gamma**2 + (alpha*k)**2 -
                2*np.sqrt((E_k-mu)**2*(Gamma**2 + (alpha*k)**2)+
                          Delta**2*Gamma**2)
                )
    return [np.sqrt(E_square_plus), np.sqrt(E_square_minus),
            -np.sqrt(E_square_plus), -np.sqrt(E_square_minus)]

alpha = 1
Gamma = 0.2
Delta = 0.9*Gamma  #0.2
k_SO = alpha
#k = np.linspace(-np.pi, np.pi, 1000)
k = np.linspace(-3*k_SO, 3*k_SO, 1000)

plt.figure("Spectrum", clear=True)
for i in range(4):
    plt.plot(k, [spectrum(k, Delta, Gamma, alpha, mu=0)[i] for k in k])
plt.ylim([-2*k_SO, 2*k_SO])
