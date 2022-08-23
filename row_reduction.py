# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:12:29 2022

@author: gabri
"""
import sympy as sym

A = sym.Matrix([[1,0,1,0,1,0,1,0], 
                  [0,1,0,1,0,1,0,1],
                  [1,0,-1,0,1j,0,-1j,0],
                  [0,-1,0,1,0,-1j,0,1j]])
B = A.rref(pivots=False)