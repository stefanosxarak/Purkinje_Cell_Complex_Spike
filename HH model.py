from Units import *
import numpy as np
import matplotlib.pyplot as plt

# Implement Hodgkin - Huxley model

# α and β are the forward and backwards rate
def alpha_n(V):

def beta_n(V):

def alpha_m(V):

def beta_m(V):

def alpha_h(V):

def beta_h(V):

def n_inf(V):
    return alpha_n(V) / (alpha_n(V) + beta_n(V))

# m and h are gating variables that vary between 1 and 0

def m_inf(V):
    return alpha_m(V) / (alpha_m(V) + beta_m(V))

def h_inf(V):
    return alpha_h(V) / (alpha_h(V) + beta_h(V))