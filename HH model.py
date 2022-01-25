from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Hodgkin - Huxley model

# V_m is the membrane potential
# g_k and g_na are the potassium and sodium conductances per unit area
# g_l and V_l are the leak conductance per unit area and leak reversal potential
I_na = g_na * (V_m - V)
I_k = g_k * (V_m - V)
I_l = g_l * (V_m - V)

# α and β are the forward and backwards rate
def alpha_n(V):
    return 0.01 * milli * (10 * milli - V)/ (np.exp((10 * milli - V) / (10 * milli)) - 1)
def beta_n(V):
    return 0.125 * milli * np.exp(- V / (80 * milli))

def alpha_m(V):
    return 0.1 * milli * (25 * milli - V)/ (np.exp((25 * milli - V) / (10 * milli)) - 1)
def beta_m(V):
    return 4 * milli  * np.exp(- V / (18 * milli))

def alpha_h(V):
    return 0.07 * milli * np.exp(- V / (20 * milli))
def beta_h(V):
    return 1 / (np.exp((30 * milli -V) / (10 * milli)) + 1)

def n_inf(V):
    return alpha_n(V) / (alpha_n(V) + beta_n(V))

def m_inf(V):
    return alpha_m(V) / (alpha_m(V) + beta_m(V))

def h_inf(V):
    return alpha_h(V) / (alpha_h(V) + beta_h(V))

def derivatives():
    dn/dt = (alpha_n(V) * (1 - n)) - (beta_n(V) * n)
    dm/dt = (alpha_m(V) * (1 - m)) - (beta_m(V) * m)
    dh/dt = (alpha_h(V) * (1 - h)) - (beta_h(V) * h)