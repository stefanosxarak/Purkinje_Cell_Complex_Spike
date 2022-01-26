from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Hodgkin - Huxley model
def default_pars():
    pars = {}
    
    ### typical neuron parameters ###
    pars['g_na']  = -55.  # Average sodoum channel conductance per unit area
    pars['g_k']   = -75.  # Average potassium channel conductance per unit area
    pars['g_l']   = 10.   # Average leak channel conductance per unit area

    pars['n']     = 10.   # steady state values
    pars['m']     = -65.  # steady state values
    pars['h']     = -75.  # steady state values

    pars['V']     = 2.    # V is the membrane potential

    pars['C_m']   = 2.    # Membrane capacitance per unit area

    pars['Vk']    = 2.    # Potassium potential
    pars['Vna']   = 2.    # Sodium potential
    pars['Vl']    = 2.    # Leak potential

    ### simulation parameters ###
    pars['tmin'] = 0
    pars['tmax'] = 0
    pars['T']    = np.linspace(pars['tmin'], pars['tmax'], 0) # Total duration of simulation [ms]
    pars['dt']   = 0.1  # Simulation time step [ms]
    
    
    pars['range_t'] = np.arange(0, pars['T'], pars['dt']) # Vector of discretized time points [ms]
        
    return pars

# α and β are the forward and backwards rate, respectively
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


def derivatives(pars):
    der = np.zeros(4)
    C_m = pars['C_m']
    V = pars['V']

    n = pars['n']
    m = pars['m']
    h = pars['h']


    I_na = pars['g_na'] * (pars['Vna'] - V)
    I_k = pars['g_k'] * (pars['Vk'] - V)
    I_l = pars['g_l'] * (pars['Vl'] - V)


    der[0] = I_na/C_m + I_k/C_m + I_l/C_m    # dv/dt
    der[1] = (alpha_n(V) * (1 - n)) - (beta_n(V) * n)    # dn/dt
    der[2] = (alpha_m(V) * (1 - m)) - (beta_m(V) * m)    # dm/dt
    der[3] = (alpha_h(V) * (1 - h)) - (beta_h(V) * h)    # dh/dt

    return der

pars = default_pars()

y = np.array([pars['V'], n_inf(), m_inf(), h_inf()])

sol = odeint(derivatives(pars), y, t)    # Solve ODE