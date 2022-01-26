from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Hodgkin - Huxley model
def default_pars():
    pars = {}
    
    ### typical neuron parameters ###
    ### ALL UNITS NEED TO BE IN S.I. ###
    pars['g_na']  = 120.0*milli   # Average sodoum channel conductance per unit area
    pars['g_k']   = 36.0*milli    # Average potassium channel conductance per unit area
    pars['g_l']   = 0.3*milli     # Average leak channel conductance per unit area

    pars['C_m']   = 0.000001     # Membrane capacitance per unit area

    pars['V']     = 0.0*milli     # V is the membrane potential
    pars['Vk']    = -12.0*milli   # Potassium potential
    pars['Vna']   = 115.0*milli   # Sodium potential
    pars['Vl']    = 10.613*milli  # Leak potential

    ### simulation parameters ###
    pars['tmin'] = 0
    pars['tmax'] = 50
    pars['T']    = np.linspace(pars['tmin'], pars['tmax'], 10000) # Total duration of simulation [ms]
    
            
    return pars

# α and β are the forward and backwards rate, respectively
def alpha_n(V):
    return 0.01 * milli * (10 * milli - V)/ (np.exp((10 * milli - V) / (10 * milli)) - 1*milli)
def beta_n(V):
    return 0.125 * milli * np.exp(- V / (80 * milli))

def alpha_m(V):
    return 0.1 * milli * (25 * milli - V)/ (np.exp((25 * milli - V) / (10 * milli)) - 1*milli)
def beta_m(V):
    return 4 * milli  * np.exp(- V / (18 * milli))

def alpha_h(V):
    return 0.07 * milli * np.exp(- V / (20 * milli))
def beta_h(V):
    return 1*milli / (np.exp((30 * milli -V) / (10 * milli)) + 1*milli)

def n_inf(V):
    return alpha_n(V) / (alpha_n(V) + beta_n(V))

def m_inf(V):
    return alpha_m(V) / (alpha_m(V) + beta_m(V))

def h_inf(V):
    return alpha_h(V) / (alpha_h(V) + beta_h(V))


def derivatives(y,t0):
    der = np.zeros(4)
    pars = default_pars()
    C_m = pars['C_m']
    
    V = y[0]
    n = y[1]
    m = y[2]
    h = y[3]

    GNa = (pars['g_na'] / C_m) * np.power(m, 3.0) * h
    GK = (pars['g_k'] / C_m) * np.power(n, 4.0)
    GL = pars['g_l'] / C_m

    I_na = GNa * (pars['Vna'] - V)
    I_k = GK * (pars['Vk'] - V)
    I_l = GL * (pars['Vl'] - V)


    der[0] = I_na/C_m + I_k/C_m + I_l/C_m    # dv/dt
    der[1] = (alpha_n(V) * (1 - n)) - (beta_n(V) * n)    # dn/dt
    der[2] = (alpha_m(V) * (1 - m)) - (beta_m(V) * m)    # dm/dt
    der[3] = (alpha_h(V) * (1 - h)) - (beta_h(V) * h)    # dh/dt

    return der

pars = default_pars()

V = pars['V']

Y = np.array([V, n_inf(V), m_inf(V), h_inf(V)])

sol = odeint(derivatives, Y, pars['T'])    # Solve ODE

# Neuron potential
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(pars['T'], sol[:, 0])
ax.set_xlabel('Time (s)')
ax.set_ylabel('V (V)')
ax.set_title('Neuron potential with two spikes')
plt.grid()
plt.show()