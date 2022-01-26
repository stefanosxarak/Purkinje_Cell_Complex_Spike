from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class HodgkinHuxley():
    # Hodgkin - Huxley model
    def default_pars(self):
        pars = {}
        
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
        pars['tmax'] = 50*milli
        pars['T']    = np.linspace(pars['tmin'], pars['tmax'], 10) # Total duration of simulation [ms]
        
        return pars

    # α and β are the forward and backwards rate, respectively
    def alpha_n(self,V):
        return 0.01 * milli * (10 * milli - V)/ (np.exp((10 * milli - V) / (10 * milli)) - 1*milli)
    def beta_n(self,V):
        return 0.125 * milli * np.exp(- V / (80 * milli))

    def alpha_m(self,V):
        return 0.1 * milli * (25 * milli - V)/ (np.exp((25 * milli - V) / (10 * milli)) - 1*milli)
    def beta_m(self,V):
        return 4 * milli  * np.exp(- V / (18 * milli))

    def alpha_h(self,V):
        return 0.07 * milli * np.exp(- V / (20 * milli))
    def beta_h(self,V):
        return 1*milli / (np.exp((30 * milli -V) / (10 * milli)) + 1*milli)

    def n_inf(self,V):
        return self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))

    def m_inf(self,V):
        return self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))

    def h_inf(self,V):
        return self.alpha_h(V) / (self.alpha_h(V) + self.beta_h(V))


    def derivatives(self,y,t0,pars):
        der = np.zeros(4)
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
        der[1] = (self.alpha_n(V) * (1 - n)) - (self.beta_n(V) * n)    # dn/dt
        der[2] = (self.alpha_m(V) * (1 - m)) - (self.beta_m(V) * m)    # dm/dt
        der[3] = (self.alpha_h(V) * (1 - h)) - (self.beta_h(V) * h)    # dh/dt

        return der

    def Main(self):
        pars = self.default_pars()

        V = pars['V']
        Y = np.array([V, self.n_inf(V), self.m_inf(V), self.h_inf(V)])

        sol = odeint(self.derivatives, Y, pars['T'], args=(pars,))    # Solve ODE

        # Neuron potential
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(pars['T'], sol[:, 0])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('V (V)')
        ax.set_title('Neuron potential')
        plt.grid()
        plt.show()

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()