from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class HodgkinHuxley():
    # Hodgkin - Huxley model

    def default_pars(self):
        ### ALL UNITS NEED TO BE IN S.I. ###
        # Paremeters are passed from the command line when the program is executed
        print("Please Enter all the values in S.I. units")
        self.g_na = float(input("Enter the value of gNa: "))                        # Average sodoum channel conductance per unit area
        self.g_k  = float(input("Enter the value of gK: "))                         # Average potassium channel conductance per unit area
        self.g_l  = float(input("Enter the value of gl: "))                         # Average leak channel conductance per unit area
        self.C_m  = float(input("Enter the value of membrane capacitance C_m: "))   # Membrane capacitance per unit area
        self.V    = float(input("Enter the value of the membrane potential V: "))   # V is the membrane potential
        self.Vna  = float(input("Enter the value of VNa: "))                        # Potassium potential
        self.Vk   = float(input("Enter the value of VK: "))                         # Sodium potential
        self.Vl   = float(input("Enter the value of Vl: "))                         # Leak potential
        self.tmin = float(input("Enter the start time : "))
        self.tmax = float(input("Enter the end time: "))
        

        # pars={}
        # pars['g_na']  = 120.0*milli  
        # pars['g_k']   = 36.0*milli    
        # pars['g_l']   = 0.3*milli    
        # pars['C_m']   = 0.000001     
        # pars['V']     = 0.0*milli     
        # pars['Vk']    = -12.0*milli   
        # pars['Vna']   = 115.0*milli   
        # pars['Vl']    = 10.613*milli  

        ### simulation parameters ###
        # pars['tmin'] = 0
        # pars['tmax'] = 30*milli

        self.T    = np.linspace(self.tmin, self.tmax, 100) # Total duration of simulation [ms]

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

    def Id(self,t):
        # time varying current
        # no specific criteria for the time selected
        if 0.0 < t < 2:
            self.I_inj = 150.0
            return self.I_inj
        elif 10.0 < t < 13.0:
            self.I_inj = 50.0
            return self.I_inj
        return 0.0


    def derivatives(self,y,t0):
        der = np.zeros(4)
        C_m = self.C_m
        
        V = y[0]
        self.n = y[1]
        self.m = y[2]
        self.h = y[3]

        GNa = (self.g_na / C_m) * np.power(self.m, 3.0) * self.h
        GK = (self.g_k / C_m) * np.power(self.n, 4.0)
        GL = self.g_l / C_m

        self.I_na = GNa * (self.Vna - V)
        self.I_k = GK * (self.Vk - V)
        self.I_l = GL * (self.Vl - V)


        der[0] = (self.Id(t0) / C_m) - self.I_na + self.I_k + self.I_l   # dv/dt
        der[1] = (self.alpha_n(V) * (1 - self.n)) - (self.beta_n(V) * self.n)    # dn/dt
        der[2] = (self.alpha_m(V) * (1 - self.m)) - (self.beta_m(V) * self.m)    # dm/dt
        der[3] = (self.alpha_h(V) * (1 - self.h)) - (self.beta_h(V) * self.h)    # dh/dt
        
        return der

    def Main(self):
        self.default_pars()

        V = self.V
        t = self.T
        Y = np.array([V, self.n_inf(V), self.m_inf(V), self.h_inf(V)])

        sol = odeint(self.derivatives, Y, t)    # Solve ODE

        # Neuron potential
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, sol[:, 0])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Membrane potential (V)')
        ax.set_title('Neuron potential')
        plt.grid()
        plt.show()

        # plt.subplot(4,1,2)
        # plt.plot(t, self.I_na, 'c', label='$I_{Na}$')
        # plt.plot(t, self.I_k, 'y', label='$I_{K}$')
        # plt.plot(t, self.I_l, 'm', label='$I_{L}$')
        # plt.ylabel('Current')
        # plt.legend()

        ax = plt.subplot()
        ax.plot(t, sol[:,1], 'r', label='n')
        ax.plot(t, sol[:,2], 'g', label='m')
        ax.plot(t, sol[:,3], 'b', label='h')
        ax.set_ylabel('Gating Value')
        ax.set_xlabel('Time (s)')
        plt.legend()

        plt.show()

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()