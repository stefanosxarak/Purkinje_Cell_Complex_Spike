from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sympy import *

class HodgkinHuxley():
    # Hodgkin - Huxley model
    def default_pars(self):
        ### ALL UNITS NEED TO BE IN S.I. ###
        # Paremeters are passed from the command line when the program is executed
       
        print("Please Enter all the values in S.I units")
        self.g_na = float(input("Enter the value of gNa: "))                        # Average sodoum channel conductance per unit area
        self.g_k  = float(input("Enter the value of gK: "))                         # Average potassium channel conductance per unit area
        self.g_l  = float(input("Enter the value of gl: "))                         # Average leak channel conductance per unit area
        self.C_m  = float(input("Enter the value of membrane capacitance in μ units: "))   # Membrane capacitance per unit area
        self.V    = float(input("Enter the value of the membrane potential V: "))   # V is the membrane potential
        self.Vna  = float(input("Enter the value of VNa: "))                        # Potassium potential
        self.Vk   = float(input("Enter the value of VK: "))                         # Sodium potential
        self.Vl   = float(input("Enter the value of Vl: "))                         # Leak potential
        self.tmin = float(input("Enter the start time : "))
        self.tmax = float(input("Enter the end time: "))
                
        # self.g_na = 120               
        # self.g_k  = 36                         
        # self.g_l  = 0.3                   
        # self.C_m  = 1
        # self.V    = -65
        # self.Vna  = 115                        
        # self.Vk   = -12                        
        # self.Vl   = 10.613                         
        # self.tmin = 0
        # self.tmax = 22  

        self.T  = np.linspace(self.tmin, self.tmax, 100) # Total duration of simulation [ms]


    # α and β are the forward and backwards rate, respectively
    # These are the original HH equations for α,β where the constants vary in order to fit adequately the data
    # α(v) = (A+B*V)/C + H*exp((V+D) / F) where A,B,C,D,F,H are constants to be fit to the data
    # NOTE: for cases where 0/0 might occur then L'Hospital's rules must apply
    def alpha_n(self,V):
        x = symbols('x')
        A = 0.01 * (10 - V)
        B = (np.exp((10 - V) / 10) - 1)

        if A == 0 and B == 0 :
            return limit(0.01 * (10 - x) /(E**((10 - x) / 10) - 1), x,10)
        else:
            return A / B
    def beta_n(self,V):
        return 0.125 * np.exp(- V / 80)

    def alpha_m(self,V):
        x = symbols('x')
        A = 0.1  * (25 - V)
        B = (np.exp((25 - V) / 10) - 1)

        if A == 0 and B == 0 :
            return limit(0.1  * (25 - x) / (E**((25 - x) / 10) - 1), x,10)
        else:
            return A / B
        
    def beta_m(self,V):
        return 4 * np.exp(- V / 18)

    def alpha_h(self,V):
        return 0.07 * np.exp(- V / 20)
    def beta_h(self,V):
        return 1 / (np.exp((30 -V) / 10) + 1)

    def n_inf(self,V):
        return self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))

    def m_inf(self,V):
        return self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))

    def h_inf(self,V):
        return self.alpha_h(V) / (self.alpha_h(V) + self.beta_h(V))

    def Id(self,t):
        # time varying current(2 injections)
        # no specific criteria for the time selected
        if 2.0 < t < 3.0:
            self.I_inj = 100.0
            return self.I_inj
        elif 10.0 < t < 11.0:
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

        GNa = self.g_na * np.power(self.m, 3.0) * self.h
        GK = self.g_k * np.power(self.n, 4.0)
        GL = self.g_l

        self.I_na = GNa * (V - self.Vna )
        self.I_k = GK * (V - self.Vk )
        self.I_l = GL * (V - self.Vl )


        der[0] = (self.Id(t0) - self.I_na - self.I_k - self.I_l) / C_m   # dv/dt
        der[1] = (self.alpha_n(V) * (1 - self.n)) - (self.beta_n(V) * self.n)    # dn/dt
        der[2] = (self.alpha_m(V) * (1 - self.m)) - (self.beta_m(V) * self.m)    # dm/dt
        der[3] = (self.alpha_h(V) * (1 - self.h)) - (self.beta_h(V) * self.h)    # dh/dt
        
        return der

    def Main(self):
        self.default_pars()

        V = self.V
        t = self.T
        Y = np.array([V, self.n_inf(V), self.m_inf(V), self.h_inf(V)], dtype= 'float64')

        sol = odeint(self.derivatives, Y, t)    # Solve ODE

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(t, sol[:, 0])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Membrane potential (V)')
        ax.set_title('Neuron potential')
        plt.grid()
        plt.show()

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