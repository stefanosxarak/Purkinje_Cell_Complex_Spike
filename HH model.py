from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sympy import limit,symbols,E

# Nelson, M.E. (2004) Electrophysiological Models In: Databasing the Brain: From Data to Knowledge. (S. Koslow and S. Subramaniam, eds.) Wiley, New York.

class HodgkinHuxley():
    # Hodgkin - Huxley model
    def default_pars(self):
        ### ALL UNITS NEED TO BE IN S.I. ###
        # Paremeters are passed from the command line when the program is executed
       
        # print("Please Enter all the values in S.I units")
        # self.g_na = float(input("Enter the value of gNa: "))                        # Average sodoum channel conductance per unit area
        # self.g_k  = float(input("Enter the value of gK: "))                         # Average potassium channel conductance per unit area
        # self.g_l  = float(input("Enter the value of gl: "))                         # Average leak channel conductance per unit area
        # self.C_m  = float(input("Enter the value of membrane capacitance in μ units: "))   # Membrane capacitance per unit area
        # self.v   = float(input("Enter the value of the membrane potential v: "))   # vis the membrane potential
        # self.vna  = float(input("Enter the value of vNa: "))                        # Potassium potential
        # self.vk   = float(input("Enter the value of vK: "))                         # Sodium potential
        # self.vl   = float(input("Enter the value of vl: "))                         # Leak potential
        # self.tmin = float(input("Enter the start time : "))
        # self.tmax = float(input("Enter the end time: "))
                
        self.g_na = 120.               
        self.g_k  = 36.0                         
        self.g_l  = 0.3                   
        self.c_m  = 1.0
        self.v   = -65.
        self.vna  = 115.                        
        self.vk   = -12.                        
        self.vl   = 10.613                         
        self.tmin = 0.
        self.tmax = 35.0  
        self.i_inj = 50.

        self.t  = np.linspace(self.tmin, self.tmax, 100) # Total duration of simulation [ms]


    # α and β are the forward and backwards rate, respectively
    # These are the original HH equations for α,β where the constants vary in order to fit adequately the data
    # α(v) = (A+B*v)/C + H*exp((v+D) / F) where A,B,C,D,F,H are constants to be fit to the data
    
    # NOTE: for cases where 0/0 might occur then L'Hospital's rules must apply
    def alpha_n(self,v):
        x = symbols('x')
        A = 0.01 * (10 - v)
        B = (np.exp((10 - v) / 10) - 1)

        if A == 0 and B == 0 :
            return limit(0.01 * (10 - x) /(E**((10 - x) / 10) - 1), x,10)
        else:
            return A / B
    def beta_n(self,v):
        return 0.125 * np.exp(- v/ 80)

    def alpha_m(self,v):
        x = symbols('x')
        A = 0.1  * (25 - v)
        B = (np.exp((25 - v) / 10) - 1)

        if A == 0 and B == 0 :
            return limit(0.1  * (25 - x) / (E**((25 - x) / 10) - 1), x,10)
        else:
            return A / B
        
    def beta_m(self,v):
        return 4 * np.exp(- v/ 18)

    def alpha_h(self,v):
        return 0.07 * np.exp(- v/ 20)
    def beta_h(self,v):
        return 1 / (np.exp((30 -v) / 10) + 1)

    def n_inf(self,v):
        return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

    def m_inf(self,v):
        return self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))

    def h_inf(self,v):
        return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))

    # def id(self,t):
    #     # time varying current(2 injections)
    #     # no specific criteria for the time selected
    #     if 2.0 < t < 3.0:
    #         self.i_inj = 100.0
    #         return self.i_inj
    #     elif 10.0 < t < 11.0:
    #         self.i_inj = 50.0
    #         return self.i_inj
    #     return 0.0


    def derivatives(self,y,t):
        der = np.zeros(4)
        c_m = self.c_m
        
        v= y[0]
        self.n = y[1]
        self.m = y[2]
        self.h = y[3]

        gNa = self.g_na * np.power(self.m, 3.0) * self.h
        gK = self.g_k * np.power(self.n, 4.0)
        gL = self.g_l

        i_na = gNa * (v- self.vna )
        i_k = gK * (v- self.vk )
        i_l = gL * (v- self.vl )


        der[0] = (self.i_inj - i_na - i_k - i_l) / c_m   # dv/dt
        der[1] = (self.alpha_n(v) * (1 - self.n)) - (self.beta_n(v) * self.n)    # dn/dt
        der[2] = (self.alpha_m(v) * (1 - self.m)) - (self.beta_m(v) * self.m)    # dm/dt
        der[3] = (self.alpha_h(v) * (1 - self.h)) - (self.beta_h(v) * self.h)    # dh/dt
        
        return der

    def Main(self):
        self.default_pars()

        v= self.v
        t = self.t
        y = np.array([v, self.n_inf(v), self.m_inf(v), self.h_inf(v)], dtype= 'float64')

        sol = odeint(self.derivatives, y, t)    # Solve ODE

        ax = plt.subplot()
        ax.plot(t, sol[:, 0])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Membrane potential (v)')
        ax.set_title('Neuron potential')
        plt.grid()
        plt.show()

        ax = plt.subplot()
        ax.plot(t, sol[:,1], 'r', label='n')
        ax.plot(t, sol[:,2], 'g', label='m')
        ax.plot(t, sol[:,3], 'b', label='h')
        ax.set_ylabel('Gating value')
        ax.set_xlabel('Time (s)')
        ax.set_title('Potassium and Sodium channels')
        plt.legend()
        plt.show()

        # ax = plt.subplot()
        # ax.plot(self.i_inj, freq)
        # ax.set_ylabel("Input Current(A)")
        # ax.set_xlabel("Firing rate(spikes/s)")
        # ax.set_title('F-I Curve')
        # plt.legend()
        # plt.show()

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()