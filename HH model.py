from scipy.misc import derivative
from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Nelson, M.E. (2004) Electrophysiological Models In: Databasing the Brain: From Data to Knowledge. (S. Koslow and S. Subramaniam, eds.) Wiley, New York.

class HodgkinHuxley():
    # Hodgkin - Huxley model
    def default_pars(self):
        ### ALL UNITS NEED TO BE IN S.I. ###
        # Paremeters are passed from the command line when the program is executed
       
        # print("Please Enter all the values in milli units (graphs will automatically convert to S.I. units)")
        # self.g_na = float(input("Enter the value of gNa: "))                        # Average sodoum channel conductance per unit area
        # self.g_k  = float(input("Enter the value of gK: "))                         # Average potassium channel conductance per unit area
        # self.g_l  = float(input("Enter the value of gl: "))                         # Average leak channel conductance per unit area
        # self.C_m  = float(input("Enter the value of membrane capacitance in μ (micro) units: "))   # Membrane capacitance per unit area
        # self.v   = float(input("Enter the value of the membrane potential v: "))   # vis the membrane potential
        # self.vna  = float(input("Enter the value of vNa: "))                        # Potassium potential
        # self.vk   = float(input("Enter the value of vK: "))                         # Sodium potential
        # self.vl   = float(input("Enter the value of vl: "))                         # Leak potential
        # self.tmin = float(input("Enter the start time : "))
        # self.tmax = float(input("Enter the end time: "))
                
        self.g_na = 120.               
        self.g_k  = 36.                       
        self.g_l  = 0.3                   
        self.c_m  = 1.
        self.v   = 0.
        self.vna  = 115.                        
        self.vk   = -12.                        
        self.vl   = 10.613                         
        self.tmin = 0.
        self.tmax = 35.  
        self.i_inj = 50.
        self.t  = np.linspace(self.tmin, self.tmax, 1000) # Total duration of simulation [ms]


    # α and β are the forward and backwards rate, respectively
    # These are the original HH equations for α,β where the constants vary in order to fit adequately the data
    # α(v) = (A+B*v)/C + H*exp((v+D) / F) where A,B,C,D,F,H are constants to be fit to the data
    
    # NOTE: for cases where 0/0 might occur then L'Hospital's rules must apply
    def alpha_n(self,v):
        nom = 0.01 * (10 - v)
        denom = (np.exp((10 - v) / 10) - 1)

        if nom == 0 and denom == 0 :
            return 0.1
        else:
            return nom / denom
    def beta_n(self,v):
        return 0.125 * np.exp(- v/ 80.)

    def alpha_m(self,v):
        nom = 0.1  * (25. - v)
        denom = (np.exp((25. - v) / 10.) - 1.)

        if nom == 0 and denom == 0 :
            return 1.5/(-1 + np.exp(3./2.))
        else:
            return nom / denom
        
    def beta_m(self,v):
        return 4. * np.exp(- v/ 18.)

    def alpha_h(self,v):
        return 0.07 * np.exp(- v/ 20.)
    def beta_h(self,v):
        return 1. / (np.exp((30. -v) / 10.) + 1.)

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

    def frequency(self,v):
        v_thresh = -6.
        v_reset = 0.
        freq = []
        rec_spikes = 0
        iinj = np.linspace(0, 50., 1000)
        for i in range(0,len(self.t)-1):
            if v[i] >= v_thresh:    # if voltage over threshold
                rec_spikes +=1      # record spike event
                v[i+1] = v_reset    # reset voltage

        for n in range(len(iinj)):
            freq.append(rec_spikes/self.tmax)
        print(freq)
        return freq

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
        iinj = np.linspace(0, 50., 1000)

        firing_rate = self.frequency(sol[:,1]*milli)


        ax = plt.subplot()
        ax.plot(t*milli, sol[:, 0]*milli)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Membrane potential (v)')
        ax.set_title('Neuron potential')
        plt.grid()
        plt.show()

        ax = plt.subplot()
        ax.plot(t*milli, sol[:,1], 'r', label='n')
        ax.plot(t*milli, sol[:,2], 'g', label='m')
        ax.plot(t*milli, sol[:,3], 'b', label='h')
        ax.set_ylabel('Gating value')
        ax.set_xlabel('Time (s)')
        ax.set_title('Potassium and Sodium channels')
        plt.legend()
        plt.show()

        ax = plt.subplot()
        ax.plot(iinj*milli, firing_rate)
        ax.set_xlabel("Input Current(A)")
        ax.set_ylabel("Firing rate(spikes/s)")
        ax.set_title('F-I Curve')
        plt.show()

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()