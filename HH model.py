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
        # self.g_na = float(input("Enter the value of gNa: "))                                   # Average sodoum channel conductance per unit area
        # self.g_k  = float(input("Enter the value of gK: "))                                   # Average potassium channel conductance per unit area
        # self.g_l  = float(input("Enter the value of gl: "))                                   # Average leak channel conductance per unit area
        # self.C_m  = float(input("Enter the value of membrane capacitance in μ (micro) units: "))   # Membrane capacitance per unit area
        # self.v    = float(input("Enter the value of the membrane potential v: "))             # vis the membrane potential
        # self.vna  = float(input("Enter the value of vNa: "))                                  # Potassium potential
        # self.vk   = float(input("Enter the value of vK: "))                                   # Sodium potential
        # self.vl   = float(input("Enter the value of vl: "))                                   # Leak potential
        # self.vthresh   = float(input("Enter the value of vthresh: "))                         # Voltage threshold for spikes
        # self.tmin = float(input("Enter the start time : "))
        # self.tmax = float(input("Enter the end time: "))
        # self.i_inj   = float(input("Enter the value of Input current i_inj: "))               # Input current

                
        self.g_na = 120.*milli               
        self.g_k  = 36.*milli                       
        self.g_l  = 0.3*milli                   
        self.c_m  = 1.*mu
        self.v   = 0.*milli
        self.vna  = 115.*milli                        
        self.vk   = -12.*milli                        
        self.vl   = 10.613*milli   
        self.vrest = 0.*milli
        self.vthresh = -55.*milli                      
        self.tmin = 0.*milli
        self.tmax = 35.*milli  
        self.i_inj = 10.*mu
        
        # self.dt = 0.001
        self.t  = np.linspace(self.tmin, self.tmax, 1000) # Total duration of simulation [ms]


    # α and β are the forward and backwards rate, respectively
    # These are the original HH equations for α,β where the constants vary in order to fit adequately the data
    # α(v) = (A+B*v)/C + H*exp((v+D) / F) where A,B,C,D,F,H are constants to be fit to the data
    
    # NOTE: for cases where 0/0 might occur then L'Hospital's rules must apply
    def alpha_n(self,v):
        nom = 0.01*kHz * (10*milli - v)
        denom = (np.exp((10*milli - v) / 10*milli) - 1)

        if nom == 0 and denom == 0 :
            return 0.1
        else:
            return nom / denom
    def beta_n(self,v):
        return 0.125*kHz * np.exp(- v/ 80.*milli)

    def alpha_m(self,v):
        nom = 0.1*kHz  * (25.*milli - v)
        denom = (np.exp((25.*milli - v) / 10.*milli) - 1.)

        if nom == 0 and denom == 0 :
            return 1.5/(-1 + np.exp(3./2.))
        else:
            return nom / denom
    def beta_m(self,v):
        return 4.*kHz * np.exp(- v/ 18.*milli)

    def alpha_h(self,v):
        return 0.07*kHz * np.exp(- v/ 20.*milli)
    def beta_h(self,v):
        return 1.*kHz / (np.exp((30.*milli -v) / 10.*milli) + 1.)

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

    # def frequency(self,v):
    #     self.firing_rate = []
    #     self.iinj = np.linspace(0, self.i_inj, 1000)
    #     rec_spikes = 0               # record spike times

    #     for n in range(len(self.iinj)):
    #         for l in range(len(self.t)-1):
    #             # print(self.vthresh)
    #             # print(v[l])
    #             if v[l] >= self.vthresh:
    #                 rec_spikes +=1
    #                 v[l+1] = self.vrest
    #         self.firing_rate.append(rec_spikes/self.tmax)

        # self.max_I = []
        # for i in range(len(self.iinj)):
        #     if self.firing_rate[i] ==0:
        #         self.max_I.append(self.iinj[i])          # threshold input current.
        # print(max(self.max_I))


    def derivatives(self,y,t):
        der = np.zeros(4)
        
        v= y[0]
        n = y[1]
        m = y[2]
        h = y[3]

        gNa = self.g_na * np.power(m, 3.0) * h
        gK = self.g_k * np.power(n, 4.0)
        gL = self.g_l

        i_na = gNa * (v- self.vna )
        i_k = gK * (v- self.vk )
        i_l = gL * (v- self.vl )


        der[0] = (self.i_inj - i_na - i_k - i_l) / self.c_m   # dv/dt
        der[1] = (self.alpha_n(v) * (1 - n)) - (self.beta_n(v) * n)    # dn/dt
        der[2] = (self.alpha_m(v) * (1 - m)) - (self.beta_m(v) * m)    # dm/dt
        der[3] = (self.alpha_h(v) * (1 - h)) - (self.beta_h(v) * h)    # dh/dt
        
        return der

    def Main(self):
        self.default_pars()
        v= self.v
        t = self.t
        y = np.array([v, self.n_inf(v), self.m_inf(v), self.h_inf(v)], dtype= 'float64')

        sol = odeint(self.derivatives, y, t)    # Solve ODE
        v = sol[:,0]
        n = sol[:,1]
        m = sol[:,2]
        h = sol[:,3]

        # self.frequency(sol[:,0]) #TODO change sol to v if you want to convert at the end

        ax = plt.subplot()
        ax.plot(t, v)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Membrane potential (V)')
        ax.set_title('Neuron potential')
        plt.grid()
        plt.savefig('Neuron Potential')
        plt.show()

        # ax = plt.subplot()
        # ax.plot(t*milli, n, 'b', label='Potassium Channel: n')
        # ax.plot(t*milli, m, 'g', label='Sodium (Opening): m')
        # ax.plot(t*milli, h, 'r', label='Sodium Channel (Closing): h')
        # ax.set_ylabel('Gating value')
        # ax.set_xlabel('Time (s)')
        # ax.set_title('Potassium and Sodium channels')
        # plt.legend()
        # plt.savefig('Potassium and Sodium channels (time)')
        # plt.show()


        # # Trajectories with limit cycles
        # ax = plt.subplot()
        # ax.plot(v, n, 'b', label='V - n')
        # ax.plot(v, m, 'g', label='V - m')
        # ax.plot(v, h, 'r', label='V - h')
        # ax.set_ylabel('Gating value')
        # ax.set_xlabel('Voltage (V)')
        # ax.set_title('Limit cycles')
        # plt.legend()
        # plt.savefig('Limit Cycles')
        # plt.show()

        # F-I curve
        # ax = plt.subplot()
        # ax.plot(self.iinj, self.firing_rate)
        # # ax.plot(max(self.max_I))
        # ax.set_xlabel("Input Current(μA/cm^2)")
        # ax.set_ylabel("Firing rate(Hz)")
        # ax.set_title('f-I Curve')
        # plt.savefig('f-I Curve')
        # plt.show()

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()