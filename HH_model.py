from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

start_time = time.time()

class HodgkinHuxley:      # Hodgkin - Huxley model
    def __init__(self):
        ### ALL UNITS NEED TO BE IN S.I. ###
       
        # print("Please Enter all the values in default units (graphs will automatically convert to S.I. units)")
        # self.g_na = float(input("Enter the value of gNa: "))                                   # Average sodoum channel conductance per unit area
        # self.g_k  = float(input("Enter the value of gK: "))                                    # Average potassium channel conductance per unit area
        # self.g_l  = float(input("Enter the value of gl: "))                                    # Average leak channel conductance per unit area
        # self.C_m  = float(input("Enter the value of membrane capacitance c_m: "))              # Membrane capacitance per unit area
        # self.v    = float(input("Enter the value of the membrane potential v: "))              # v is the membrane potential
        # self.vna  = float(input("Enter the value of vNa: "))                                   # Potassium potential
        # self.vk   = float(input("Enter the value of vK: "))                                    # Sodium potential
        # self.vl   = float(input("Enter the value of vl: "))                                    # Leak potential
        # self.vthresh   = float(input("Enter the value of voltage threshold: "))                # Voltage threshold for spikes
        # self.tmax = float(input("Enter the total duration of simulation: "))
        # self.i_inj   = float(input("Enter the value of Input current i_inj: "))                # Input current

                
        self.g_na = 120.               
        self.g_k  = 36.                      
        self.g_l  = 0.3                   
        self.c_m  = 1.
        self.v   = 0.
        self.vna  = 115.                        
        self.vk   = -12.                        
        self.vl   = 10.613   
        self.vthresh = 55.                      
        self.tmax = 35.   
        self.i_inj = 10.   # TODO: ask about this 10**(-2) conversion is done in cm2?
        
        self.t  = np.linspace(0, self.tmax, 100)

    def alpha_n(self,v):

        nom = 0.01* (10. - v)
        denom = (np.exp((10. - v) / 10.) - 1.)
        if nom == 0 and denom == 0 :
            return 0.1
        # return (0.22 * np.exp( (v-30)/ 26.5))      # Research paper equation
        return (nom / denom)                         # Wiki equation(original HH)

    def beta_n(self,v):
        # return (0.22 * np.exp(- (v-30)/ 26.5))     # Research paper equation
        return (0.125 * np.exp(- v/ 80.))            # Wiki equation(original HH)

    def alpha_m(self,v):
        nom = 0.1  * (25. - v)
        denom = (np.exp((25. - v) / 10.) - 1.)
        if nom == 0 and denom == 0 :
            return (1.5/(-1 + np.exp(3./2.)))
        return (nom / denom)

    def beta_m(self,v):
        return (4. * np.exp(- v/ 18.))

    def alpha_h(self,v):
        return (0.07 * np.exp(- v/ 20.))

    # h is replaced by the markovian scheme
    def beta_h(self,v):
        return (1. / (np.exp((30. -v) / 10.) + 1.))

    def n_inf(self,v):
        return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

    # m is replaced by the markovian scheme
    def m_inf(self,v):
        return self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))

    def h_inf(self,v):
        return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))

    def frequency(self,y):
        firing_rate = []
        self.max_I = []
        self.var_inj = [8,23,25,52,115,215]          # np.linspace(0, self.i_inj, 100)
        spikes = 0               

        for i in range(len(self.var_inj)):
            spikes = 0
            result = solve_ivp(self.derivatives, t_span=(0,self.tmax), y0=y, t_eval=self.t, args=(self.var_inj[i],),method='BDF') 

            for n in range(len(self.t)):
                if result.y[0,n] >= self.vthresh and result.y[0,n-1] < self.vthresh:        # OPTIMISE FOR loops with NUMPY or list comprehensions
                    spikes += 1
            firing_rate.append(spikes/self.tmax)


        for i in range(len(self.var_inj)):
            if firing_rate[i] ==0:
                self.max_I.append(self.var_inj[i])          # threshold input current.
        print(max(self.max_I))

        return firing_rate
   
    def derivatives(self,t,y,inj):
        der = np.zeros(4)
        v = y[0]
        n = y[1]
        m = y[2]
        h = y[3]

        GNa = self.g_na * m**3.0 * h    #    m and h is replaced by the markovian scheme
        GK = self.g_k * n**4.0
        GL = self.g_l

        i_na = GNa * (v - self.vna )
        i_k = GK * (v - self.vk )
        i_l = GL * (v - self.vl )


        der[0] = (inj - i_na - i_k - i_l) / self.c_m                   # dv/dt
        der[1] = (self.alpha_n(v) * (1 - n)) - (self.beta_n(v) * n)    # dn/dt
        der[2] = (self.alpha_m(v) * (1 - m)) - (self.beta_m(v) * m)    # dm/dt
        der[3] = (self.alpha_h(v) * (1 - h)) - (self.beta_h(v) * h)    # dh/dt

        return der

    def graphs(self,v,t,ina,ik,il,bigo,bigb,bigi6,bigc5):
        # ADD injection current graph(either constant or the F-I curve)
        # ADD Channel conductances
        ax = plt.subplot()
        ax.plot(t, ina, 'b', label='Potassium Current')
        ax.plot(t, ik,  'g', label='Sodium Current')
        ax.plot(t, il,  'r', label='Leak Current')
        ax.set_title('Channel currents')
        plt.grid()
        plt.savefig('Figures/Channel currents')
        plt.show()

        # ax = plt.subplot()
        # ax.plot(bigt, bigo,c='r',label='o')
        # ax.plot(bigt, bigb,c='b',label='b')
        # ax.plot(bigt, bigi6,c='orange',label='i6')
        # ax.plot(bigt, bigc5,c='g',label='c5')
        # ax.set_xlabel('Time (ms)')
        # ax.set_ylabel('Fraction')
        # plt.legend()
        # plt.grid()
        # plt.savefig('Figures/Markovian fraction')
        # plt.show()

    # def lala(self):
    #     v = self.v
    #     t = self.t

    #     y = np.array([v, self.n_inf(v), self.m_inf(v), self.h_inf(v)], dtype= 'float64')

    #     result = solve_ivp(self.derivatives, t_span=(0,self.tmax), y0=y, t_eval=self.t, args=(self.i_inj,)) 
        
    #     vp = result.y[0,:]    #TODO: if conversion is done properly at the beggining then *milli is not needed
    #     n = result.y[1,:]
    #     m = result.y[2,:]
    #     h = result.y[3,:]

        # firing_rate = self.frequency(y)

        # Markov.error(self,105.40*milli,max(vp))   #compare simulation peak height with actual paper(careful with parameters)

        # ax = plt.subplot()
        # ax.plot(t, vp)
        # ax.set_xlabel('Time (ms)')
        # ax.set_ylabel('Membrane potential (mV)')
        # ax.set_title('Neuron potential')
        # plt.grid()
        # plt.savefig('Figures/Neuron Potential')
        # plt.show()

        # ax = plt.subplot()
        # ax.plot(t, n, 'b', label='Potassium Channel: n')
        # ax.plot(t, m, 'g', label='Sodium (Opening): m')
        # ax.plot(t, h, 'r', label='Sodium Channel (Closing): h')
        # ax.set_ylabel('Gating value')
        # ax.set_xlabel('Time (ms)')
        # ax.set_title('Potassium and Sodium channels using the Hodgking-Huxley model')
        # plt.legend()
        # plt.savefig('Figures/Ion channel gating variables with respect to time')
        # plt.show()


        # # Trajectories with limit cycles
        # ax = plt.subplot()
        # ax.plot(vp, n, 'b', label='V - n')
        # ax.plot(vp, m, 'g', label='V - m')
        # ax.plot(vp, h, 'r', label='V - h')
        # ax.set_ylabel('Gating value')
        # ax.set_xlabel('Voltage (mV)')
        # ax.set_title('Limit cycles')
        # plt.legend()
        # plt.savefig('Figures/Limit Cycles')
        # plt.show()
        # return vp

        # F-I curve
        # ax = plt.subplot()
        # ax.plot(self.var_inj, firing_rate)
        # ax.plot(max(self.max_I),0,c='r',marker='o', label="threshold input current")
        # ax.set_xlabel("Input Current(uA)")
        # ax.set_ylabel("Firing rate(kHz)")
        # ax.set_title('f-I Curve')
        # plt.legend()
        # plt.savefig('f-I Curve')
        # plt.show()