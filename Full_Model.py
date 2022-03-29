from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Markov_Model import *
from HH_model import *
import time

start_time = time.time()

class FullModel():
    def __init__(self):
        self.g_na = 120.               
        self.g_k  = 36.                      
        self.g_l  = 0.3                   
        self.c_m  = 1.
        self.v   = 0.
        self.vna  = 115.                        
        self.vk   = -12.                        
        self.vl   = 10.613   
        self.vthresh = 55.*milli                      
        self.tmax = 35.   
        self.i_inj = 10.   
        
        self.t  = np.linspace(0, self.tmax, 100)
        # self.markovian = Markov() 

    def alpha_n(self,v):
        nom = 0.01* (10 - v)
        denom = (np.exp((10 - v) / 10) - 1)

        if nom == 0 and denom == 0 :
            return 0.1
        else:
            return (nom / denom)
    def beta_n(self,v):
        return (0.125 * np.exp(- v/ 80.))

    def alpha_h(self,v):
        return (0.07 * np.exp(- v/ 20.))
    def beta_h(self,v):
        return (1. / (np.exp((30. -v) / 10.) + 1.))

    def n_inf(self,v):
        return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

    def h_inf(self,v):
        return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))

    def frequency(self,y):
        firing_rate = []
        self.max_I = []
        self.var_inj = np.linspace(0, self.i_inj, 6)
        spikes = 0               

        for i in range(len(self.var_inj)):
            spikes = 0
            result = solve_ivp(self.derivatives, t_span=(0,self.tmax), y0=y, t_eval=self.t, args=(self.var_inj[i],),method='BDF') 

            for n in range(len(self.t)):
                if result.y[0,n]*milli >= self.vthresh and result.y[0,n-1]*milli < self.vthresh:
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

        GNa = self.g_na * m**3.0 * h    #This will need to change when we merge the files
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

    def lala(self):
        v = self.v
        t = self.t

        # m = self.markovian.scheme(v)
        # print(np.shape(m))

        y = np.array([v, self.n_inf(v), self.m_inf(v), self.h_inf(v)], dtype= 'float64')

        result = solve_ivp(self.derivatives, t_span=(0,self.tmax), y0=y, t_eval=self.t, args=(self.i_inj,)) 
        
        vp = result.y[0,:]    #TODO: if conversion is done properly at the beggining then *milli is not needed
        n = result.y[1,:]
        m = result.y[2,:]
        h = result.y[3,:]

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
        # ax.set_title('Potassium and Sodium channels')
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
 

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()
    print("--- %s seconds ---" % (time.time() - start_time))