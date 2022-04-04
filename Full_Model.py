from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Markov_Model import *
from HH_model import *
import time

start_time = time.time()
tmax = 35

class FullModel():
    def __init__(self):
        self.g_na     = 120.               
        self.g_k      = 36.                      
        self.g_l      = 0.3                   
        self.c_m      = 1.
        self.v_init   = 0.
        self.vna      = 115.                        
        self.vk       = -12.                        
        self.vl       = 10.613   
        
        self.t  = np.linspace(0, tmax, 1000)
        self.markovian = Markov() 
        self.hh = HodgkinHuxley() 

        self.c0 = self.markovian.c0
        self.c1 = self.markovian.c1
        self.c2 = self.markovian.c2
        self.c3 = self.markovian.c3
        self.c4 = self.markovian.c4
        self.I0 = self.markovian.I0
        self.I1 = self.markovian.I1
        self.I2 = self.markovian.I2
        self.I3 = self.markovian.I3
        self.I4 = self.markovian.I4
        self.I5 = self.markovian.I5
        self.o = self.markovian.o
        self.b = self.markovian.b
        
    def all_derivatives(self,t,y,inj,o):
        der = np.zeros(2)

        v = y[0]
        n = y[1]

        GNa = self.g_na * o    
        GK = self.g_k * n**4.0
        GL = self.g_l

        i_na = GNa * (v - self.vna )
        i_k = GK * (v - self.vk )
        i_l = GL * (v - self.vl )

        der[0] = (inj - i_na - i_k - i_l) / self.c_m                          # dV/dt
        der[1] = (self.hh.alpha_n(v) * (1 - n)) - (self.hh.beta_n(v) * n)     # dn/dt

        # hhder  = self.hh.derivatives(t,y,inj)
        # der[0] = hhder[0]
        # der[1] = hhder[1]
        # der[2] = hhder[3]
        print(der)
        return der


    def Main(self):
        v = self.v_init

        bigy = np.array([])
        bigt = np.array([])

        my = np.array([self.c0,self.c1,self.c2,self.c3,self.c4,self.I0,self.I1,self.I2,self.I3,self.I4,self.I5,self.o,self.b])
        y = np.array([v, self.hh.n_inf(v)], dtype= 'float64')
        
        for i in range(0,tmax):
            my = self.markovian.mark_intgr(v,i,my)     # shape(13,)

            result = solve_ivp(self.all_derivatives, t_span=(0,35), y0=y, t_eval=(np.linspace(i, i+0.0025, 1)), method='BDF', args=(self.hh.i_inj, my[11]))

            bigy = np.concatenate((bigy,result.y[0,:]))   
            bigt = np.concatenate((bigt,result.t))

            v = result.y[0,:]
            y = np.squeeze(np.array([v, self.hh.n_inf(v)], dtype= 'float64'))


        ax = plt.subplot()
        ax.plot(bigt, bigy)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Membrane potential (mV)')
        ax.set_title('Neuron potential')
        plt.grid()
        plt.savefig('Figures/Neuron Potential Full model')
        plt.show()
 

if __name__ == '__main__':
    runner = FullModel()
    runner.Main()
    print("--- %s seconds ---" % (time.time() - start_time))