from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Markov_Model import *
from HH_model import *
import time

start_time = time.time()
tmax = 35.

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
        
    def normalize(y):
        # y is a array of shape (16,)
        total=0
        for i in range(2,15):
            total+=y[i]

        for i in range(2,15):
            y[i]/=total

        return y    

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

        der[0] = (inj - i_na - i_k - i_l) / self.c_m                          # dv/dt
        der[1] = (self.hh.alpha_n(v) * (1 - n)) - (self.hh.beta_n(v) * n)     # dn/dt

        # print(der)
        return der


    def Main(self):
        v_init = self.v_init
        t = self.t

        y = np.array([v_init, self.hh.n_inf(v_init)], dtype= 'float64')

        my = self.markovian.mark_intgr(v_init) # this is the last y from markov hence no resurgence, shape(13,)
        result = solve_ivp(self.all_derivatives, t_span=(0,tmax), y0=y, t_eval=t, method='BDF', args=(self.hh.i_inj, my[11])) 
        # print(result)

        vp  = result.y[0,:]   
        n   = result.y[1,:]

        ax = plt.subplot()
        ax.plot(t, vp)
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