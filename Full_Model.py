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
        bigy = np.array([])
        bigt = np.array([])

        y = np.array([v_init, self.hh.n_inf(v_init)], dtype= 'float64')

        for i in range(0,tmax):
            my = self.markovian.mark_intgr(v_init,i) # this is the last y from markov hence no resurgence, shape(13,)
            result = solve_ivp(self.all_derivatives, t_span=(i,i+1), y0=y,t_eval=([i]), method='BDF', args=(self.hh.i_inj, my[11]))
            bigy = np.concatenate((bigy,result.y[0]))   
            bigt = np.concatenate((bigt,result.t))

            print(np.shape(result.y[0,:]))
            y = np.array([result.y[0,:], self.hh.n_inf(result.y[0,:])], dtype= 'float64') # Y does not update properly
            print(np.shape(y))

        print(np.shape(bigy))

        # vp  = result.y[0,:]   
        # n   = result.y[1,:]

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