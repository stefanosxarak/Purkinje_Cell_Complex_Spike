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
        self.g_na = 120.               
        self.g_k  = 36.                      
        self.g_l  = 0.3                   
        self.c_m  = 1.
        self.v   = 0.
        self.vna  = 115.                        
        self.vk   = -12.                        
        self.vl   = 10.613   
        self.i_inj = 10.   
        
        self.t  = np.linspace(0, tmax, 100)
        self.markovian = Markov() 

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
        
    def derivatives(self,t,y,inj,α,β,ξ,γ,δ,ε,d,u,n,f,a):
        der = np.zeros(16)
        v = y[0]
        n = y[1]
        h = y[2]

        GNa = self.g_na * self.markovian.o    
        GK = self.g_k * n**4.0
        GL = self.g_l

        i_na = GNa * (v - self.vna )
        i_k = GK * (v - self.vk )
        i_l = GL * (v - self.vl )

        mder   = self.markovian.derivatives(t,y,v,α,β,ξ,γ,δ,ε,d,u,n,f,a)

        der[0] = (inj - i_na - i_k - i_l) / self.c_m                   # dv/dt
        der[1] = (self.alpha_n(v) * (1 - n)) - (self.beta_n(v) * n)    # dn/dt
        der[2] = mder[0]                                               # dC1/dt
        der[3] = mder[1]                                               # dC2/dt
        der[4] = mder[2]                                               # dC3/dt
        der[5] = mder[3]                                               # dC4/dt
        der[6] = mder[4]                                               # dC5/dt
        der[7] = mder[5]                                               # dI1/dt
        der[8] = mder[6]                                               # dI2/dt
        der[9] = mder[7]                                               # dI3/dt
        der[10] = mder[8]                                              # dI4/dt
        der[11] = mder[9]                                              # dI5/dt
        der[12] = mder[10]                                             # dI6/dt
        der[13] = mder[11]                                             # do/dt
        der[14] = mder[12]                                             # db/dt
        der[15] = (self.alpha_h(v) * (1 - h)) - (self.beta_h(v) * h)   # dh/dt

        return der

    def Main(self):
        v = self.v
        t = self.t

        y = np.array([v, self.n_inf(v), self.h_inf(v),self.markovian.c0,self.markovian.c1,self.markovian.c2,self.markovian.c3,self.markovian.c4,self.markovian.I0,self.markovian.I1,self.markovian.I2,self.markovian.I3,self.markovian.I4,self.markovian.I5,self.markovian.o,self.markovian.b], dtype= 'float64')

        result = solve_ivp(self.derivatives, t_span=(0,tmax), y0=y, t_eval=t,method='BDF', args=(self.i_inj,self.markovian.alpha(v), self.markovian.beta(v), self.markovian.ksi(v), self.markovian.γ,self.markovian.δ,self.markovian.ε,self.markovian.d,self.markovian.u,self.markovian.n,self.markovian.f,self.markovian.a,)) 
        
        vp = result.y[0,:]   
        n = result.y[1,:]
        m1 = result.y[2,:]
        m2 = result.y[3,:]
        m3 = result.y[4,:]
        m4 = result.y[5,:]
        m5 = result.y[6,:]
        m6 = result.y[7,:]
        m7 = result.y[8,:]
        m8 = result.y[9,:]
        m9 = result.y[10,:]
        m10 = result.y[11,:]
        m11 = result.y[12,:]
        m12 = result.y[13,:]
        m13 = result.y[14,:]
        h = result.y[15,:]

        print(vp)

        # Markov.error(self,105.40,max(vp))   #compare simulation peak height with actual paper(careful with parameters)

        # ax = plt.subplot()
        # ax.plot(t, vp)
        # ax.set_xlabel('Time (ms)')
        # ax.set_ylabel('Membrane potential (mV)')
        # ax.set_title('Neuron potential')
        # plt.grid()
        # plt.savefig('Figures/Neuron Potential')
        # plt.show()
 

if __name__ == '__main__':
    runner = FullModel()
    runner.Main()
    print("--- %s seconds ---" % (time.time() - start_time))