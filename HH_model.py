import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class HodgkinHuxley:      

    def __init__(self):        
        #   Initial parameters as stated in Hodgkin-Huxley(1952)
        #   Units are not in S.I.
        self.g_na = 120.               
        self.g_k  = 36.                      
        self.g_l  = 0.3                   
        self.c_m  = 1.
        self.v   = 0.
        self.vna  = 115.                        
        self.vk   = -12.                        
        self.vl   = 10.613   
        self.vthresh = 55.
        self.i_inj = 10.   

        self.tmax = 35.   
        self.t  = np.linspace(0, self.tmax, 100000)

    def alpha_n(self,v):

        nom = 0.01* (10. - v)
        denom = (np.exp((10. - v) / 10.) - 1.)
        if nom == 0 and denom == 0 :
            return 0.1

        return (nom / denom)                         # original HH

    def beta_n(self,v):

        return (0.125 * np.exp(- v/ 80.))            # original HH

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

    def beta_h(self,v):

        return (1. / (np.exp((30. -v) / 10.) + 1.))

    def n_inf(self,v):

        return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

    def m_inf(self,v):

        return self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))

    def h_inf(self,v):

        return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))
   
    def derivatives(self,t,y,inj):

        der = np.zeros(4)
        v = y[0]
        n = y[1]
        m = y[2]
        h = y[3]

        GNa = self.g_na * m**3.0 * h   
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

    def run(self):
    
        v = self.v
        t = self.t

        y = np.array([v, self.n_inf(v), self.m_inf(v), self.h_inf(v)], dtype= 'float64')

        result = solve_ivp(self.derivatives, t_span=(0,self.tmax), y0=y, t_eval=t, args=(self.i_inj,)) 
        
        vp = result.y[0,:]    
        n = result.y[1,:]
        m = result.y[2,:]
        h = result.y[3,:]

        err = np.abs((max(vp) - 1.05400725e+02)/(1.05400725e+02) *100)
        print("The error percentage is:", err,"%")

        ax = plt.subplot()
        ax.plot(t, vp)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Membrane potential (mV)')
        ax.set_title('Neuron potential')
        plt.savefig('Figures/Neuron Potential')
        plt.show()

        ax = plt.subplot()
        ax.plot(t, n, 'b', label='Potassium Channel: n')
        ax.plot(t, m, 'g', label='Sodium Channel (Opening): m')
        ax.plot(t, h, 'r', label='Sodium Channel (Closing): h')
        ax.set_ylabel('Gating value')
        ax.set_xlabel('Time (ms)')
        ax.set_title('Ion gating variables')
        plt.savefig('Figures/Ion channel gating variables with respect to time')
        plt.show()

        ax = plt.subplot()
        ax.plot(vp, n, 'b', label='n(t)')
        ax.plot(vp, m, 'g', label='m(t)')
        ax.plot(vp, h, 'r', label='h(t)')
        ax.set_ylabel('Gating value')
        ax.set_xlabel('Voltage (mV)')
        ax.set_title('Limit cycles of the gating equations')
        plt.savefig('Figures/Limit Cycles')
        plt.show()

        i_na = self.g_na * m**3.0 * h * (vp - self.vna )
        i_k = self.g_k * n**4.0 * (vp - self.vk )
        i_l = self.g_l * (vp - self.vl )

        ax = plt.subplot()
        ax.plot(t, -i_na, 'b', label='$I_{Na}$')
        ax.plot(t, -i_k,  'g', label='$I_{K}$')
        ax.plot(t, -i_l,  'r', label='$I_{L}$')
        ax.set_title('Channel currents')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Current (μA)')
        plt.savefig('Figures/Channel currents')
        plt.show()

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.run()