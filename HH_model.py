from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Markov_Model import *
import time
start_time = time.time()

class HodgkinHuxley():
    # Hodgkin - Huxley model

    def __init__(self):
        ### ALL UNITS NEED TO BE IN S.I. ###
        # Paremeters are passed from the command line when the program is executed
       
        # print("Please Enter all the values in default units (graphs will automatically convert to S.I. units)")
        # self.g_na = float(input("Enter the value of gNa: "))                                   # Average sodoum channel conductance per unit area
        # self.g_k  = float(input("Enter the value of gK: "))                                    # Average potassium channel conductance per unit area
        # self.g_l  = float(input("Enter the value of gl: "))                                    # Average leak channel conductance per unit area
        # self.C_m  = float(input("Enter the value of membrane capacitance c_m: "))              # Membrane capacitance per unit area
        # self.v    = float(input("Enter the value of the membrane potential v: "))              # vis the membrane potential
        # self.vna  = float(input("Enter the value of vNa: "))                                   # Potassium potential
        # self.vk   = float(input("Enter the value of vK: "))                                    # Sodium potential
        # self.vl   = float(input("Enter the value of vl: "))                                    # Leak potential
        # self.vthresh   = float(input("Enter the value of voltage threshold: "))                # Voltage threshold for spikes
        # self.tmax = float(input("Enter the total duration of simulation: "))
        # self.i_inj   = float(input("Enter the value of Input current i_inj: "))                # Input current

                
        self.g_na = 120.*milli               
        self.g_k  = 36.*milli                       
        self.g_l  = 0.3*milli                   
        self.c_m  = 1.*mu
        self.v   = 0.
        self.vna  = 115.                        
        self.vk   = -12.                        
        self.vl   = 10.613   
        self.vthresh = 55.*milli                      
        self.tmin = 0.*milli
        self.tmax = 35.*milli    # Total duration of simulation [ms]
        self.i_inj = 10.*milli   # TODO: ask about this 10**(-2) conversion is done in cm2?
        
        self.t  = np.linspace(self.tmin, self.tmax, 1000) 

    def alpha_n(self,v):
        nom = 0.01* (10 - v)
        denom = (np.exp((10 - v) / 10) - 1)

        if nom == 0 and denom == 0 :
            return 0.1*kHz
        else:
            return (nom / denom)*kHz
    def beta_n(self,v):
        return (0.125 * np.exp(- v/ 80.))*kHz

    def alpha_m(self,v):
        nom = 0.1  * (25. - v)
        denom = (np.exp((25. - v) / 10.) - 1.)

        if nom == 0 and denom == 0 :
            return (1.5/(-1 + np.exp(3./2.)))*kHz
        else:
            return (nom / denom)*kHz
    def beta_m(self,v):
        return (4. * np.exp(- v/ 18.))*kHz

    def alpha_h(self,v):
        return (0.07 * np.exp(- v/ 20.))*kHz
    def beta_h(self,v):
        return (1. / (np.exp((30. -v) / 10.) + 1.))*kHz

    def n_inf(self,v):
        return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

    def m_inf(self,v):
        return self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))

    def h_inf(self,v):
        return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))


    def frequency(self,y):
        firing_rate = []
        self.max_I = []
        self.var_inj = np.linspace(0, self.i_inj, 100)
        spikes = 0               

        for i in range(len(self.var_inj)):
            sol = odeint(self.derivatives, y, self.t, args=(self.var_inj[i],))

            for n in range(len(self.t)):
                if sol[n][0]*milli >= self.vthresh and sol[n-1][0]*milli < self.vthresh:
                    spikes += 1
            firing_rate.append(spikes/self.tmax)


        for i in range(len(self.var_inj)):
            if firing_rate[i] ==0:
                self.max_I.append(self.var_inj[i])          # threshold input current.
        print(max(self.max_I))

        return firing_rate
        

    def derivatives(self,y,t,inj):
        der = np.zeros(4)
        v = y[0]
        n = y[1]
        m = y[2]
        h = y[3]

        gNa = self.g_na * m**3.0 * h
        gK = self.g_k * n**4.0
        gL = self.g_l

        i_na = gNa * (v - self.vna )
        i_k = gK * (v - self.vk )
        i_l = gL * (v - self.vl )


        der[0] = (inj - i_na - i_k - i_l) / self.c_m                   # dv/dt
        der[1] = (self.alpha_n(v) * (1 - n)) - (self.beta_n(v) * n)    # dn/dt
        der[2] = (self.alpha_m(v) * (1 - m)) - (self.beta_m(v) * m)    # dm/dt
        der[3] = (self.alpha_h(v) * (1 - h)) - (self.beta_h(v) * h)    # dh/dt

        return der

    def Main(self):
        v = self.v
        t = self.t
        y = np.array([v, self.n_inf(v), self.m_inf(v), self.h_inf(v)], dtype= 'float64')

        sol = odeint(self.derivatives, y, t, args=(self.i_inj,))    # Solve ODE
        vp = sol[:,0]*milli     #TODO: if conversion is done properly at the beggining then *milli is not needed
        n = sol[:,1]
        m = sol[:,2]
        h = sol[:,3]

        firing_rate = self.frequency(y)

        ax = plt.subplot()
        ax.plot(t, vp)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Membrane potential (V)')
        ax.set_title('Neuron potential')
        plt.grid()
        plt.savefig('Neuron Potential')
        plt.show()

        ax = plt.subplot()
        ax.plot(t, n, 'b', label='Potassium Channel: n')
        ax.plot(t, m, 'g', label='Sodium (Opening): m')
        ax.plot(t, h, 'r', label='Sodium Channel (Closing): h')
        ax.set_ylabel('Gating value')
        ax.set_xlabel('Time (s)')
        ax.set_title('Potassium and Sodium channels')
        plt.legend()
        plt.savefig('Potassium and Sodium channels (time)')
        plt.show()


        # Trajectories with limit cycles
        ax = plt.subplot()
        ax.plot(vp, n, 'b', label='V - n')
        ax.plot(vp, m, 'g', label='V - m')
        ax.plot(vp, h, 'r', label='V - h')
        ax.set_ylabel('Gating value')
        ax.set_xlabel('Voltage (V)')
        ax.set_title('Limit cycles')
        plt.legend()
        plt.savefig('Limit Cycles')
        plt.show()

        # F-I curve
        ax = plt.subplot()
        ax.plot(self.var_inj, firing_rate)
        ax.plot(max(self.max_I),0,c='r',marker='o', label="threshold input current")
        ax.set_xlabel("Input Current(A)")
        ax.set_ylabel("Firing rate(Hz)")
        ax.set_title('f-I Curve')
        plt.legend()
        plt.savefig('f-I Curve')
        plt.show()

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()
    print("--- %s seconds ---" % (time.time() - start_time))