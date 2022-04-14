from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Markov_Model import *
from HH_model import *
import time

# Na = Sodium
# K = Potassium

start_time = time.time()    # Monitor performance
tmax = 35                   # Duration of the simulation
i_na_trace = []             # Trace for graph purposes
i_k_trace = []
i_leak_trace = []

class Somatic_Voltage:

    def __init__(self,g_na,g_k,g_l,c_m,v_init,vna,vk,vl,i_inj):

        self.g_na     = g_na               
        self.g_k      = g_k                      
        self.g_l      = g_l                   
        self.c_m      = c_m
        self.v_init   = v_init
        self.vna      = vna                        
        self.vk       = vk                        
        self.vl       = vl
        self.i_inj    = i_inj
        
    def derivative(self,v,o,n):

        GNa = self.g_na * o    
        GK = self.g_k * n**4.0
        GL = self.g_l

        i_na = GNa * (v - self.vna )
        i_k = GK * (v - self.vk )
        i_l = GL * (v - self.vl )

        i_na_trace.append(i_na)
        i_k_trace.append(i_k)
        i_leak_trace.append(i_l)

        deriv = (self.i_inj - i_na - i_k - i_l) / self.c_m                          # dV/dt
        return deriv

class All_Derivatives:

    def __init__(self,somatic_voltage,mstates,hh):

        self.hh = hh
        self.somatic_voltage = somatic_voltage
        self.markov = mstates

    def __call__(self,t,y):     # t and y are called by the integrator

        v_soma   = y[0]
        n        = y[1]
        markov_y = y[2:15]
        markov_o = y[13]

        dv = self.somatic_voltage.derivative(v_soma,markov_o,n)                                                                 # Call dV/dt and pass the parameters
        dn = (self.hh.alpha_n(v_soma) * (1 - n)) - (self.hh.beta_n(v_soma) * n)                                                 # dn/dt
        dm = self.markov.derivatives(markov_y,self.markov.alpha(v_soma),self.markov.beta(v_soma),self.markov.ksi(v_soma))

        all_derivatives = np.append([dv,dn],dm)
        return all_derivatives

class Model:
    def Main(self):

        def normalize(y):
            norm = y[2:15]/np.sum(y[2:15])      # normalise only the markov results from y[2] to y[15]
            y[2:15] = norm
            return y

        mstates = Markov(γ=150., δ=40., ε=1.75, d=0.005, u=0.5, n=0.75, f=0.005, a=3.3267) 
        hh = HodgkinHuxley() 

        # somatic_voltage=Somatic_Voltage(g_na=60., g_k=200., g_l=2., c_m=1., v_init=-50., vna=115., vk=-88., vl=-30, i_inj=10.)
        # somatic_voltage=Somatic_Voltage(105.,15.,2.,1.,-70.,45.,-88.,-88,10.)              # Parameter values from Parameters2.py
        somatic_voltage=Somatic_Voltage(60.,200.,2.,1.,-50,45.,-88.,-88.,40.)             # Parameter values from research paper

        v_initial = somatic_voltage.v_init

        f = All_Derivatives(somatic_voltage,mstates,hh)  # All derivatives in one function

        # Initialisation
        y = np.array([v_initial,hh.n_inf(v_initial),1,0,0,0,0,0,0,0,0,0,0,0,0],dtype='float64')
        bigv=bigt=bigo=bigb=bigi6=bigc5 = np.array([])
        i = status = 0
        step = 0.0025
        norm_const = print_const = 1
        j = 1
        step = step*j
        print_n = 10
        while i< tmax and status==0 :

            result = solve_ivp(f, t_span=(i,i+step), y0=y, method='BDF')
            # print(np.shape(result.y))                                      # (15,len(t_eval)) 15 derivatives and the length of t_eval if one exists
            y_norm = result.y[:,-1]
            y = normalize(y_norm)                                        # Normalise only the markov derivatives at every step

            if print_const ==1:
                bigv  = np.concatenate((bigv,result.y[0]))                       # result.y[0] = result.y[0,:]
                bigt  = np.concatenate((bigt,result.t))   
                bigc5 = np.concatenate((bigc5,result.y[6]))     
                bigi6 = np.concatenate((bigi6,result.y[12]))
                bigo  = np.concatenate((bigo,result.y[13]))
                bigb  = np.concatenate((bigb,result.y[14]))
                print_const=print_n

            status = result.status                                       # -1: Integration step failed.
            i+=step                                                      #  0: The solver successfully reached the end of t_span.
            print_const     -= 1  

        ax = plt.subplot()
        ax.plot(bigt, bigo,c='r',label='o')
        ax.plot(bigt, bigb,c='b',label='b')
        ax.plot(bigt, bigi6,c='orange',label='i6')
        ax.plot(bigt, bigc5,c='g',label='c5')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Fraction')
        plt.legend()
        plt.grid()
        plt.show()

        ax = plt.subplot()
        ax.plot(bigt, bigv)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Membrane potential (mV)')
        ax.set_title('Neuron potential')
        plt.grid()
        plt.savefig('Figures/Neuron Potential Full model')
        plt.show()

# hh.graphs(bigv,bigt,i_na_trace,i_k_trace,i_leak_trace,bigo,bigb,bigi6,bigc5)
if __name__ == '__main__':
    runner = Model()
    runner.Main()
print("--- %s seconds ---" % (time.time() - start_time))
