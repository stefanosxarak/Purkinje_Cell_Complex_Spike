from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Markov_Model import *
from HH_model import *
import time

start_time = time.time()    # Monitor performance
tmax = 35                   # Duration of the simulation
i_na_trace = []             # Trace for graph purposes
i_k_trace = []
i_leak_trace = []
class Somatic_Voltage():
    def __init__(self,g_na,g_k,g_l,c_m,v_init,vna,vk,vl):
        self.g_na     = g_na               
        self.g_k      = g_k                      
        self.g_l      = g_l                   
        self.c_m      = c_m
        self.v_init   = v_init
        self.vna      = vna                        
        self.vk       = vk                        
        self.vl       = vl   
        
    def derivative(self,t,y,inj,o,n):
        deriv = np.zeros(1)
        v = y[0]

        GNa = self.g_na * o    
        GK = self.g_k * n**4.0
        GL = self.g_l

        i_na = GNa * (v - self.vna )
        i_k = GK * (v - self.vk )
        i_l = GL * (v - self.vl )

        i_na_trace.append(i_na)
        i_k_trace.append(i_k)
        i_leak_trace.append(i_l)

        deriv = (inj - i_na - i_k - i_l) / self.c_m                          # dV/dt

        return deriv


class All_Derivatives:
    def __init__(self,somatic_voltage,markov_states):
        self.hh = HodgkinHuxley() 
        self.somatic_voltage = somatic_voltage
        self.markov = markov_states
        
    def __call__(self,t,y): # t and y are called by the integrator
        v_soma   = y[0]
        n        = y[1]
        markov_y = y[2:15]
        markov_o = y[13]

        dv = self.somatic_voltage.derivative(t,y,self.hh.i_inj,markov_o,n)         # dV/dt
        
        dn = (self.hh.alpha_n(v_soma) * (1 - n)) - (self.hh.beta_n(v_soma) * n)    # dn/dt

        dm = self.markov.derivatives(markov_y,self.markov.alpha(v_soma),self.markov.beta(v_soma),self.markov.ksi(v_soma)).ravel()
        return np.append([dv,dn],dm)

def normalize(y):
    total=0

    for i in range(2,15):
        total+=y[i]

    for i in range(2,15):
        y[i]/=total
    # norm = y[2::16]/np.sum(y[2::16])      # normalise only the markov results from y[2] to y[15] but return the whole y
    # y[2::16] = norm
    return y


# call markovian scheme to get the markov derivatives
mstates = Markov(150.,40.,1.75,0.005,0.5,0.75,0.005,3.3267)
hh = HodgkinHuxley() 

somatic_voltage=Somatic_Voltage(120.,36.,0.3,1.,0.,115.,-12.,10.613)
v_initial = somatic_voltage.v_init

f = All_Derivatives(somatic_voltage,mstates)  # All derivatives in one function

# Initialisation
y = np.array([v_initial,hh.n_inf(v_initial),1,0,0,0,0,0,0,0,0,0,0,0,0],dtype='float64')
bigv = np.array([])
bigt = np.array([])
i=0
status = 0
while i< tmax and status==0 :
    result = solve_ivp(f, t_span=(i,i+1), y0=y, t_eval=(np.linspace(i, i+1, 1000)), method='BDF')   # t_span stops at tmax
    # print(np.shape(result.y[:,0]))
    y = result.y[:,0]
    y = normalize(y)    # TODO: y does not update
    
    bigv = np.concatenate((bigv,result.y[0,:-1]))           # :-1 to remove the last elemt of the array as it creates duplicate values when concatenated
    bigt = np.concatenate((bigt,result.t[:-1]))             # :-1 to remove the last elemt of the array as it creates duplicate values when concatenated

    status = result.status                                  # -1: Integration step failed.
    i+=1                                                    #  0: The solver successfully reached the end of t_span.
   

ax = plt.subplot()
ax.plot(bigt, bigv)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane potential (mV)')
ax.set_title('Neuron potential')
plt.grid()
plt.savefig('Figures/Neuron Potential Full model')
plt.show()

# hh.graphs(bigv,bigt,i_na_trace,i_k_trace,i_leak_trace)
# if __name__ == '__main__':
#     runner = FullModel()
#     runner.Main()
print("--- %s seconds ---" % (time.time() - start_time))




































#     def Main(self):
#         v = self.v_init

#         bigy = np.array([])
#         bigt = np.array([])

#         my = np.array([self.c0,self.c1,self.c2,self.c3,self.c4,self.I0,self.I1,self.I2,self.I3,self.I4,self.I5,self.o,self.b])
#         y = np.array([v, self.hh.n_inf(v)], dtype= 'float64')
        
#         for i in range(0,tmax):
#             my = self.markovian.mark_intgr(v,i,my)     # shape(13,)
#             o = my[11]
#             result = solve_ivp(self.voltage_derivatives, t_span=(i,i+1), y0=y, method='BDF', args=(self.hh.i_inj, o))
#             print(np.shape(result.y))

#             bigy = np.concatenate((bigy,result.y[0,:]))   
#             bigt = np.concatenate((bigt,result.t))

#             v = result.y[0,:]
#             # print(v)
#             # print(np.shape(v))
#             y = np.squeeze(np.array([v, self.hh.n_inf(v)], dtype= 'float64'))
#         # print(y)

#         ax = plt.subplot()
#         ax.plot(bigt, bigy)
#         ax.set_xlabel('Time (ms)')
#         ax.set_ylabel('Membrane potential (mV)')
#         ax.set_title('Neuron potential')
#         plt.grid()
#         plt.savefig('Figures/Neuron Potential Full model')
#         plt.show()
 

# if __name__ == '__main__':
#     runner = FullModel()
#     runner.Main()
#     print("--- %s seconds ---" % (time.time() - start_time))