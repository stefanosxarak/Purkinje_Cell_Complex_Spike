from Units import *
import numpy as np
# from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from HH_model import *

start_time = time.time()
class Markov:                               # Current through the sodium channel is described using a Markovian Scheme

    def __init__(self,γ,δ,ε,d,u,n,f,a):    

        self.γ = γ       # m*s**(-1)
        self.δ = δ       # m*s**(-1)
        self.ε = ε       # m*s**(-1)
        self.d = d       # m*s**(-1)
        self.u = u       # m*s**(-1)
        self.n = n       # m*s**(-1)
        self.f = f       # m*s**(-1)
        self.a = a       # m*s**(-1)

    def alpha(self,v):

        return 150.* np.exp(v/20.)

    def beta(self,v):

        return 3. * np.exp(-v/20.)

    def ksi(self,v):

        return 0.03 * np.exp(-v/25.)

    def derivatives(self,y,α,β,ξ):

        c0 = y[0]
        c1 = y[1]
        c2 = y[2]
        c3 = y[3]
        c4 = y[4]
        i0 = y[5]
        i1 = y[6]
        i2 = y[7]
        i3 = y[8]
        i4 = y[9]
        i5 = y[10]
        o  = y[11]
        b  = y[12] 

        γ = self.γ     
        δ = self.δ       
        ε = self.ε       
        d = self.d       
        u = self.u       
        n = self.n       
        f = self.f       
        a = self.a
        aav=α*a
        bav=α/a

        der = np.zeros(13)
        
        der[0] = u * i0 + β*c1 - c0*4*α - d*c0                                       # dC1/dt
        der[1] = u/a*i1 + 2*β*c2+4*α*c0-(3*α+ β + d*a)*c1                            # dC2/dt
        der[2] = (u/(a**2)) *i2 +3*β *c3 + 3*α *c1-(2*α +2*β + d* a**2)*c2           # dC3/dt
        der[3] = (u/(a**3)) *i3+ 4*β *c4 +2*α *c2-(α + 3*β + d* a**3)*c3             # dC4/dt
        der[4] = (u/(a**4)) *i4+ δ*o+ α*c3 -(γ +4*β + d*a**4)*c4                     # dC5/dt
        der[5]  = d * c0 + 4*β/a *i1 - i0 *u - i0 *a*α                               # di1/dt
        der[6]  = d*a *c1 + 3*bav*i2 + aav*i0      - (u/a + 2*aav + 4*bav) *i1       # di2/dt
        der[7]  = d*a**2 *c2 + 2*bav*i3 + 2*aav*i1 - (u/a**2 + 3*aav + 3*bav) *i2    # di3/dt
        der[8]  = d*a**3 *c3 + bav*i4 + 3*aav*i2   - (u/a**3 + 4*aav + 2*bav) *i3    # di4/dt
        der[9]  = d*a**4 *c4 + δ*i5 + 4*aav*i3     - (u/a**4 + γ + bav) *i4          # di5/dt
        der[10] = n*o + γ*i4 - (f + δ) *i5                                           # dI6/dt
        der[11] = γ* c4+ ξ*b+ f*i5 - (δ + n + ε)*o                                   # do/dt
        der[12] = o * ε - b*ξ                                                        # db/dt

        return der

    # def mark_intgr(self):
        # v = self.hh.lala()
        # for i in range(0,len(v)):
        #     v[i] = 0
        # self.bigy = np.array([])
        # self.bigt = np.array([])
        # self.bigo = np.array([])
        # self.bigb = np.array([])
        # y = np.array([self.c0,self.c1,self.c2,self.c3,self.c4,self.i0,self.i1,self.i2,self.i3,self.i4,self.i5,self.o,self.b])

        # j=0 # time step being self.tmax/len(v)
        # for i in range(0,len(v)):
        #     markov = solve_ivp(self.derivatives, t_span=(j,j+self.tmax/len(v)), y0=y, method='BDF', args=(v[i], self.alpha(v[i]), self.beta(v[i]), self.ksi(v[i]), self.γ,self.δ,self.ε,self.d,self.u,self.n,self.f,self.a))
            
        #     # markov.y has shape (13,100) and y has shape (13,)
        #     # np.shape(markov.y[-1,:]) # (100,)

        #     self.bigo = np.concatenate((self.bigo,markov.y[11]))
        #     self.bigb = np.concatenate((self.bigb,markov.y[12]))        
        #     self.bigy = np.concatenate((self.bigy,markov.y[-1,:]))    
        #     self.bigt = np.concatenate((self.bigt,markov.t))          # last element of markov.t is the same with the first one from the next iteration

        #     #   Updating and normalising the y values
        #     y = markov.y[:,-1]
        #     y = y/np.sum(y)

        #     j+=self.tmax/len(v)
