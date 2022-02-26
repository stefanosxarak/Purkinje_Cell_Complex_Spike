from Units import *
import numpy as np
from scipy.integrate import odeint
import vorpy as vp      # vorpy.symplectic_integration

# 13 equations drop 1 replace with np.ones
# or add 1 to c1 and 0 to rest

class Markov():
    # Current through the resurgent sodium channel is described using a Markovian Scheme
    def __init__(self):
        c_init = np.zeros(5)
        I_init = np.zeros(6)
        c_init[0] = 1
        self.v = -70.
        self.c = c_init
        self.I = I_init
        self.o = 0
        self.b = 0
        self.γ = 150         # m*s**(-1)
        self.δ = 40          # m*s**(-1)
        self.ε = 1.75        # m*s**(-1)
        self.d = 0.005       # m*s**(-1)
        self.u = 0.5         # m*s**(-1)
        self.n = 0.75        # m*s**(-1)
        self.f = 0.005       # m*s**(-1)
        self.a = ((self.u/self.d)/(self.f/self.n))**(1/8)

        self.tmin = 0.
        self.tmax = 35.    
        self.t  = np.linspace(self.tmin, self.tmax, 1000) # Total duration of simulation [s]

    def error(self,accepted,experiment):
        new_exp = 0
        err = np.abs((experiment - accepted)/(accepted+1)) *100
        print("The error percentage is:", err)
        new_exp = experiment/experiment
        print("The final value is: ",new_exp)
        if err==0 :
            new_exp = experiment/experiment
            print("The final value is: ",new_exp)
        return new_exp

    def alpha(self,v):
        return 150* np.exp(v/20)
    def beta(self,v):
        return 3 * np.exp(-v/20) 
    def ksi(self,v):
        return 0.03 * np.exp(-v/25)

    def derivatives(self,y,t,c,I,o):
        α = y[1]
        β = y[2]
        ξ = y[3]

        γ = y[4]        # m*s**(-1)
        δ = y[5]        # m*s**(-1)
        ε = y[6]        # m*s**(-1)
        d = y[7]        # m*s**(-1)
        u = y[8]        # m*s**(-1)
        n = y[9]        # m*s**(-1)
        f = y[10]       # m*s**(-1)
        a = y[11]

        b = y[12]       # This is B channel after O channel

        aav=α*a
        bav=α/a
        der = np.zeros(13)
        
        der[0] = u * I[0] + β*c[1] - c[0]*4*α - d*c[0]                                       # dC1/dt
        der[1] = u/a*I[1] + 2*β*c[2]+4*α*c[0]-(3*α+ β + d*a)*c[1]                            # dC2/dt
        der[2] = (u/(a**2)) *I[2] +3*β *c[3] + 3*α *c[1]-(2*α +2*β + d* a**2)*c[2]           # dC3/dt
        der[3] = (u/(a**3)) *I[3]+ 4*β *c[4] +2*α *c[2]-(α + 3*β + d* a**3)*c[3]             # dC4/dt
        der[4] = (u/(a**4)) *I[4]+ δ*o+ α*c[3] -(γ +4*β + d*a**4)*c[4]                       # dC5/dt

        der[5]  = d * c[0] + 4*β/a *I[1] - I[0] *u - I[0] *a*α                               #dI1/dt
        der[6]  = d*a *c[1] + 3*bav*I[2] + aav*I[0]      - (u/a + 2*aav + 4*bav) *I[1]       #dI2/dt
        der[7]  = d*a**2 *c[2] + 2*bav*I[3] + 2*aav*I[1] - (u/a**2 + 3*aav + 3*bav) *I[2]    #dI3/dt
        der[8]  = d*a**3 *c[3] + bav*I[4] + 3*aav*I[2]   - (u/a**3 + 4*aav + 2*bav) *I[3]    #dI4/dt
        der[9]  = d*a**4 *c[4] + δ*I[5] + 4*aav*I[3]     - (u/a**4 + γ + bav) *I[4]          #dI5/dt
        der[10] = n*o + γ*I[4] - (f + δ) *I[5]                                               #dI6/dt

        der[11] = γ* c[4]+ ξ*b+ f*I[5] - (δ + n + ε)*o                                       # do/dt
        der[12] = o * ε - b*ξ                                                                # db/dt

        # result = self.error(1, (sum(c) + sum(I) + o + b))
        # print(sum(c) + sum(I) + o + b)  # equals 1
        print(sum(der))
        # result = self.error(0, (sum(der)))

        return der

    def Main(self):
        v = self.v
        y = np.array([v, self.alpha(v), self.beta(v), self.ksi(v), self.γ,self.δ,self.ε,self.d,self.u,self.n,self.f,self.a, self.b], dtype= 'float64')
        c = self.c
        I = self.I
        o = self.o
        sam = 0
        markov = odeint(self.derivatives, y, self.t, args=(c,I,o))    #(1000,13)
        # print(np.sum(markov))
        # for i in range(13):
        #     for j in range(1000):
        #         sam += markov[j][i]
        for i in range(13):
                sam += markov[15][i]
        # print(sam)
        # result = self.error(0, (sam))

if __name__ == '__main__':
    runner = Markov()
    runner.Main()
