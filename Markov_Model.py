from Units import *
import numpy as np
from scipy.integrate import odeint

class Markov():
    # Current through the resurgent sodium channel is described using a Markovian Scheme
    def __init__(self):
        self.v = 0.
        self.tmin = 0.
        self.tmax = 35.    
        self.t  = np.linspace(self.tmin, self.tmax, 1000) # Total duration of simulation [s]

    def error(self,accepted,experiment):
        err = (experiment - accepted)/accepted *100
        print("The error percentage is:", err)
        if err!=0 :
            new_exp = experiment/experiment
            print("The error was minimised by dividing with:",experiment)

    def alpha(self,v):
        return 150* np.exp(v/20*mv)
    def beta(self,v):
        return 3 * np.exp(-v/20*mv) 
    def ksi(self,v):
        return 0.03 * np.exp(-v/25*mv)

    def derivatives(self,y,t,c,I,O,B):
        γ = 150         # m*s**(-1)
        δ = 40          # m*s**(-1)
        ε = 1.75        # m*s**(-1)
        d = 0.005       # m*s**(-1)
        u = 0.5         # m*s**(-1)
        n = 0.75        # m*s**(-1)
        f = 0.005       # m*s**(-1)
        a = ((u/d)/(f/n))**(1/8)

        α = y[1]
        β = y[2]
        ξ = y[3]

        der = np.zeros(13)
        
        der[0] = u * I[0] + β*c[1] - c[0]*4*α - d*c[0]                  # dC1/dt
        der[1] = u/a * I[1] + 2*β*c[2] - c[1]*3*α - d*a*c[1]            # dC2/dt
        der[2] = (u/(a**2)) * I[2] + 3*β*c[3] - c[2]*2*α - d*a**2*c[2]  # dC3/dt
        der[3] = (u/(a**3)) * I[3] + 4*β*c[4] - c[3]*α - d*a**3*c[3]    # dC4/dt
        der[4] = (u/(a**4)) * I[4] + δ*O - c[4]*γ - d*a**4*c[4]         # dC5/dt

        der[5]  = d * c[0] + 4*β/a *I[1] - I[0] *u - I[0] *a*α                #dI1/dt
        der[6]  = a* d * c[1] + 3*β/a *I[2] - I[1] *u/a - I[1] *a*α*2         #dI2/dt
        der[7]  = a**2* d * c[2] + 2*β/a *I[3] - I[2] *u/a**2 - I[2] *a*α*3   #dI3/dt
        der[8]  = a**3* d * c[3] + β/a *I[4] - I[3] *u/a**3 - I[3] *a*α*4     #dI4/dt
        der[9]  = a**4*d*c[4] + δ *I[5] - I[4]*u/a**4 - I[4]*γ                #dI5/dt
        der[10] = n*O - I[5]*f                                                #dI6/dt

        der[11] = ξ*B + f*I[5] - O*n - O*ε  # dO/dt
        der[12] = O * ε - B*ξ   # dB/dt

        self.error(1, (sum(c) + sum(I) + O + B))
        return der

    def Main(self):
        v = self.v
        y = np.array([v, self.alpha(v), self.beta(v), self.ksi(v)], dtype= 'float64')

        markov = odeint(self.derivatives, y, self.t, args=(c,I,O,B))    # Solve ODE
        self.error(0, (sum(markov[0,5]) + sum(markov[5,11]) + markov[-2:]))
 
if __name__ == '__main__':
    runner = Markov()
    runner.Main()
