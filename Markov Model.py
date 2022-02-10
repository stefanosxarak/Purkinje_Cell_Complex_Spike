from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Markov():

    def alpha(self,v):
        return 150* np.exp(v/20*mv)
    def beta(self,v):
        return 3 * np.exp(-v/20*mv) 
    def ksi(self,v):
        return 0.03 * np.exp(-v/25*mv)

    def derivatives(self,c,I,O,B):
        v = 0.
        γ = 150         # m*s**(-1)
        δ = 40          # m*s**(-1)
        ε = 1.75        # m*s**(-1)
        D = 0.005       # m*s**(-1)
        U = 0.5         # m*s**(-1)
        N = 0.75        # m*s**(-1)
        F = 0.005       # m*s**(-1)
        a = ((U/D)/(F/N))**(1/8)

        α = self.alpha(v)
        β = self.beta(v)
        ξ = self.ksi(v)

        der = np.zeros(13)
        
        der[0] = U * I[0] + β*c[1] - c[0]*4*α - D*c[0]                  # dC1/dt
        der[1] = U/a * I[1] + 2*β*c[2] - c[1]*3*α - D*a*c[1]            # dC2/dt
        der[2] = (U/(a**2)) * I[2] + 3*β*c[3] - c[2]*2*α - D*a**2*c[2]  # dC3/dt
        der[3] = (U/(a**3)) * I[3] + 4*β*c[4] - c[3]*α - D*a**3*c[3]    # dC4/dt
        der[4] = (U/(a**4)) * I[4] + δ*O - c[4]*γ - D*a**4*c[4]         # dC5/dt

        der[5]  = D * c[0] + 4*β/a *I[1] - I[0] *U - I[0] *a*α                #dI1/dt
        der[6]  = a* D * c[1] + 3*β/a *I[2] - I[1] *U/a - I[1] *a*α*2         #dI2/dt
        der[7]  = a**2* D * c[2] + 2*β/a *I[3] - I[2] *U/a**2 - I[2] *a*α*3   #dI3/dt
        der[8]  = a**3* D * c[3] + β/a *I[4] - I[3] *U/a**3 - I[3] *a*α*4     #dI4/dt
        der[9]  = a**4*D*c[4] + δ *I[5] - I[4]*U/a**4 - I[4]*γ                #dI5/dt
        der[10] = N*O - I[5]*F                                                #dI6/dt

        der[11] = ξ*B + F*I[5] - O*N - O*ε  # dO/dt
        der[12] = O * ε - B*ξ   # dB/dt

        print(der[0]+der[1]+der[5]) # has to be 0
        print(c[0]+c[1]+I[0])       # has to be 1

        return der

    def Main(self):
        t = self.t
        y = np.array([v, self.n_inf(v), self.m_inf(v), self.h_inf(v)], dtype= 'float64')
        # self.derivatives(c,I,O,B)
        markov = odeint(self.derivatives, y, t)    # Solve ODE

if __name__ == '__main__':
    runner = Markov()
    runner.Main()
