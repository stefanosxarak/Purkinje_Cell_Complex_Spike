from Units import *
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from HH_model import *

start_time = time.time()

# solve a 13 equation system drop 1 replace with np.ones for more accurate initialisation
class Markov():
    # Current through the resurgent sodium channel is described using a Markovian Scheme
    def __init__(self):
        self.v = -50.

        self.c0 = 1.
        self.c1 = 0.
        self.c2 = 0.
        self.c3 = 0.
        self.c4 = 0.
        self.I0 = 0.
        self.I1 = 0.
        self.I2 = 0.
        self.I3 = 0.
        self.I4 = 0.
        self.I5 = 0.
        self.o = 0.
        self.b = 0.

        self.γ = 150.        # m*s**(-1)
        self.δ = 40.         # m*s**(-1)
        self.ε = 1.75        # m*s**(-1)
        self.d = 0.005       # m*s**(-1)
        self.u = 0.5         # m*s**(-1)
        self.n = 0.75        # m*s**(-1)
        self.f = 0.005       # m*s**(-1)
        self.a = ((self.u/self.d)/(self.f/self.n))**(1/8)   # m*s**(-1)
        self.hh = HodgkinHuxley() 

        self.tmax = 35 
        # self.t  = np.linspace(0, self.tmax, 100)
        # self.scheme(self.v)


    def error(self,accepted,experiment):
        if accepted == 0 :
            err = np.abs(100 -(experiment - accepted)/(accepted+1) *100)
            print("The error percentage is:", err,"%")
        else:
            err = np.abs((experiment - accepted)/(accepted) *100)
            print("The error percentage is:", err,"%")

    def alpha(self,v):
        return 150.* np.exp(v/20.)
    def beta(self,v):
        return 3. * np.exp(-v/20.) 
    def ksi(self,v):
        return 0.03 * np.exp(-v/25.)

    def derivatives(self,t,y ,v,α,β,ξ,γ,δ,ε,d,u,n,f,a):
        c0 = y[0]
        c1 = y[1]
        c2 = y[2]
        c3 = y[3]
        c4 = y[4]
        I0 = y[5]
        I1 = y[6]
        I2 = y[7]
        I3 = y[8]
        I4 = y[9]
        I5 = y[10]
        o  = y[11]
        b  = y[12] 

        aav=α*a
        bav=α/a

        der = np.zeros(13)
        
        der[0] = u * I0 + β*c1 - c0*4*α - d*c0                                       # dC1/dt
        der[1] = u/a*I1 + 2*β*c2+4*α*c0-(3*α+ β + d*a)*c1                            # dC2/dt
        der[2] = (u/(a**2)) *I2 +3*β *c3 + 3*α *c1-(2*α +2*β + d* a**2)*c2           # dC3/dt
        der[3] = (u/(a**3)) *I3+ 4*β *c4 +2*α *c2-(α + 3*β + d* a**3)*c3             # dC4/dt
        der[4] = (u/(a**4)) *I4+ δ*o+ α*c3 -(γ +4*β + d*a**4)*c4                     # dC5/dt

        der[5]  = d * c0 + 4*β/a *I1 - I0 *u - I0 *a*α                               # dI1/dt
        der[6]  = d*a *c1 + 3*bav*I2 + aav*I0      - (u/a + 2*aav + 4*bav) *I1       # dI2/dt
        der[7]  = d*a**2 *c2 + 2*bav*I3 + 2*aav*I1 - (u/a**2 + 3*aav + 3*bav) *I2    # dI3/dt
        der[8]  = d*a**3 *c3 + bav*I4 + 3*aav*I2   - (u/a**3 + 4*aav + 2*bav) *I3    # dI4/dt
        der[9]  = d*a**4 *c4 + δ*I5 + 4*aav*I3     - (u/a**4 + γ + bav) *I4          # dI5/dt
        der[10] = n*o + γ*I4 - (f + δ) *I5                                           # dI6/dt

        der[11] = γ* c4+ ξ*b+ f*I5 - (δ + n + ε)*o                                   # do/dt
        der[12] = o * ε - b*ξ                                                        # db/dt

        return der

    def Main(self):
        v = self.hh.lala()
        for i in range(0,len(v)):
            y = np.array([self.c0,self.c1,self.c2,self.c3,self.c4,self.I0,self.I1,self.I2,self.I3,self.I4,self.I5,self.o,self.b])
            self.bigy = np.array([])
            self.bigt = np.array([])
            self.bigo = np.array([])

            for j in range(0,self.tmax):
                markov = solve_ivp(self.derivatives, t_span=(j,j+1), y0=y, method='BDF',t_eval=np.linspace(j, j+1, 100), args=(v[i], self.alpha(v[i]), self.beta(v[i]), self.ksi(v[i]), self.γ,self.δ,self.ε,self.d,self.u,self.n,self.f,self.a))

                # markov.y has shape (13,100) and y has shape (13,)
                # np.shape(markov.y[-1,:]) # (100,)

                self.bigo = np.concatenate((self.bigo,markov.y[11]))    
                self.bigy = np.concatenate((self.bigy,markov.y[-1,:]))    
                self.bigt = np.concatenate((self.bigt,markov.t))          # last element of markov.t is the same with the first one from the next iteration

                #   Updating and normalising the y 
                y = markov.y[:,-1]
                y = y/np.sum(y)


        ax = plt.subplot()
        ax.plot(self.bigt, self.bigo)
        ax.set_xlabel('Time (ms)')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    runner = Markov()
    runner.Main()
    print("--- %s seconds ---" % (time.time() - start_time))