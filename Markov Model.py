from Units import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Markov():
    γ = 150         # m*s**(-1)
    δ = 40          # m*s**(-1)
    ε = 1.75        # m*s**(-1)
    D = 0.005       # m*s**(-1)
    U = 0.5         # m*s**(-1)
    N = 0.75        # m*s**(-1)
    F = 0.005       # m*s**(-1)

    α = 150* np.exp(v/20*mv)
    β = 3 * np.exp(-v/20*mv) 
    ξ = 0.03 * np.exp(-v/25*mv)
    a = ((U/D)/(F/N))**(1/8)

    def derivatives():
        der = np.zeros(13)
        
        der[0] =
        der[1] =
        der[2] =
        der[3] =
        der[4] =
        der[5] =
        der[6] =
        der[7] =
        der[8] =
        der[9] =
        der[10] =
        der[11] =
        der[12] =

    def Main(self):
if __name__ == '__main__':
    runner = Markov()
    runner.Main()
