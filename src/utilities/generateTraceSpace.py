import numpy as np

def generateTraceSpace(epsilon, beta, alpha, N):

    gamma = (1 + alpha**2) / beta; # Twiss gamma
    
    sigx = np.sqrt(beta * epsilon); # rms beam size
    sigxp = np.sqrt(gamma * epsilon); # rms divergence
    rho = - alpha / np.sqrt(1 + alpha**2); # correlation
    
    us = np.random.normal(size = N) # Gaussian random variable 1
    vs = np.random.normal(size = N) # Gaussian random variable 2
    
    xs = sigx*us; # particle positions
    xps = sigxp*us*rho + sigxp*vs*np.sqrt(1 - rho**2); # particle angles
    
    return xs, xps