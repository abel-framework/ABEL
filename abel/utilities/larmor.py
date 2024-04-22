import numpy as np
import scipy.constants as SI
import abel.utilities.plasma_physics as pp

const = SI.e**2 / 6 / np.pi / SI.epsilon_0 / SI.c

def larmor_formula(s, xp, yp, gammas):
    dz = s[1] - s[0]
    n = s.size
    z = np.zeros(n-1)
    d_beta_x = np.zeros(n-1)
    d_beta_y = np.zeros(n-1)
    gamma = np.zeros(n-1)

    for i in range(n-1):
        gamma[i] = 1/2 * (gammas[i+1] + gammas[i])
        z[i] = 1/2 * (s[i+1] + s[i])
        d_beta_x[i] = (xp[i+1] - xp[i])/dz
        d_beta_y[i] = (yp[i+1] - yp[i])/dz

    P = 1/SI.c * const * np.power(gamma, 4) * SI.c**2 * (np.power(d_beta_x, 2) + np.power(d_beta_y, 2)) # [J/m] 1/c transforms 1/s to 1/m

    return z, P

def mean_larmor_formula(s, n0, gammas, rx, ry):
    wp = np.sqrt(n0*SI.e**2/SI.m_e/SI.epsilon_0)
    kp = wp/SI.c
    k_beta = kp/np.sqrt(2*gammas)
    E_loss = SI.e**2*s/12/np.pi/SI.epsilon_0*(gammas*k_beta)**4*(rx**2+ry**2)

    return s, E_loss










    
