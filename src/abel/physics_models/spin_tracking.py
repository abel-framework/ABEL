from abel import *
import numpy as np
import scipy.constants as SI
import matplotlib.pyplot as plt
from abel.physics_models import *
from abel.utilities.relativity import *
import scipy.special as scispec

SI.r_e = SI.physical_constants['classical electron radius'][0]

"""
Simulates particle motion and tracks spin motion using the T-BMT equation
"""

def plasma_E_field(x, y, k_p, Ez=0.0):
    Er_coeff = SI.m_e * (k_p**2) * SI.c**2 / (2*SI.e)
    Ex = Er_coeff * x
    Ey = Er_coeff * y
    return np.array([Ex, Ey, Ez], dtype=float)


def plasma_B_field(x, y, k_p, beta_z=1.0, Bz=0.0):
    Br_coeff = (SI.m_e * k_p**2 * SI.c / (2*SI.e)) * beta_z
    Bx = -Br_coeff * y
    By =  Br_coeff * x
    Bz = np.ones_like(x) * Bz
    return np.array([Bx, By, Bz], dtype=float)


def TBMT_spin_evolution(sxs0, sys0, szs0, k_p, ss, xs_all, ys_all, uxs_all, uys_all, gammas_all):

    # make spin vector
    S = np.array([sxs0, sys0, szs0])
    
    # pre-calculate betas
    betaxs_all = gamma2beta(proper_velocity2gamma(uxs_all))
    betays_all = gamma2beta(proper_velocity2gamma(uys_all))
    betazs_all = gamma2beta(gammas_all)

    for i, s in enumerate(ss):
        
        # skip the last step
        if i == len(ss)-1:
            break

        # calculate time step
        ds = (ss[i+1]-ss[i])
        dt = ds/SI.c

        # average gamma for this and next step
        gammas_step = (gammas_all[:,i+1]+gammas_all[:,i])/2
        Ezs_step = (gammas_all[:,i+1]-gammas_all[:,i])/ds*(SI.m_e*SI.c**2/SI.e)
        
        # find the beta (velocity) vector
        betaxs_step = (betaxs_all[:,i+1]+betaxs_all[:,i])/2
        betays_step = (betays_all[:,i+1]+betays_all[:,i])/2
        betazs_step = (betazs_all[:,i+1]+betazs_all[:,i])/2
        beta_vector_step = np.array([betaxs_step, betays_step, betazs_step])
        
        # find the E and B fields
        xs_step = (xs_all[:,i+1]+xs_all[:,i])/2
        ys_step = (ys_all[:,i+1]+ys_all[:,i])/2
        E_step = plasma_E_field(xs_step, ys_step, k_p, Ez=Ezs_step)
        B_step = plasma_B_field(xs_step, ys_step, k_p, beta_z=betazs_step)
        
        # update the spin vector
        S = TBMT_timestep(S, E_step, B_step, beta_vector_step, gammas_step, dt)
    
    # back to individual components
    sxs = S[0,:]
    sys = S[1,:]
    szs = S[2,:]
    
    return sxs, sys, szs 


def TBMT_timestep(S, E, B, beta_vec, gamma, dt):

    # use electrons
    q = -SI.e

    # anomalous magnetic moment
    a = 0.00115965218128

    # Thomas-BMT equation
    Omega_B = (a + 1.0 / gamma) * B
    Omega_E = - (a + 1.0/(gamma+1.0)) * np.linalg.cross(beta_vec, E, axis=0) / SI.c
    Omega_v = - (a*gamma/(gamma + 1.0)) * np.linalg.vecdot(beta_vec, B, axis=0) * beta_vec
    Omega = - (q / SI.m_e) * (Omega_B + 0*Omega_E + 0*Omega_v)
    
    # normalization
    w = np.linalg.norm(Omega, axis=0)
    
    # return same if no step or zero change
    if w.all() == 0 or dt == 0:
        return S.copy()

    # normalized vector
    n = Omega / w

    # incremental rotation angle
    theta = w * dt
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Rodrigues rotation of S about axis n by angle theta
    S_rot = S * c + np.linalg.cross(n, S, axis=0) * s + n * np.linalg.vecdot(n, S, axis=0) * (1 - c)
    
    return S_rot

        