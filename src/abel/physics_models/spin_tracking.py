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


def evolve_hills_equation_analytic_evolution(x0, ux0, L, gamma0, dgamma_ds, kp=None, g=None, num_steps=None):

    # set equivalent plasma density from magnetic field gradient
    if kp is None:
        kp = np.sqrt(2 * g * SI.e / (SI.m_e * SI.c))

    # find number of steps
    if num_steps is None:
        beta_mat = np.sqrt(2*np.mean(gamma0))/kp
        num_steps = round(L/beta_mat*10)

    # define propagation axis
    s = np.linspace(0, L, num_steps).T

    # find initial angles
    xp0 = ux0 / gamma2proper_velocity(gamma0)

    # calculate energy evolution
    gamma = np.outer(gamma0, np.ones_like(s)) + np.outer(dgamma_ds, s)

    # calculate x evolution
    C = np.sqrt(2) * kp / dgamma_ds
    A0 = C * np.sqrt(gamma0)
    Di = (kp**2 * x0 * scispec.iv(1, A0 * 1j) + A0 * 1j * dgamma_ds * xp0 * scispec.iv(0, A0 * 1j))
    Dk = (kp**2 * x0 * scispec.kv(1, A0 * 1j) - A0 * 1j * dgamma_ds * xp0 * scispec.kv(0, A0 * 1j))
    E = kp**2 * (scispec.iv(1, A0 * 1j) * scispec.kv(0, A0 * 1j) +
                 scispec.iv(0, A0 * 1j) * scispec.kv(1, A0 * 1j)) * 1j
    A = np.outer(C, np.ones_like(s)) * np.sqrt(gamma)
    x = np.real(1j * (np.outer(Di,np.ones_like(s)) * scispec.kv(0, A * 1j) + np.outer(Dk,np.ones_like(s)) * scispec.iv(0, A * 1j)) / np.outer(E,np.ones_like(s)))

    # calculate momentum evolution
    K = dgamma_ds * C**2 / (2 * E)
    xp = -np.real( np.outer(K,np.ones_like(s)) / A *
                  (np.outer(Dk,np.ones_like(s)) * scispec.iv(1, A * 1j) - np.outer(Di,np.ones_like(s)) * scispec.kv(1, A * 1j)))
    ux = xp * gamma2proper_velocity(gamma)
    
    return x, ux, gamma, s


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

        