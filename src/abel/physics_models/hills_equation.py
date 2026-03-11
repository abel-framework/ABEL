# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import numpy as np
import scipy.constants as SI

# damped Hill's equation: x''(s) + gamma'(s)/gamma(s)*x'(s) + k_p^2/(2*gamma(s))*x(s) = 0
def ode_Hills(u, s, gamma, kp):
    x, ux = u
    return ux/(gamma(s)*SI.c), -x*(SI.c/2)*kp(s)**2


# solve Hill's equation
def evolve_hills_equation_ode(x0, ux0, L, gamma, kp):
    
    from scipy.integrate import odeint
    
    # find longitudinal steps
    gamma_min = min(gamma(0), gamma(L))
    kp_max = max(kp(0), kp(L))
    beta_matched_max = np.sqrt(2*gamma_min)/kp_max
    res = 1/10
    Nstep = min(10000, max(int(L/(beta_matched_max*res)),10))
    ss = np.linspace(0, L, Nstep)
    evolution = np.empty([3, len(ss)])
    
    # numerical integration
    u0 = np.array([x0, ux0])
    sol = odeint(ode_Hills, u0, ss, args=(gamma, kp))
    x = sol[-1,0]
    ux = sol[-1,1]
    
    return x, ux


# solve Hill's equation
def evolve_hills_equation_analytic(x0, ux0, L, gamma0, dgamma_ds, kp):
    """
    Solves Hill's equation in 1D homogeneous plasma with wavenumber kp. 
    Assuming positive ions and the charge of a physical particle in the 
    beam is -e.

    Parameters
    ----------
    x0 : [m] 1D float ndarray
        The initial positions of macroparticles.

    ux0 : [m/s] 1D float ndarray
        The initial proper velocities of macroparticles.

    L : [m] float
        The beam propagation distance.

    gamma0 : 1D float ndarray
        The initial Lorentz factors of the macroparticles.
    
    dgamma_ds : float or 1D float ndarray
        The average changes of Lorentz factor over distance ``L``.

    kp : [m^-1] float or 1D float ndarray
        Plasma wavenumber.


    Returns
    ----------
    x0 : [m] 1D float ndarray
        The final positions of macroparticles.

    ux0 : [m/s] 1D float ndarray
        The final proper velocities of macroparticles.
    """

    import scipy.special as scispec
    from abel.utilities.relativity import gamma2proper_velocity

    x = np.empty(len(x0))
    xp = np.empty(len(x0))
    
    # convert initial proper velocities to angles
    xp0 = ux0 / gamma2proper_velocity(gamma0)
    
    if np.all(np.asarray(dgamma_ds) == 0):  # True when ALL elements are 0. Handles both a float and an ndarray.

        # convert to focusing strength
        k = kp/np.sqrt(2*gamma0)

        # calculate evolution (drift or pure sinusoids)
        # special case for k ≈ 0. Drift only
        zero_indices = np.abs(k) == 0.0
        x[zero_indices] = x0[zero_indices] + xp0[zero_indices]*L
        xp[zero_indices] = xp0[zero_indices]

        # sinusoid
        x[~zero_indices] = np.real(x0[~zero_indices]*np.cos(k[~zero_indices]*L) + (xp0[~zero_indices]/k[~zero_indices])*np.sin(k[~zero_indices]*L))
        xp[~zero_indices] = np.real(xp0[~zero_indices]*np.cos(k[~zero_indices]*L) - x0[~zero_indices]*k[~zero_indices]*np.sin(k[~zero_indices]*L))

        # no acceleration
        gamma = gamma0
        
    else:
        # find final gamma factor
        gamma = gamma0 + dgamma_ds*L
        
        # calculate Bessel arguments
        C = np.sqrt(2)*kp/dgamma_ds
        A0 = C * np.sqrt(gamma0)
        A = C * np.sqrt(gamma)
        
        # calculate complex co-efficients
        Di = (kp**2*x0*scispec.iv(1,A0*1j) + A0*1j*dgamma_ds*xp0*scispec.iv(0,A0*1j))
        Dk = (kp**2*x0*scispec.kv(1,A0*1j) - A0*1j*dgamma_ds*xp0*scispec.kv(0,A0*1j))
        E = kp**2*(scispec.iv(1,A0*1j)*scispec.kv(0,A0*1j) + scispec.iv(0,A0*1j)*scispec.kv(1,A0*1j))*1j
        
        # calculate final positions and angles (imaginary part is zero)
        x = np.real(1j*(Di*scispec.kv(0,A*1j) + Dk*scispec.iv(0,A*1j))/E)
        xp = -np.real((dgamma_ds*C**2/(2*A*E))*(Dk*scispec.iv(1,A*1j) - Di*scispec.kv(1,A*1j)))
    
    # convert angles back to proper velocities
    ux = xp * gamma2proper_velocity(gamma)
    
    return x, ux