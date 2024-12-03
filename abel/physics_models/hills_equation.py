import numpy as np
from scipy.integrate import odeint
import scipy.constants as SI
from abel.utilities.relativity import gamma2proper_velocity, gamma2energy
import scipy.special as scispec


# damped Hill's equation: x''(s) + gamma'(s)/gamma(s)*x'(s) + k_p^2/(2*gamma(s))*x(s) = 0
def ode_Hills(u, s, gamma, kp):
    x, ux = u
    return ux/(gamma(s)*SI.c), -x*(SI.c/2)*kp(s)**2


# solve Hill's equation
def evolve_hills_equation_ode(x0, ux0, L, gamma, kp):
    
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
def evolve_hills_equation_analytic(x0, ux0, L, gamma0, dgamma_ds, kp=None, g=None):
    
    # convert initial proper velocities to angles
    xp0 = ux0 / gamma2proper_velocity(gamma0)
    
    if dgamma_ds.any() == 0:
        
        # convert to g if not given
        if g is None:
            g = kp**2*SI.m_e*SI.c/(2*SI.e)

        # convert to focusing strength
        k = g*SI.c/gamma2energy(gamma0)

        # calculate evolution (pure sinusoids)
        if k.any() == 0:
            x = x0 + xp0*L
            xp = xp0
        else:
            x = np.real(x0*np.cos(k*L) + (xp0/k)*np.sin(k*L))
            xp = np.real(xp0*np.cos(k*L) - x0*k*np.sin(k*L))

        # no acceleration
        gamma = gamma0
        
    else:

        # convert to kp if not given
        if kp is None:
            kp = np.sqrt(2*g*SI.e/(SI.m_e*SI.c))
            
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