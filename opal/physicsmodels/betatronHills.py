import numpy as np
from scipy.integrate import odeint
from opal.utilities import SI

# damped Hill's equation: x''(s) + gamma'(s)/gamma(s)*x'(s) - k_p^2/(2*gamma(s))*x(s) = 0
def ode_Hills(u, s, gamma, kp):
    x, wx = u
    return wx/(gamma(s)*SI.c), -x*(SI.c/2)*kp(s)**2

# solve Hill's equation
def evolveHillsEquation(x0, wx0, L, gamma, kp, fast=False):
    
    # find longitudinal steps
    if not fast:
        gamma_min = min(gamma(0), gamma(L))
        kp_max = max(kp(0), kp(L))
        beta_matched_max = np.sqrt(2*gamma_min)/kp_max
        res = 1/10
        Nstep = int(round(L/(beta_matched_max*res)))
        ss = np.linspace(0, L, Nstep)
        evolution = np.empty([3, len(ss)])
    else:
        ss = np.linspace(0, L, 2)
        evolution = None
    
    # numerical integration
    u0 = np.array([x0, wx0])
    sol = odeint(ode_Hills, u0, ss, args=(gamma, kp))
    x = sol[-1, 0]
    wx = sol[-1, 1]
    
    # save evolution
    if not fast:
        evolution[0,:] = ss
        evolution[1,:] = sol[:, 0]
        evolution[2,:] = sol[:, 1]
    
    return x, wx, evolution