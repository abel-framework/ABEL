import numpy as np
from scipy.integrate import odeint
from opal.utilities import SI
from opal.utilities.plasmaphysics import *
from matplotlib import pyplot as plt


# Lu equation (from Lebedev 2017 paper): r1' = r2, r2' = 2*(r2)^2/r1 + 1 - 2*dNs_dz(z)/(pi*n0*r1^3)
def ode_Lu(kr, kz, dN_dkz, n0):
    return kr[1], -2*kr[1]**2/kr[0] - 1/kr[0] - 2*k_p(n0)**3*dN_dkz(kz)/(np.pi*n0*kr[0]**3)


# calculate the evolution of radius the sheath electrons
def wakefield_Lu(n0, kRb, beam=None):
        
    # normalized longitudinal number density
    if beam is not None:
        dN_dz, zs = beam.longitudinalNumberDensity()
        kzs0 = k_p(n0)*zs
    else:
        dN_dz = np.array([0, 0])
        kzs0 = np.array([-kRb, 0])
    dN_dkzFcn = lambda kz: np.interp(-kz, kzs0, dN_dz/k_p(n0), left=0, right=0)
    
    # prepare initial conditions (and flip the z direction)
    kr0 = np.array([kRb, 0])
    kpmax = max([max(-kzs0), kRb, 1])
    kpmin = min([0, min(-kzs0)])
    dkr = 1/500
    Nstep = round(kpmax/dkr)
    kzs = np.linspace(kpmin, kpmax, Nstep)
    
    # numerical integration
    sol = odeint(ode_Lu, kr0, kzs, args=(dN_dkzFcn, n0), mxstep=1000, mxhnil=1, hmin=dkr*1e-4, hmax=dkr*1e-1)
    krbs = sol[:,0]
    dkrb_dkzs = sol[:,1]
    
    # clean zero radii and extreme fields
    kzs_good = kzs[np.logical_and(np.logical_and(krbs > dkr, krbs < 100), abs(dkrb_dkzs) < 100)]
    mask = kzs <= max(kzs_good)
    kzs, krbs, dkrb_dkzs = kzs[mask], krbs[mask], dkrb_dkzs[mask]
    
    # convert to longitudinal wakefield
    Ezs = (SI.e * n0 / (2 * SI.eps0 * k_p(n0))) * krbs * dkrb_dkzs
    
    # return, unflipped and un-normalized
    return np.flip(Ezs), -np.flip(kzs/k_p(n0)), np.flip(krbs/k_p(n0))
