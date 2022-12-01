import numpy as np
from scipy.integrate import odeint
from opal.utilities import SI
from opal.utilities.plasmaphysics import *


# Lu equation (from Lebedev 2017 paper): r1' = r2, r2' = 2*(r2)^2/r1 + 1 - 2*dNs_dz(z)/(pi*n0*r1^3)
def ode_Lu(kr, kz, dN_dkz, n0):
    return kr[1], -2*kr[1]**2/kr[0] - 1/kr[0] - 2*k_p(n0)**3*dN_dkz(kz)/(np.pi*n0*kr[0]**3)


# calculate the evolution of radius the sheath electrons
def sheathRadius_Lu(n0, kRb, beam=None):
        
    # normalized longitudinal number density
    if beam is not None:
        dN_dz, zs = beam.longitudinalNumberDensity()
        kzs0 = k_p(n0)*zs
    else:
        dN_dz = np.array([0, 0])
        kzs0 = np.array([-kRb, 0])
        
    dN_dkzFcn = lambda kz: np.interp(-kz, kzs0, dN_dz/k_p(n0), left=0, right=0)
    
    # prepare initial conditions
    kr0 = np.array([kRb, 0])
    resolution = 1/500
    
    # flip the z direction
    kpmax = max([max(-kzs0), kRb])
    Nstep = round(kpmax/resolution)
    kzs = np.linspace(0, kpmax, Nstep)
    
    # numerical integration
    sol = odeint(ode_Lu, kr0, kzs, args=(dN_dkzFcn, n0))
    krs = sol[:, 0]
    dkr_dkzs = sol[:, 1]
    
    # clean zero radii
    kzs_not = np.delete(kzs, np.where(np.logical_and(krs > resolution, krs < kRb*2)))
    mask = kzs > min(kzs_not)
    kzs = np.delete(kzs, np.where(mask))
    krs = np.delete(krs, np.where(mask))
    dkr_dkzs = np.delete(dkr_dkzs, np.where(mask))
    
    # unflip the z direction
    return np.flip(krs), -np.flip(kzs), np.flip(dkr_dkzs)


# calculate the longitudinal electric field
def wakefield_Lu(n0, kRb, beam=None):
    
    # calculate sheath radius evolution
    krs, kzs, dkr_dkzs = sheathRadius_Lu(n0, kRb, beam)

    # convert to longitudinal wakefield
    Ezs = (SI.e * n0 / (2 * SI.eps0 * k_p(n0))) * krs * dkr_dkzs
    
    # clean unphysically large fields
    mask = abs(Ezs) > Ez_wavebreaking(n0)*kRb*2
    kzs = np.delete(kzs, np.where(mask))
    krs = np.delete(krs, np.where(mask))
    Ezs = np.delete(Ezs, np.where(mask))
    
    # un-normalize the spatial dimensions
    zs = kzs/k_p(n0)
    rs = krs/k_p(n0)
    
    return Ezs, zs, rs
    
    
    
    
    