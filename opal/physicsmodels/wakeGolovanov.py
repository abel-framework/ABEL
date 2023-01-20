import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import RegularGridInterpolator
from opal.utilities import SI
from opal.utilities.plasmaphysics import *
from matplotlib import pyplot as plt


# Golovanov equation (from Golovanov 2021 PPCF paper)
def ode_Golovanov(psi, kz, alephFcn, n0):
    return psi[1], -psi[1]**2/(2*psi[0]) + alephFcn(2*np.sqrt(psi[0]), kz) - 1/2


# calculate the longitudinal electric field
def wakefield_Golovanov(n0, driver):

    # make cylindrical beam density (density vs. r and z)
    Nbins_r = round(np.sqrt(driver.Npart())/5)
    Nbins_z = round(np.sqrt(driver.Npart()))
    rhos_r, rs_edges, zs_edges = np.histogram2d(driver.rs(), -driver.zs(), bins=(Nbins_r,Nbins_z), weights=driver.qs())
    kzs_ctrs = k_p(n0)*(zs_edges[:-1]+zs_edges[1:])/2
    krs_ctrs = k_p(n0)*(rs_edges[:-1]+rs_edges[1:])/2
    krs_ctrs2 = krs_ctrs[1:]
    
    # normalize correctly
    dkr = k_p(n0)*np.median(np.diff(rs_edges))
    dkz = k_p(n0)*np.median(np.diff(zs_edges))
    rhos_kr = rhos_r * k_p(n0)**3/(2*np.pi*dkz*dkr*n0*SI.e) # TODO: understand factor 2pi
    rhos = (rhos_kr.T / krs_ctrs).T
    
    # interpolate beam density (enforce correct on-axis beam density)
    krs_ctrs = np.concatenate(([0],krs_ctrs))
    rhos = np.concatenate(([rhos[0,:]], rhos), axis=0)
    rhos_interp = RegularGridInterpolator((krs_ctrs, kzs_ctrs), rhos, fill_value=0, bounds_error=False)
    
    # interpolate beam density
    kr_span = lambda kr: np.linspace(0, kr, 100).T
    alephFcn = lambda kr, kz: -2*(np.trapz(kr_span(kr) * rhos_interp(np.array([kr_span(kr), kz*np.ones(kr_span(kr).shape)]).T), x=kr_span(kr), axis=0).T / kr**2).T
    
    # find z axis for integration (starts at a given current treshold)
    rhos_onaxis = rhos_interp(np.array([np.zeros(kzs_ctrs.shape).T, kzs_ctrs.T]).T)
    rho_threshold = 1
    kzs = np.arange(min(kzs_ctrs[abs(rhos_onaxis) > rho_threshold]), max(kzs_ctrs)+1, dkz)
    
    # numerical integration
    psi0 = np.array([1e-5, 0])
    sol = odeint(ode_Golovanov, psi0, kzs, args=(alephFcn, n0), mxhnil=1)
    psi = sol[:, 0]
    dpsi_dkz = sol[:, 1]
    krbs = 2*np.sqrt(psi)
    dkrb_dkzs = dpsi_dkz/np.sqrt(psi)
    
    # clean zero radii and extreme fields
    kzs_good = kzs[np.logical_and(np.logical_and(krbs > dkr, krbs < 100), abs(dkrb_dkzs) < 100)]
    mask = kzs <= max(kzs_good)
    kzs, krbs, dpsi_dkz = kzs[mask], krbs[mask], dpsi_dkz[mask]
    
    # convert to longitudinal wakefield
    Ezs = (SI.e * n0 / (2 * SI.eps0 * k_p(n0))) * dpsi_dkz
    
    # unflip the z direction
    return np.flip(Ezs), -np.flip(kzs/k_p(n0)), np.flip(krbs/k_p(n0))
    
    
    