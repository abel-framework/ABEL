import numpy as np
import scipy.integrate as sciint
import scipy.interpolate as sciinterp
import scipy.constants as SI
from opal.utilities.plasma_physics import k_p


# calculate the longitudinal electric field
def wakefield_1d(n0, driver, beam=None):

    # make cylindrical beam density (density vs. r and z)
    Nbins_r = round(np.sqrt(len(driver))/5)
    Nbins_z = round(np.sqrt(len(driver))/5)
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
    rhos_interp = sciinterp.RegularGridInterpolator((krs_ctrs, kzs_ctrs), rhos, fill_value=0, bounds_error=False)
    
    # interpolate beam density
    kr_span = lambda kr: np.linspace(0, kr, 100).T
    alephFcn = lambda kr, kz: -2*(np.trapz(kr_span(kr) * \
               rhos_interp(np.array([kr_span(kr), kz*np.ones(kr_span(kr).shape)]).T), x=kr_span(kr), axis=0).T / kr**2).T
    
    # Golovanov equation (from Golovanov 2021 PPCF paper)
    def ode_Golovanov(psi, kz):
        return psi[1], -psi[1]**2/(2*psi[0]) + alephFcn(2*np.sqrt(psi[0]), kz) - 1/2
    
    # find z axis for integration (starts at a given current treshold)
    rhos_onaxis = rhos_interp(np.array([np.zeros(kzs_ctrs.shape).T, kzs_ctrs.T]).T)
    rho_threshold = 1
    kzs = np.arange(min(kzs_ctrs[abs(rhos_onaxis) > rho_threshold]), max(kzs_ctrs), dkz/20)
    
    # numerical integration (driver)
    psi0 = np.array([1e-3, 0])
    sol = sciint.odeint(ode_Golovanov, psi0, kzs, mxhnil=1, mxstep=1000)
    psi = sol[:, 0]
    dpsi_dkz = sol[:, 1]
    krbs = 2*np.sqrt(psi)
    
    # add witness bunch to current profile (use Lu equation)
    if beam is not None:
        
        # get beam current profile
        bins = beam.z_offset() + 5*np.linspace(-1, 1, int(np.sqrt(len(beam))))*beam.bunch_length()
        dN_dz, zs_beam = beam.longitudinal_num_density(bins=bins)
        kzs_beam0 = k_p(n0)*zs_beam
        dN_dkzFcn = lambda kz: np.interp(-kz, kzs_beam0, dN_dz/k_p(n0), left=0, right=0)
        
        # Lu equation (from Lu et al. PRL 2016)
        def ode_Lu(kz, kr):
            return kr[1], -2*kr[1]**2/kr[0] - 1/kr[0] - 2*k_p(n0)**3*dN_dkzFcn(kz)/(np.pi*n0*kr[0]**3)
        
        # numerical integration with a stiff solver (BDF or Radau)
        kr0 = [krbs[-1], dpsi_dkz[-1]/np.sqrt(psi[-1])]
        dkz_beam = np.median(np.diff(kzs_beam0))
        kzs_eval = np.arange(max(kzs), max(-kzs_beam0)+0.5, dkz_beam)
        sol_beam = sciint.solve_ivp(ode_Lu, [min(kzs_eval), max(kzs_eval)], kr0, method='Radau', vectorized=True, t_eval=kzs_eval)
        kzs_beam = sol_beam.t[1:]
        krbs_beam = sol_beam.y[0,1:]
        dkrb_dkzs_beam = sol_beam.y[1,1:]
        dpsi_dkz_beam = krbs_beam*dkrb_dkzs_beam/2
        
        # append to the driver solution
        kzs = np.append(kzs, kzs_beam)
        krbs = np.append(krbs, krbs_beam)
        dpsi_dkz = np.append(dpsi_dkz, dpsi_dkz_beam)
    
    # convert to longitudinal wakefield
    Ezs = (SI.e * n0 / (2 * SI.epsilon_0 * k_p(n0))) * dpsi_dkz
    
    # unflip the z direction
    return np.flip(Ezs), -np.flip(kzs/k_p(n0)), np.flip(krbs/k_p(n0))
    
    