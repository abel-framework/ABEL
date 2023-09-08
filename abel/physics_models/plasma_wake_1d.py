import numpy as np
import scipy.integrate as sciint
import scipy.interpolate as sciinterp
import scipy.constants as SI
from abel.utilities.plasma_physics import k_p


# calculate the longitudinal electric field
def wakefield_1d(n0, driver, beam=None):

    # make cylindrical beam density (density vs. r and z)
    bins_r = 5 * np.std(driver.rs()) * np.linspace(0, 1, round(np.sqrt(len(driver))/4))
    bins_z = -driver.z_offset() + 4 * driver.bunch_length() * np.linspace(-1, 0, round(np.sqrt(len(driver))/10))
    rhos_r, rs_edges, zs_edges = np.histogram2d(driver.rs(), -driver.zs(), bins=(bins_r,bins_z), weights=driver.qs())
    kzs_ctrs = k_p(n0)*(zs_edges[:-1]+zs_edges[1:])/2
    krs_ctrs = k_p(n0)*(rs_edges[:-1]+rs_edges[1:])/2
    
    # normalize correctly
    dkr = np.median(np.diff(krs_ctrs))
    dkz = np.median(np.diff(kzs_ctrs))
    rhos_kr = rhos_r * k_p(n0)**3/(2*np.pi*dkz*dkr*n0*SI.e) # TODO: understand factor 2pi
    rhos = (rhos_kr.T / krs_ctrs).T

    # interpolate beam density (enforce correct on-axis beam density)
    krs_ctrs = np.concatenate(([0],krs_ctrs))
    rhos = np.concatenate(([rhos[0,:]], rhos), axis=0)
    rhos_interp = sciinterp.RegularGridInterpolator((krs_ctrs, kzs_ctrs), rhos, fill_value=0, bounds_error=False)
    
    # interpolate beam density
    kr_span = lambda kr: np.linspace(0, kr, 200).T
    alephFcn = lambda kr, kz: -2*(np.trapz(kr_span(kr) * rhos_interp(np.array([kr_span(kr), kz*np.ones(kr_span(kr).shape)]).T), x=kr_span(kr), axis=0).T / kr**2).T
    
    # Golovanov equation (from Golovanov 2021 PPCF)
    def ode_Golovanov_excite(psi, kz):
        return psi[1], -psi[1]**2/(2*psi[0]) + alephFcn(2*np.sqrt(psi[0]), kz) - 1/2
    
    # find z axis for integration (starts at a given current treshold)
    rhos_onaxis = rhos_interp(np.array([np.zeros(kzs_ctrs.shape).T, kzs_ctrs.T]).T)
    rho_threshold = 0.5
    kzs = np.arange(min(kzs_ctrs[abs(rhos_onaxis) > rho_threshold]), max(kzs_ctrs), dkz/40)
    
    # numerical integration (driver)
    psi0 = np.array([1e-3, 0])
    sol = sciint.odeint(ode_Golovanov_excite, psi0, kzs, mxhnil=0, mxstep=1000)
    psi = sol[:, 0]
    dpsi_dkz = sol[:, 1]
    krbs = 2*np.sqrt(psi)
    
    # add witness bunch to current profile (use Lu equation)
    if beam is not None:
        
        # get beam current profile
        bins = np.arange(beam.z_offset()-5*beam.bunch_length(), driver.z_offset()+5*driver.bunch_length(), 5*beam.bunch_length()/np.sqrt(len(beam)))
        dN_dz, zs_beam = (beam + driver).longitudinal_num_density(bins=bins)
        kzs_beam0 = k_p(n0)*zs_beam
        dN_dkzFcn = lambda kz: np.interp(-kz, kzs_beam0, dN_dz/k_p(n0), left=0, right=0)
        
        # Golovanov equation (from Golovanov et al. PRL 2023)
        def ode_Golovanov(kz, kr):
            A = kr[0]**3/4 + kr[0]
            B = kr[0]**2/2 + 1
            C = kr[0]**2/4
            return kr[1], -(B/A)*kr[1]**2 - C/A - k_p(n0)**3*dN_dkzFcn(kz)/(2*np.pi*n0*A)

        # find transition (at driver centroid)
        ind_transition = np.abs(kzs + k_p(n0)*driver.z_offset()).argmin()
        
        # numerical integration with a stiff solver (BDF or Radau)
        kr0 = [krbs[ind_transition], dpsi_dkz[ind_transition]/np.sqrt(psi[ind_transition])]
        dkz_beam = np.median(np.diff(kzs_beam0))
        kzs_eval = np.arange(kzs[ind_transition], max(-kzs_beam0)+0.5, dkz_beam)
        sol_beam = sciint.solve_ivp(ode_Golovanov, [min(kzs_eval), max(kzs_eval)], kr0, method='Radau', vectorized=True, t_eval=kzs_eval)
        kzs_beam = sol_beam.t[1:]
        krbs_beam = sol_beam.y[0,1:]
        dkrb_dkzs_beam = sol_beam.y[1,1:]
        dpsi_dkz_beam = krbs_beam*dkrb_dkzs_beam/2
        
        # append to the driver solution
        mask = kzs < min(kzs_beam)
        kzs = np.append(kzs[mask], kzs_beam)
        krbs = np.append(krbs[mask], krbs_beam)
        dpsi_dkz = np.append(dpsi_dkz[mask], dpsi_dkz_beam)
    
    # convert to longitudinal wakefield
    Ezs = (SI.e * n0 / (SI.epsilon_0 * k_p(n0))) * dpsi_dkz
    
    # unflip the z direction
    return np.flip(Ezs), -np.flip(kzs/k_p(n0)), np.flip(krbs/k_p(n0))

