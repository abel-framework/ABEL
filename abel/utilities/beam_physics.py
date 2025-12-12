# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import numpy as np


# =============================================
def generate_trace_space(epsilon, beta, alpha, N, symmetrize=False):
    """
    Generate a 2D transverse trace space defined by geometric emittance 
    ``epsilon`` and beam Twiss parameters ``beta`` and ``alpha``.
    

    Parameters
    ----------
    epsilon : [m rad] float
        Geometric emittance of the beam in the plane of interest.

    beta : [m] float
        Beta function value at the sampling location. Determines beam size
        and correlation structure together with ``alpha``.

    alpha : float
        Twiss parameter representing the correlation between ``x`` and ``x'``.

    N : int
        Number of particles to generate. If ``symmetrize=True`` this represents
        the final number of returned samples.

    symmetrize : bool, optional
        If ``True``, generate only ``N/2`` unique samples and then mirror them
        symmetrically in phase space (zero mean). Defaults to ``False``.


    Returns
    -------
    xs : [m] 1D float ndarray
        Transverse particle positions.

    xps : [rad] 1D float ndarray
        Transverse particle angles sampled with correct correlation to ``xs``.
    """

    # calculate beam size, divergence and correlation
    sigx = np.sqrt(epsilon * beta)
    sigxp = np.sqrt(epsilon * (1 + alpha**2) / beta)
    rho = - alpha / np.sqrt(1 + alpha**2)

    # make underlying Gaussian variables
    if symmetrize:
        N_actual = round(N/2)
    else:
        N_actual = N
    us = np.random.normal(size=N_actual)
    vs = np.random.normal(size=N_actual)

    # particle positions and angles
    xs = sigx*us
    xps = sigxp*us*rho + sigxp*vs*np.sqrt(1 - rho**2)

    if symmetrize:
        xs = np.concatenate((xs, -xs))
        xps = np.concatenate((xps, -xps))
    
    return xs, xps


# =============================================
def generate_trace_space_xy(epsilon_x, beta_x, alpha_x, epsilon_y, beta_y, alpha_y, N, L=0, symmetrize=False):
    """
    Generate transverse trace-space samples in both x- and y-planes.

    The samples reproduce a correlated 4D Gaussian phase-space defined by
    the beam Twiss parameters in each plane and an optional canonical
    angular momentum correlation term ``L``. 


    Parameters
    ----------
    epsilon_x : [m rad] float
        Geometric emittance in the horizontal plane.

    beta_x : [m] float
        Horizontal beta function at the sampling location.

    alpha_x : float
        Horizontal Twiss alpha parameter.

    epsilon_y : [m rad] float
        Geometric emittance in the vertical plane.

    beta_y : [m] float
        Vertical beta function at the sampling location.

    alpha_y : float
        Vertical Twiss alpha parameter.

    N : int
        Number of macroparticles to generate. If ``symmetrize=True`` this
        represents the final number of returned samples.

    L : [m rad] float, optional
        Canonical angular momentum (x–y coupling). ``L = 0`` produces
        uncoupled transverse motion. Defaults to 0.

    symmetrize : bool, optional
        If ``True``, generate ``N/4`` unique samples and mirror them to
        enforce transverse symmetry (zero mean). Defaults to ``False``.


    Returns
    -------
    xs : [m] 1D float ndarray
        Transverse positions in the horizontal plane.

    xps : [rad] 1D float ndarray
        Transverse angles (slopes) in the horizontal plane.

    ys : [m] 1D float ndarray
        Transverse positions in the vertical plane.

    yps : [rad] 1D float ndarray
        Transverse angles (slopes) in the vertical plane.
    """

    # calculate beam size, divergence and correlation
    sigx = np.sqrt(epsilon_x * beta_x)
    sigy = np.sqrt(epsilon_y * beta_y)
    sigxp = np.sqrt(epsilon_x * (1 + alpha_x**2) / beta_x)
    sigyp = np.sqrt(epsilon_y * (1 + alpha_y**2) / beta_y)
    rho_x = - alpha_x / np.sqrt(1 + alpha_x**2)
    rho_y = - alpha_y / np.sqrt(1 + alpha_y**2)

    # make underlying Gaussian variables
    if symmetrize:
        N_actual = round(N/4)
    else:
        N_actual = N
    us_x = np.random.normal(size=N_actual)
    vs_x = np.random.normal(size=N_actual)
    us_y = np.random.normal(size=N_actual)
    vs_y = np.random.normal(size=N_actual)

    # do symmetrization
    if symmetrize:
        us_x  = np.concatenate((us_x, -us_x, us_x, -us_x))
        vs_x = np.concatenate((vs_x, -vs_x, vs_x, -vs_x))
        us_y  = np.concatenate((us_y, us_y, -us_y, -us_y))
        vs_y = np.concatenate((vs_y, vs_y, -vs_y, -vs_y))
        
    # angular momentum correlations
    ratio = L/np.sqrt(epsilon_x*epsilon_y)
    rho_L = np.sqrt(1 + ratio**2)
    
    # particle positions
    xs = sigx*(us_x + ratio*vs_y)/np.sqrt(rho_L)
    ys = sigy*(us_y - ratio*vs_x)/np.sqrt(rho_L)
    
    # particle angles
    xps = (sigxp*us_x*rho_x + sigxp*vs_x*np.sqrt(1 - rho_x**2))*np.sqrt(rho_L)
    yps = (sigyp*us_y*rho_y + sigyp*vs_y*np.sqrt(1 - rho_y**2))*np.sqrt(rho_L)
    
    return xs, xps, ys, yps


# =============================================
def generate_symm_trace_space_xyz(epsilon_x, beta_x, alpha_x, epsilon_y, beta_y, alpha_y, N, bunch_length, energy_spread, L=0):
    """
    Generate a fully symmetrized 6D trace-space particle distribution.

    This function constructs a Gaussian beam distribution symmetrised around 
    zero in all six phase-space dimensions (x, x', y, y', z, E). The 
    distribution is defined by the geometric emittances, Twiss parameters,
    bunch length, and relative energy spread. Optional canonical angular
    momentum ``L`` introduces correlated x–y coupling.


    Parameters
    ----------
    epsilon_x : [m rad] float
        Geometric emittance in the horizontal plane.

    beta_x : [m] float
        Horizontal beta function at the sampling location.

    alpha_x : float
        Horizontal Twiss alpha parameter.

    epsilon_y : [m rad] float
        Geometric emittance in the vertical plane.

    beta_y : [m] float
        Vertical beta function at the sampling location.

    alpha_y : float
        Vertical Twiss alpha parameter.

    N : int
        Number of macroparticles to generate. If ``symmetrize=True`` this
        represents the final number of returned samples.

    bunch_length : [m] float
        RMS bunch length.

    energy_spread : [eV] float
        Energy spread (std).

    L : [m rad] float, optional
        Canonical angular momentum (x–y coupling). ``L = 0`` produces
        uncoupled transverse motion. Defaults to 0.


    Returns
    -------
    xs : [m] 1D float ndarray
        Transverse positions in the horizontal plane.

    xps : [rad] 1D float ndarray
        Transverse angles (slopes) in the horizontal plane.

    ys : [m] 1D float ndarray
        Transverse positions in the vertical plane.

    yps : [rad] 1D float ndarray
        Transverse angles (slopes) in the vertical plane.

    zs : [m] 1D float ndarray
        Longitudinal positions.

    Es : [eV] 1D float ndarray
        Particle energies.
    """

    # calculate beam size, divergence and correlation
    sigx = np.sqrt(epsilon_x * beta_x)
    sigy = np.sqrt(epsilon_y * beta_y)
    sigxp = np.sqrt(epsilon_x * (1 + alpha_x**2) / beta_x)
    sigyp = np.sqrt(epsilon_y * (1 + alpha_y**2) / beta_y)
    rho_x = - alpha_x / np.sqrt(1 + alpha_x**2)
    rho_y = - alpha_y / np.sqrt(1 + alpha_y**2)

    # make underlying Gaussian variables
    N_actual = round(N/8)
    us_x = np.random.normal(size=N_actual*2)
    vs_x = np.random.normal(size=N_actual*2)
    us_y = np.random.normal(size=N_actual*2)
    vs_y = np.random.normal(size=N_actual*2)
    zs = np.random.normal(scale=bunch_length, size=N_actual)
    Es = np.random.normal(scale=energy_spread, size=N_actual)

    # do symmetrization
    us_x  = np.concatenate((us_x, -us_x, us_x, -us_x))
    vs_x = np.concatenate((vs_x, -vs_x, vs_x, -vs_x))
    us_y  = np.concatenate((us_y, us_y, -us_y, -us_y))
    vs_y = np.concatenate((vs_y, vs_y, -vs_y, -vs_y))
    zs = np.concatenate((-zs, zs))
    zs = np.tile(zs, 4)
    Es = np.concatenate((-Es, Es))
    Es = np.tile(Es, 4)
    
    # angular momentum correlations
    ratio = L/np.sqrt(epsilon_x*epsilon_y)
    rho_L = np.sqrt(1 + ratio**2)
    
    # particle positions
    xs = sigx*(us_x + ratio*vs_y)/np.sqrt(rho_L)
    ys = sigy*(us_y - ratio*vs_x)/np.sqrt(rho_L)
    
    # particle angles
    xps = (sigxp*us_x*rho_x + sigxp*vs_x*np.sqrt(1 - rho_x**2))*np.sqrt(rho_L)
    yps = (sigyp*us_y*rho_y + sigyp*vs_y*np.sqrt(1 - rho_y**2))*np.sqrt(rho_L)
    
    return xs, xps, ys, yps, zs, Es


# =============================================
def Rmat(l, k=0, plasmalens=True):
    """
    General focusing transfer matrix (quadrupole and drift)
    """
    import warnings
    warnings.filterwarnings("ignore")
    
    if k == 0:
        return np.matrix([[1,l,0,0],
                          [0,1,0,0],
                          [0,0,1,l],
                          [0,0,0,1]])
    elif plasmalens:
        if k > 0:
            return np.matrix([[np.cos(np.sqrt(k)*l),np.sin(np.sqrt(k)*l)/np.sqrt(k),0,0],
                          [-np.sin(np.sqrt(k)*l)*np.sqrt(k),np.cos(np.sqrt(k)*l),0,0],
                          [0,0,np.cos(np.sqrt(k)*l),np.sin(np.sqrt(k)*l)/np.sqrt(k)],
                          [0,0,-np.sin(np.sqrt(k)*l)*np.sqrt(k),np.cos(np.sqrt(k)*l)]])
        elif k < 0:
            return np.matrix([[np.cosh(np.sqrt(-k)*l),np.sinh(np.sqrt(-k)*l)/np.sqrt(-k),0,0],
                          [np.sinh(np.sqrt(-k)*l)*np.sqrt(-k),np.cosh(np.sqrt(-k)*l),0,0],
                          [0,0,np.cosh(np.sqrt(-k)*l),np.sinh(np.sqrt(-k)*l)/np.sqrt(-k)],
                          [0,0,np.sinh(np.sqrt(-k)*l)*np.sqrt(-k),np.cosh(np.sqrt(-k)*l)]])
    elif not plasmalens:
        if k > 0:
            return np.matrix([[np.cos(np.sqrt(k)*l),np.sin(np.sqrt(k)*l)/np.sqrt(k),0,0],
                          [-np.sin(np.sqrt(k)*l)*np.sqrt(k),np.cos(np.sqrt(k)*l),0,0],
                          [0,0,np.cosh(np.sqrt(k)*l),np.sinh(np.sqrt(k)*l)/np.sqrt(k)],
                          [0,0,np.sinh(np.sqrt(k)*l)*np.sqrt(k),np.cosh(np.sqrt(k)*l)]])
        elif k < 0:
            return np.matrix([[np.cosh(np.sqrt(-k)*l),np.sinh(np.sqrt(-k)*l)/np.sqrt(-k),0,0],
                          [np.sinh(np.sqrt(-k)*l)*np.sqrt(-k),np.cosh(np.sqrt(-k)*l),0,0],
                          [0,0,np.cos(np.sqrt(-k)*l),np.sin(np.sqrt(-k)*l)/np.sqrt(-k)],
                          [0,0,-np.sin(np.sqrt(-k)*l)*np.sqrt(-k),np.cos(np.sqrt(-k)*l)]])


# =============================================
def Dmat(l, inv_rho=0, k=0):
    """
    General dispersion transfer matrix (dipole, quadrupole and drift)
    """
    R = Rmat(l,k)
    if inv_rho == 0:
        return np.matrix([[R[0,0], R[0,1], 0],
                          [R[1,0], R[1,1], 0],
                          [0, 0, 1]])
    else:
        return np.matrix([[R[0,0], R[0,1], -(1-np.cos(l*inv_rho))/inv_rho],
                          [R[1,0], R[1,1], -2*np.tan(l*inv_rho/2)],
                          [0, 0, 1]])
    
    
# =============================================
def evolve_beta_function(ls, ks, beta0, alpha0=0, inv_rhos=None, fast=False, plot=False):
    """
    Evolution of the beta function.
    """
    
    # overwrite fast-calculation toggle if plotting 
    if plot and fast:
        fast = False
        
    if not fast:
        Nres = 100
        evolution = np.empty([3, Nres*len(ls)])
    else:
        evolution = None

    # if given, take into account the weak focusing effect
    if inv_rhos is not None:
        ks = ks + inv_rhos**2
    
    # Twiss gamma function
    gamma0 = (1+alpha0**2)/beta0
      
    # calculate transfer matrix evolution
    Rtot = np.identity(4)
    for i in range(len(ls)):
        
        # full beta evolution
        if not fast:
            s0 = np.sum(ls[:i])
            ss_l = s0 + np.linspace(0, ls[i], Nres)
            for j in range(len(ss_l)):
                R_l = Rmat(ss_l[j]-s0, ks[i]) @ Rtot
                evolution[0,i*Nres+j] = ss_l[j]
                evolution[1,i*Nres+j] = beta0*R_l[0,0]**2 - 2*alpha0*R_l[0,0]*R_l[0,1] + gamma0*R_l[0,1]**2
                evolution[2,i*Nres+j] = -beta0*R_l[0,0]*R_l[1,0] + alpha0*(R_l[1,1]*R_l[0,0]+R_l[0,1]*R_l[1,0]) - gamma0*R_l[0,1]*R_l[1,1]
        
        # final matrix
        Rtot = Rmat(ls[i],ks[i]) @ Rtot
    
    # calculate final Twiss parameters
    beta = beta0*Rtot[0,0]**2 - 2*alpha0*Rtot[0,0]*Rtot[0,1] + gamma0*Rtot[0,1]**2
    alpha = -beta0*Rtot[0,0]*Rtot[1,0] + alpha0*(Rtot[1,1]*Rtot[0,0]+Rtot[0,1]*Rtot[1,0]) - gamma0*Rtot[0,1]*Rtot[1,1]

    if plot:
        from matplotlib import pyplot as plt
        # prepare plot
        fig, ax = plt.subplots(1,1)
        ax.plot(evolution[0,:], np.sqrt(evolution[1,:]))
        ax.set_xlabel('s (m)')
        ax.set_ylabel('Square root of beta function (m^0.5)]')
        
    return beta, alpha, evolution


# =============================================
def evolve_dispersion(ls, inv_rhos, ks, Dx0=0, Dpx0=0, fast=False, plot=False, high_res=False):
    """
    Numerically compute the evolution of the first-order transverse dispersion 
    along a beamline with dipole and quadrupole fields.


    Parameters
    ----------
    ls : [m] 1D float ndarray
        Lattice element lengths.

    inv_rhos : [m^-1] 1D float ndarray
            Inverse bending radii.

    ks : [m^-2] 1D float ndarray
        Plasma lens focusing strengths.

    Dx0 : float [m], optional
        Initial horizontal dispersion value at s=0.

    Dpx0 : float, optional
        Initial derivative of dispersion at s=0.

    fast : bool, optional
        If ``True``, skips detailed integration within each element for faster 
        computation (no intermediate data are returned). Defaults to ``False``.

    plot : bool, optional
        If ``True``, sets ``fast`` to ``False`` and plots the evolution of the 
        first-order dispersion along the beamline. Defaults to ``False``.

    high_res : bool, optional
        Increase the evolution sampling frequency. Defaults to ``False``.
    """
    
    # overwrite fast-calculation toggle if plotting 
    if plot:
        fast = False
        
    if not fast:
        Nres_high = 100
        Nres_low = 10
        if high_res:
            Nres_high = Nres_high*10
            Nres_low = Nres_low*10
        Ntot = Nres_high*np.sum(abs(inv_rhos)>0) + Nres_low*np.sum(abs(inv_rhos)==0)
        evolution = np.empty([3, Ntot])
    else:
        evolution = None
    
    # calculate transfer matrix evolution
    Dtot = np.identity(3)
    i_last = 0
    for i in range(len(ls)):
        
        # full dispersion evolution
        if not fast:
            
            if abs(inv_rhos[i]) > 0:
                Nres = Nres_high
            else:
                Nres = Nres_low
            
            s0 = np.sum(ls[:i])
            ss_l = s0 + np.linspace(0, ls[i], Nres)
            for j in range(len(ss_l)):
                D_l = Dmat(ss_l[j]-s0, inv_rhos[i], ks[i]) @ Dtot
                evolution[0,i_last+j] = ss_l[j]
                evolution[1,i_last+j] = Dx0*D_l[0,0] + Dpx0*D_l[1,2] + D_l[0,2]
                evolution[2,i_last+j] = Dx0*D_l[1,0] + Dpx0*D_l[1,1] + D_l[1,2]
            i_last = i_last + Nres
            
        # final matrix
        Dtot = Dmat(ls[i],inv_rhos[i],ks[i]) @ Dtot
    
    # calculate final dispersion
    Dx = Dx0*Dtot[0,0] + Dpx0*Dtot[0,1] + Dtot[0,2]
    Dpx = Dx0*Dtot[1,0] + Dpx0*Dtot[1,1] + Dtot[1,2]
    
    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(evolution[0,:], evolution[1,:])
        ax.set_xlabel('s (m)')
        ax.set_ylabel('Dispersion (m)')
    
    return Dx, Dpx, evolution


# =============================================
def evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=False, plot=False):
    """
    Numerically compute the evolution of second- and higher-order horizontal 
    dispersion functions along a beamline with dipole, quadrupole, and sextupole 
    fields.

    This routine tracks several particles with different relative momentum 
    deviations through a sequence of magnetic elements, integrates their 
    trajectories, and computes the derivatives of transverse displacement with 
    respect to relative momentum deviation (``delta``) up to fourth order using 
    a five-point finite difference stencil. 

    It can return the final second-order dispersion values (``DDx``, ``DDpx``) 
    and, optionally, the full evolution of ``DDx`` and ``DDpx`` along the 
    beamline.

    Parameters
    ----------
    ls : [m] 1D float ndarray
        Lattice element lengths.

    inv_rhos : [m^-1] 1D float ndarray
            Inverse bending radii.

    ks : [m^-2] 1D float ndarray
        Plasma lens focusing strengths.

    ms : [m^-3] 1D float ndarray
        Sextupole strengths.

    taus : [m^-1] 1D float ndarray
        Plasma lens transverse taper coefficients.

    fast : bool, optional
        If ``True``, skips detailed integration within each element for faster 
        computation (no intermediate data are returned). Defaults to ``False``.

    plot : bool, optional
        If ``True``, sets ``fast`` to ``False`` and plots the evolution of the 
        second-order dispersion along the beamline. Defaults to ``False``.

    Returns
    -------
    DDx : [m] float
        Final second-order dispersion at the end of the beamline.

    DDpx : float
        Final derivative of the second-order dispersion wrt. s.

    evolution : ndarray or None
        If ``fast=False``, an array of shape (5, N) containing:

            - ``evolution[0, :]``: longitudinal position s [m]
            - ``evolution[1, :]``: first-order dispersion [m]
            - ``evolution[2, :]``: second-order dispersion [m]
            - ``evolution[3, :]``: third-order dispersion [m]
            - ``evolution[4, :]``: fourth-order dispersion [m]

        If ``fast=True``, returns ``None``.

    Notes
    -----
    - The integration uses a simple symplectic (leapfrog) method for stability.
    - Derivatives with respect to ``delta`` are computed using a five-point stencil for high accuracy.
    - The dispersion expansion follows:
          x(delta) = x_0 + Dx * delta + DDx * delta^2 + DDDx * delta^3 + DDDDx * delta^4 + O(delta^5)
    """
    
    # overwrite fast-calculation toggle if plotting 
    if plot and fast:
        fast = False
        
    # element resolution
    Nress = np.zeros(len(ls))
    for i in range(len(ls)):
        Nress[i] = 100
    
    # use five energy offsets for good accuracy
    delta = 1e-4
    deltas = delta * np.arange(-2,3)
    
    # declare lists
    if not fast:
        ss = np.zeros(1+int(np.sum(Nress)))
        xs = np.zeros([1+int(np.sum(Nress)),len(deltas)])
        xps = np.zeros([1+int(np.sum(Nress)),len(deltas)])
        evolution = np.zeros([5,1+int(np.sum(Nress))])
    else:
        evolution = None

    # initialize at zero offset and angle
    x = np.zeros(len(deltas))
    xp = np.zeros(len(deltas))

    # go through elements one by one
    for i in range(len(ls)):
        
        # longitudinal positions within element
        ds = ls[i]/Nress[i]
        ss_l = np.sum(ls[:i]) + np.linspace(ds, ls[i], int(Nress[i]))
        
        # numerically integrate step by step
        if not fast:
            xs_l = np.zeros([len(ss_l),len(deltas)])
            xps_l = np.zeros([len(ss_l),len(deltas)])
        
        # dipole force (constant)
        d2x_ds2_dip = inv_rhos[i]*(1/(1+deltas)-1)

        for j in range(len(ss_l)):
            for k in range(len(deltas)):

                # sample location for x (and y = 0) at a half step forward
                x_samp = x[k] + xp[k]*ds/2

                # plasma lens force
                d2x_ds2_lens = -ks[i]/(1+deltas[k])*(x_samp + taus[i]*x_samp**2/2)

                # sextupole force
                d2x_ds2_sext = -ms[i]/(1+deltas[k])*x_samp**2/2

                # total force
                dxps_ds = d2x_ds2_dip[k] + d2x_ds2_lens + d2x_ds2_sext

                # save last step for next iteration
                xp_last = xp[k]
                xp[k] = xp_last + dxps_ds*ds
                x[k] = x[k] + (xp[k]+xp_last)/2*ds

            # step particles based on fields
            if not fast:
                xps_l[j,:] = xp
                xs_l[j,:] = x
        
        # add longitudinal positions, offsets and angles
        if not fast:
            inds = int(np.sum(Nress[:i])) + np.arange(int(Nress[i])) + 1
            ss[inds] = ss_l
            xs[inds,:] = xs_l
            xps[inds,:] = xps_l
           

    # dispersion evolution up to fourth order (see https://en.wikipedia.org/wiki/Five-point_stencil)
    # here the dispersion is defined as x(delta) = x_0 + Dx*delta + DDx*delta^2 + DDDx*delta^3 + DDDDx*delta^4 + O(delta^5)
    # this definition requires multiplying by the factorial of the order
    if not fast:
        evolution[0,:] = ss
        evolution[1,:] = (-xs[:,4] + 8*xs[:,3] - 8*xs[:,1] + xs[:,0])/(12*delta)  # First order
        evolution[2,:] = 2*(-xs[:,4] + 16*xs[:,3] - 30*xs[:,2] + 16*xs[:,1] - xs[:,0])/(12*delta**2)  # Second order
        evolution[3,:] = 3*2*(xs[:,4] - 2*xs[:,3] + 2*xs[:,1] - xs[:,0])/(2*delta**3)  # Third order
        evolution[4,:] = 4*3*2*(xs[:,4] - 4*xs[:,3] + 6*xs[:,2] - 4*xs[:,1] + xs[:,0])/delta**4  # Fourth order

    # return dispersions
    DDx = 2*(-x[4] + 16*x[3] - 30*x[2] + 16*x[1] - x[0])/(12*delta**2)
    DDpx = 2*(-xp[4] + 16*xp[3] - 30*xp[2] + 16*xp[1] - xp[0])/(12*delta**2) # Second order
    
    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(evolution[0,:], evolution[2,:])
        ax.set_xlabel('s (m)')
        ax.set_ylabel('Second-order dispersion (m)')
        
    return DDx, DDpx, evolution


# =============================================
def evolve_R56(ls, inv_rhos, ks, Dx0=0, Dpx0=0, fast=False, plot=False, high_res=False, evolution_disp=None):
    """
    Evolution of longitudinal dispersion, R56.
    """
    
    # overwrite fast-calculation toggle if plotting 
    if plot and fast:
        fast = False
       
    # get the dispersion evolution
    if evolution_disp is None:
        _, _, evolution_disp = evolve_dispersion(ls, inv_rhos, ks, Dx0=Dx0, Dpx0=Dpx0, fast=False, plot=False, high_res=high_res)
    ss = evolution_disp[0]
    Dxs = evolution_disp[1]
    R56s = np.empty_like(ss)

    # intialize at zero R56
    R56s[0] = 0

    # make cumulative lengths
    ssl = np.append([0.0], np.cumsum(ls))[:-1]
    
    # calculate the evolution
    for i in range(len(ss)-1):

        s_prev = ss[i]
        index_element_prev = np.argmin(abs(ssl - s_prev))
        index_element_ceil_prev = index_element_prev + int(ssl[index_element_prev] <= s_prev) - 1
        inv_rho_prev = inv_rhos[index_element_ceil_prev]

        s = ss[i+1]
        index_element = np.argmin(abs(ssl - s))
        index_element_ceil = index_element + int(ssl[index_element] <= s) - 1
        inv_rho = inv_rhos[index_element_ceil]

        ds = ss[i+1]-ss[i]
        inv_rho_halfstep = (inv_rho_prev+inv_rho)/2
        Dx = Dxs[i+1]
        deltaR56 = - Dx * inv_rho_halfstep * ds
        R56s[i+1] = R56s[i] + deltaR56

    # save evolution
    if not fast:
        evolution = np.empty([2, len(ss)])
        evolution[0,:] = ss
        evolution[1,:] = R56s
    else:
        evolution = None

    # extract final R56
    R56 = R56s[-1]

    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(evolution[0,:], evolution[1,:])
        ax.set_xlabel('s (m)')
        ax.set_ylabel('Longitudinal dispersion, R56 (m)')

    return R56, evolution


# =============================================
def evolve_orbit(ls, inv_rhos, x0=0, y0=0, s0=0, theta0=0, plot=False):
    """
    Evolution of the beam orbit (i.e., the top view).
    """

    # points per dipole
    num_steps = 25
    
    # calculate the orbit
    xs = np.array([x0])
    ys = np.array([y0])
    ss = np.array([s0])
    thetas = np.array([theta0])
    
    for i in range(len(ls)):

        dtheta = -ls[i]*inv_rhos[i]
        thetas_next = np.linspace(0, dtheta, num_steps)
        ss_next = np.linspace(0, ls[i], num_steps)
        if abs(dtheta) > 0:
            xs_next = -np.sin(thetas_next)/inv_rhos[i]
            ys_next = (1-np.cos(thetas_next))/inv_rhos[i]
        else:
            xs_next = ss_next
            ys_next = np.zeros_like(ss_next)

        ss = np.append(ss, ss[-1] + ss_next)
        xs = np.append(xs, xs[-1] + xs_next*np.cos(thetas[-1]) + ys_next*np.sin(thetas[-1]))
        ys = np.append(ys, ys[-1] - xs_next*np.sin(thetas[-1]) + ys_next*np.cos(thetas[-1]))
        thetas = np.append(thetas, thetas[-1] + thetas_next)

    # save orbit
    evolution = np.zeros([4, len(ss)])
    evolution[0,:] = xs
    evolution[1,:] = ys
    evolution[2,:] = ss
    evolution[3,:] = thetas
    
    # final angle
    theta = thetas[-1]

    # plot if required
    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(evolution[0,:], evolution[1,:])
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    
    return theta, evolution


# =============================================
def evolve_curlyH(ls, inv_rhos, ks, beta0, alpha0=0, Dx0=0, Dpx0=0, plot=False):
    """
    Evolution of the curly H function (i.e., single-particle emittance of the dispersion).
    """
    
    _, _, evol_disp = evolve_dispersion(ls, inv_rhos, ks, Dx0=0, Dpx0=0, fast=False, plot=False, high_res=True)
    ss = evol_disp[0]
    Dxs = evol_disp[1]
    Dpxs = evol_disp[2]
    
    _, _, evol_beta = evolve_beta_function(ls, ks, beta0, alpha0=alpha0, fast=False, plot=False)
    betas = np.interp(ss, evol_beta[0], np.sqrt(evol_beta[1]))**2
    alphas = np.interp(ss, evol_beta[0], evol_beta[2])
    gammas = (1+alphas**2)/betas

    # combine into curly H function (i.e., the "dispersion emittance")
    curlyHs = Dxs**2*gammas + 2*alphas*Dxs*Dpxs + betas*Dpxs**2

    # save evolution
    evolution = np.zeros([2, len(ss)])
    evolution[0,:] = ss
    evolution[1,:] = curlyHs
    
    curlyH = curlyHs[-1]

    # plot if required
    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(ss, curlyHs*1e6)
        ax.set_xlabel('s (m)')
        ax.set_ylabel(r'$\mathscr{H}$ (mm mrad)')
    
    return curlyH, evolution


# =============================================
def evolve_I2(ls, inv_rhos, fast=False, plot=False):
    """
    Evolution of the second synchrotron radiation integral I_2.
    """
    
    # make cumulative lengths
    ss = np.append([0.0], np.cumsum(ls))
    I2s = np.empty_like(ss)
    
    # intialize at zero  I2
    I2s[0] = 0
    
    # calculate the evolution
    for i in range(len(ls)):
        deltaI2 = ls[i]*inv_rhos[i]**2
        I2s[i+1] = I2s[i] + deltaI2

    # save evolution
    if not fast:
        evolution = np.empty([2, len(ss)])
        evolution[0,:] = ss
        evolution[1,:] = I2s
    else:
        evolution = None

    # extract final I2
    I2 = I2s[-1]

    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(ss, I2s)
        ax.set_xlabel('s (m)')
        ax.set_ylabel(r'Second synchrotron radiation integral, $I_2$ (m$^{-1}$)')

    return I2, evolution


# =============================================
def evolve_I3(ls, inv_rhos, fast=False, plot=False):
    """
    Evolution of the third synchrotron radiation integral I_3.
    """

    # make cumulative lengths
    ss = np.append([0.0], np.cumsum(ls))
    I3s = np.empty_like(ss)
    
    # intialize at zero I3
    I3s[0] = 0
    
    # calculate the evolution
    for i in range(len(ls)):
        deltaI3 = ls[i]*abs(inv_rhos[i])**3
        I3s[i+1] = I3s[i] + deltaI3

    # save evolution
    if not fast:
        evolution = np.empty([2, len(ss)])
        evolution[0,:] = ss
        evolution[1,:] = I3s
    else:
        evolution = None

    # extract final I3
    I3 = I3s[-1]

    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(ss, I3s)
        ax.set_xlabel('s (m)')
        ax.set_ylabel(r'Third synchrotron radiation integral, $I_3$ (m$^{-1}$)')

    return I3, evolution


# =============================================
def evolve_I4(ls, inv_rhos, ks, Dx0=0, Dpx0=0, fast=False, plot=False):
    """
    Evolution of the fourth synchrotron radiation integral I_4.
    """
    
    _, _, evol_disp = evolve_dispersion(ls, inv_rhos, ks, Dx0=0, Dpx0=0, fast=False, plot=False, high_res=True)
    ss = evol_disp[0]
    Dxs = -evol_disp[1] # TODO: check this sign
    I4s = np.empty_like(ss)
    
    # make cumulative lengths
    ssl = np.append([0.0], np.cumsum(ls))[:-1]
    
    # intialize at zero I5
    I4s[0] = 0
    
    # calculate the evolution
    for i in range(len(ss)-1):

        s_prev = ss[i]
        index_element_prev = np.argmin(abs(ssl - s_prev))
        index_element_ceil_prev = index_element_prev + int(ssl[index_element_prev] <= s_prev) - 1
        inv_rho_prev = inv_rhos[index_element_ceil_prev]
        k_prev = ks[index_element_ceil_prev]

        s = ss[i+1]
        index_element = np.argmin(abs(ssl - s))
        index_element_ceil = index_element + int(ssl[index_element] <= s) - 1
        inv_rho = inv_rhos[index_element_ceil]
        k = ks[index_element_ceil]

        ds = ss[i+1]-ss[i]
        inv_rho_halfstep = (inv_rho_prev+inv_rho)/2
        k_halfstep = (k_prev+k)/2
        Dx = Dxs[i+1]
        deltaI4 = Dx * inv_rho_halfstep * (inv_rho_halfstep**2 + 2*k_halfstep) * ds
        I4s[i+1] = I4s[i] + deltaI4

    # save evolution
    if not fast:
        evolution = np.empty([2, len(ss)])
        evolution[0,:] = ss
        evolution[1,:] = I4s
    else:
        evolution = None

    # extract final I5
    I4 = I4s[-1]

    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(ss, I4s)
        ax.set_xlabel('s (m)')
        ax.set_ylabel(r'Fourth synchrotron radiation integral, $I_4$ (m$^{-1}$)')
        #ax.set_yscale('log')

    return I4, evolution


# =============================================
def evolve_I5(ls, inv_rhos, ks, beta0, alpha0=0, Dx0=0, Dpx0=0, fast=False, plot=False):
    """
    Evolution of the fifth synchrotron radiation integral I_5.
    """
    
    _, evol = evolve_curlyH(ls, inv_rhos, ks, beta0, alpha0=alpha0, Dx0=Dx0, Dpx0=Dpx0, plot=False)
    ss = evol[0,:]
    curlyHs = evol[1,:]
    I5s = np.empty_like(ss)
    
    # make cumulative lengths
    ssl = np.append([0.0], np.cumsum(ls))[:-1]
    
    # intialize at zero I5
    I5s[0] = 0
    
    # calculate the evolution
    for i in range(len(ss)-1):

        s_prev = ss[i]
        index_element_prev = np.argmin(abs(ssl - s_prev))
        index_element_ceil_prev = index_element_prev + int(ssl[index_element_prev] <= s_prev) - 1
        inv_rho_prev = inv_rhos[index_element_ceil_prev]

        s = ss[i+1]
        index_element = np.argmin(abs(ssl - s))
        index_element_ceil = index_element + int(ssl[index_element] <= s) - 1
        inv_rho = inv_rhos[index_element_ceil]

        ds = ss[i+1]-ss[i]
        inv_rho_halfstep = (inv_rho_prev+inv_rho)/2
        curlyH = curlyHs[i+1]
        deltaI5 = curlyH * abs(inv_rho_halfstep)**3 * ds
        I5s[i+1] = I5s[i] + deltaI5

    # save evolution
    if not fast:
        evolution = np.empty([2, len(ss)])
        evolution[0,:] = ss
        evolution[1,:] = I5s
    else:
        evolution = None

    # extract final I5
    I5 = I5s[-1]

    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(ss, I5s)
        ax.set_xlabel('s (m)')
        ax.set_ylabel(r'Fifth synchrotron radiation integral, $I_5$ (m$^{-1}$)')
        ax.set_yscale('log')

    return I5, evolution


# =============================================
def evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, beta0, alpha0=0, Dx0=0, Dpx0=0, fast=False, plot=False, bending_plane=True):
    """
    Evolution of the first-order chromatic amplitude W.
    """
    
    # overwrite fast-calculation toggle if plotting 
    if plot and fast:
        fast = False
      
    # use five energy offsets for good accuracy
    delta0 = 1e-4
    deltas = delta0 * np.arange(-2,3)

    # get the dispersion for calculation of effect of chromaticity correction)
    _, _, evol_disp = evolve_dispersion(ls, inv_rhos, ks, Dx0=0, Dpx0=0, fast=False, plot=False, high_res=False)
    ss_disp = evol_disp[0]
    Dxs = evol_disp[1]
    
    # calculate the average dispersion inside each element to find the effect of nonlinear elements
    ssl = np.append([0.], np.cumsum(ls))

    # prepare arrays of effect of nonlinear plasma lens (tau) and sextupole (m)
    dks_ddelta_m = np.empty(0)
    dks_ddelta_tau = np.empty(0)
    ls_refined = np.empty(0)
    ks_refined = np.empty(0)
    inv_rhos_refined = np.empty(0)
    for i in range(len(ls)):
        
        if abs(taus[i]) > 0 or abs(ms[i]) > 0:
            inds = np.logical_and(ss_disp >= ssl[i], ss_disp <= ssl[i+1])
            num_slice = 30
            ss_slice = np.linspace(ssl[i], ssl[i+1], num_slice)
            Dxs_slices = np.interp(ss_slice, ss_disp, Dxs)
            dk_ddelta_m = ms[i]*Dxs_slices
            dk_ddelta_tau = ks[i]*taus[i]*Dxs_slices
            ls_element = ls[i]/num_slice*np.ones_like(ss_slice)
            ks_element = ks[i]*np.ones_like(ss_slice)
            inv_rhos_element = inv_rhos[i]*np.ones_like(ss_slice)
        else:
            dk_ddelta_m = np.array([0.0])
            dk_ddelta_tau = np.array([0.0])
            ls_element = np.array([ls[i]])
            ks_element = np.array([ks[i]])
            inv_rhos_element = np.array([inv_rhos[i]])
            
        dks_ddelta_m = np.append(dks_ddelta_m, dk_ddelta_m)
        dks_ddelta_tau = np.append(dks_ddelta_tau, dk_ddelta_tau)
        ls_refined = np.append(ls_refined, ls_element)
        ks_refined = np.append(ks_refined, ks_element)
        inv_rhos_refined = np.append(inv_rhos_refined, inv_rhos_element)

    # prepare arrays
    betas = np.empty_like(deltas)
    alphas = np.empty_like(deltas)
    evols = [None]*len(deltas)

    if not bending_plane:
        ks_refined = -ks_refined
        dks_ddelta_m = -dks_ddelta_m
        inv_rhos_refined = None
    
    # evolve the beta and alpha for different energies
    for i, delta in enumerate(deltas):
        ks_corrected = (ks_refined + (dks_ddelta_tau + dks_ddelta_m)*delta)/(1+delta)
        if inv_rhos_refined is not None:
            inv_rhos_corrected = inv_rhos_refined/(1+delta)
        else:
            inv_rhos_corrected = inv_rhos_refined
        betas[i], alphas[i], evols[i] = evolve_beta_function(ls_refined, ks_corrected, beta0, alpha0=alpha0, inv_rhos=inv_rhos_corrected, fast=fast, plot=False)
    
    # calculate the chromatic amplitude W
    beta = betas[2]
    alpha = alphas[2]
    dbeta_ddelta = (-betas[4] + 8*betas[3] - 8*betas[1] + betas[0])/(12*delta0)
    dalpha_ddelta = (-alphas[4] + 8*alphas[3] - 8*alphas[1] + alphas[0])/(12*delta0)
    W = np.sqrt((dalpha_ddelta - (alpha/beta)*dbeta_ddelta)**2 + (dbeta_ddelta/beta)**2)
    
    # save evolution
    if not fast: 
        evolution = np.empty((2,len(evols[2][0,:])))
        evolution[0,:] = evols[2][0,:]
        betas = evols[2][1,:]
        alphas = evols[2][2,:]
        dbeta_ddeltas = (-evols[4][1,:] + 8*evols[3][1,:] - 8*evols[1][1,:] + evols[0][1,:])/(12*delta0)
        dalpha_ddeltas = (-evols[4][2,:] + 8*evols[3][2,:] - 8*evols[1][2,:] + evols[0][2,:])/(12*delta0)        
        evolution[1,:] = np.sqrt((dalpha_ddeltas - (alphas/betas)*dbeta_ddeltas)**2 + (dbeta_ddeltas/betas)**2)
    else:
        evolution = None
    
    # make plots
    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(evolution[0,:], evolution[1,:])
        ax.set_xlabel('s (m)')
        ax.set_ylabel('W_x')

    return W, evolution


# =============================================
def phase_advance(ss, betas):
    """
    Calculate the phase advance in one dimesion by using the composite Simpson’s 
    rule (:func:`scipy.integrate.simpson() <scipy.integrate.simpson>`) to 
    integrate two arrays containing the location and the beta function.
    """
    
    from scipy import integrate
    inv_betas = 1/betas
    return integrate.simpson(y=inv_betas, x=ss)


# =============================================
def arc_lengths(s_trajectory, x_trajectory):
    """
    Docstring for arc_length
    
    :param s_trajectory: Description
    :param x_trajectory: Description
    """

    ds = np.diff(s_trajectory)
    dx = np.diff(x_trajectory)

    length = np.cumsum(np.sqrt(ds**2 + dx**2))

    length = np.insert(length, 0, 0.0)

    return length
