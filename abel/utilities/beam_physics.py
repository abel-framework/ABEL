import numpy as np

# generate trace space from geometric emittance and twiss parameters
def generate_trace_space(epsilon, beta, alpha, N, symmetrize=False):

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


# generate trace space from geometric emittance and twiss parameters (2 planes)
def generate_trace_space_xy(epsilon_x, beta_x, alpha_x, epsilon_y, beta_y, alpha_y, N, L=0, symmetrize=False):

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


# generate trace space from geometric emittance and twiss parameters for a beam symmetrised in 6D
def generate_symm_trace_space_xyz(epsilon_x, beta_x, alpha_x, epsilon_y, beta_y, alpha_y, N, bunch_length, energy_spread, L=0):

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

    
# general focusing transfer matrix (quadrupole and drift)
def Rmat(l, k=0, plasmalens=True):
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


# general dispersion transfer matrix (dipole, quadrupole and drift)
def Dmat(l, inv_rho=0, k=0):
    R = Rmat(l,k)
    if inv_rho == 0:
        return np.matrix([[R[0,0], R[0,1], 0],
                          [R[1,0], R[1,1], 0],
                          [0, 0, 1]])
    else:
        return np.matrix([[R[0,0], R[0,1], (1-np.cos(l*inv_rho))/inv_rho],
                          [R[1,0], R[1,1], 2*np.tan(l*inv_rho/2)],
                          [0, 0, 1]])
    
    

def evolve_beta_function(ls, ks, beta0, alpha0=0, fast=False, plot=False):

    # overwrite fast-calculation toggle if plotting 
    if plot and fast:
        fast = False
        
    if not fast:
        Nres = 100
        evolution = np.empty([3, Nres*len(ls)])
    else:
        evolution = None
    
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


def evolve_dispersion(ls, inv_rhos, ks, Dx0=0, Dpx0=0, fast=False, plot=False):
    
    # overwrite fast-calculation toggle if plotting 
    if plot and fast:
        fast = False
        
    if not fast:
        Nres_high = 200
        Nres_low = 20
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


def evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=False, plot=False):
    
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
        d2x_ds2_dip = inv_rhos[i]*(1-1/(1+deltas))

        for j in range(len(ss_l)):
            for k in range(len(deltas)):

                # sample location for x (and y = 0) at a half step forward
                x_samp = x[k] + xp[k]*ds/2

                # plasma lens force
                d2x_ds2_lens = -ks[i]/(1+deltas[k])*(x_samp + taus[i]*x_samp**2/2)

                # sextupole force
                d2x_ds2_sext = -ms[i]/(1+deltas[k])*x_samp**2/2

                # total force
                dxps = d2x_ds2_dip[k] + d2x_ds2_lens + d2x_ds2_sext

                # save last step for next iteration
                xp_last = xp[k]
                xp[k] = xp[k] + dxps*ds
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
           

    # dispersion evolution up to fourth order
    if not fast:
        evolution[0,:] = ss
        evolution[1,:] = (-xs[:,4] + 8*xs[:,3] - 8*xs[:,1] + xs[:,0])/(12*delta)  # First order
        evolution[2,:] = (-xs[:,4] + 16*xs[:,3] - 30*xs[:,2] + 16*xs[:,1] - xs[:,0])/(12*delta**2)  # Second order
        evolution[3,:] = (xs[:,4] - 2*xs[:,3] + 2*xs[:,1] - xs[:,0])/(2*delta**3)  # Third order
        evolution[4,:] = (xs[:,4] - 4*xs[:,3] + 6*xs[:,2] - 4*xs[:,1] + xs[:,0])/delta**4  # Fourth orderz

    # return dispersions
    DDx = (-x[4] + 16*x[3] - 30*x[2] + 16*x[1] - x[0])/(12*delta**2)
    DDpx = -(xp[2] - 2*xp[1] + xp[0])/(2*delta**2)

    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(evolution[0,:], evolution[2,:])
        ax.set_xlabel('s (m)')
        ax.set_ylabel('Second-order dispersion (m)')
        
    return DDx, DDpx, evolution


def evolve_R56(ls, inv_rhos, ks, Dx0=0, Dpx0=0, plot=False):

    # get the dispersion evolution
    _, _, evolution_disp = evolve_dispersion(ls, inv_rhos, ks, Dx0=Dx0, Dpx0=Dpx0, fast=False, plot=False)
    ss = evolution_disp[0]
    Dxs = evolution_disp[1]
    R56s = np.empty_like(ss)

    # intialize at zero R56
    R56s[0] = 0

    # make cumulative lengths
    ssl = np.append([0.0], np.cumsum(ls))
    ssl = ssl[:-1]
    
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
        Dx_halfstep = (Dxs[i]+Dxs[i+1])/2
        deltaR56 = Dx_halfstep * inv_rho * ds
        R56s[i+1] = R56s[i] + deltaR56

    # save evolution
    evolution = np.empty([2, len(ss)])
    evolution[0,:] = ss
    evolution[1,:] = R56s

    # extract final R56
    R56 = R56s[-1]

    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(evolution[0,:], evolution[1,:])
        ax.set_xlabel('s (m)')
        ax.set_ylabel('Longitudinal dispersion, R56 (m)')

    return R56, evolution

    

