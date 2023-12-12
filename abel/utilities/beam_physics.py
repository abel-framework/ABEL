import numpy as np
import time

# generate trace space from geometric emittance and twiss parameters
def generate_trace_space(epsilon, beta, alpha, N, symmetrize=False):

    gamma = (1 + alpha**2) / beta; # Twiss gamma
    
    sigx = np.sqrt(beta * epsilon); # rms beam size
    sigxp = np.sqrt(gamma * epsilon); # rms divergence
    rho = - alpha / np.sqrt(1 + alpha**2); # correlation

    if not symmetrize:
        us = np.random.normal(size = N) # Gaussian random variable 1
        vs = np.random.normal(size = N) # Gaussian random variable 2
    else:
        us = np.random.normal(size = round(N/4)) # Gaussian random variable 1
        vs = np.random.normal(size = round(N/4)) # Gaussian random variable 2
    
    xs = sigx*us; # particle positions
    xps = sigxp*us*rho + sigxp*vs*np.sqrt(1 - rho**2); # particle angles

    if symmetrize:
        xs = np.concatenate((xs, -xs, xs, -xs))
        xps = np.concatenate((xps, xps, -xps, -xps))
    
    return xs, xps


# generate trace space from geometric emittance and twiss parameters
def generate_trace_space_xy(epsilon_x, beta_x, alpha_x, epsilon_y, beta_y, alpha_y, N, L=0, symmetrize=False):

    # Twiss gamma
    gamma_x = (1 + alpha_x**2) / beta_x
    gamma_y = (1 + alpha_y**2) / beta_y
    
    sigx = np.sqrt(beta_x * epsilon_x) # rms beam size
    sigy = np.sqrt(beta_y * epsilon_y) # rms beam size
    sigxp = np.sqrt(gamma_x * epsilon_x) # rms divergence
    sigyp = np.sqrt(gamma_y * epsilon_y) # rms divergence
    rho_x = - alpha_x / np.sqrt(1 + alpha_x**2) # correlation
    rho_y = - alpha_y / np.sqrt(1 + alpha_y**2) # correlation

    if not symmetrize:
        us_x = np.random.normal(size = N) # Gaussian random variable 1
        vs_x = np.random.normal(size = N) # Gaussian random variable 2
        us_y = np.random.normal(size = N) # Gaussian random variable 1
        vs_y = np.random.normal(size = N) # Gaussian random variable 2
    else:
        us_x = np.random.normal(size = round(N/4)) # Gaussian random variable 1
        vs_x = np.random.normal(size = round(N/4)) # Gaussian random variable 2
        us_y = np.random.normal(size = round(N/4)) # Gaussian random variable 1
        vs_y = np.random.normal(size = round(N/4)) # Gaussian random variable 2

    # angular momentum correlations
    ratio = L/np.sqrt(epsilon_x*epsilon_y)
    rho_L = np.sqrt(1 + ratio**2)
    
    # particle positions
    xs = sigx*(us_x + abs(ratio)*vs_y)/np.sqrt(rho_L)
    ys = sigy*(us_y - ratio*vs_x)/np.sqrt(rho_L)
    
    # particle angles
    xps = (sigxp*us_x*rho_x + sigxp*vs_x*np.sqrt(1 - rho_x**2))*np.sqrt(rho_L)
    yps = (sigyp*us_y*rho_y + sigyp*vs_y*np.sqrt(1 - rho_y**2))*np.sqrt(rho_L)
    
    # complete the symmetrization
    if symmetrize:
        xs = np.concatenate((xs, -xs, xs, -xs))
        xps = np.concatenate((xps, xps, -xps, -xps))
        ys = np.concatenate((ys, -ys, ys, -ys))
        yps = np.concatenate((yps, yps, -yps, -yps))
    
    return xs, xps, ys, yps

    
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
    
    

def evolve_beta_function(ls, ks, beta0, alpha0=0, fast=False):
    if not fast:
        Nres = 50
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
    
    return beta, alpha, evolution


def evolve_dispersion(ls, inv_rhos, ks, Dx0=0, Dpx0=0, fast=False):
    if not fast:
        Nres = 100
        evolution = np.empty([3, Nres*len(ls)])
    else:
        evolution = None
    
    # calculate transfer matrix evolution
    Dtot = np.identity(3)
    for i in range(len(ls)):
        
        # full beta evolution
        if not fast:
            s0 = np.sum(ls[:i])
            ss_l = s0 + np.linspace(0, ls[i], Nres)
            for j in range(len(ss_l)):
                D_l = Dmat(ss_l[j]-s0, inv_rhos[i], ks[i]) @ Dtot
                #ss[i*Nres+j] = ss_l[j]
                #Dxs[i*Nres+j] = Dx0*D_l[0,0] + Dpx0*D_l[1,2] + D_l[0,2]
                evolution[0,i*Nres+j] = ss_l[j]
                evolution[1,i*Nres+j] = Dx0*D_l[0,0] + Dpx0*D_l[1,2] + D_l[0,2]
                evolution[2,i*Nres+j] = Dx0*D_l[1,0] + Dpx0*D_l[1,1] + D_l[1,2]
        
        # final matrix
        Dtot = Dmat(ls[i],inv_rhos[i],ks[i]) @ Dtot
    
    # calculate final dispersion
    Dx = Dx0*Dtot[0,0] + Dpx0*Dtot[0,1] + Dtot[0,2]
    Dpx = Dx0*Dtot[1,0] + Dpx0*Dtot[1,1] + Dtot[1,2]
    
    return Dx, Dpx, evolution


def evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=False):
    
    # element resolution
    Nress = np.zeros(len(ls))
    for i in range(len(ls)):
        Nress[i] = 100
    
    # use five energy offsets for good accuracy
    delta = 1e-5
    deltas = delta * np.arange(-2,3)
    
    # declare lists
    if not fast:
        #ss = np.zeros(Nres*len(ls))
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
        evolution[1,:] = (-xs[:,4] + 8*xs[:,3] - 8*xs[:,1] + xs[:,0])/(12*delta)
        evolution[2,:] = (-xs[:,4] + 16*xs[:,3] - 30*xs[:,2] + 16*xs[:,1] - xs[:,0])/(12*delta**2)
        evolution[3,:] = (xs[:,4] - 2*xs[:,3] + 2*xs[:,1] - xs[:,0])/(2*delta**3)
        evolution[4,:] = (xs[:,4] - 4*xs[:,3] + 6*xs[:,2] - 4*xs[:,1] + xs[:,0])/delta**4

    # return dispersions
    DDx = (-x[4] + 16*x[3] - 30*x[2] + 16*x[1] - x[0])/(12*delta**2)
    DDpx = (xp[2] - 2*xp[1] + xp[0])/(delta**2)

    return DDx, DDpx, evolution

