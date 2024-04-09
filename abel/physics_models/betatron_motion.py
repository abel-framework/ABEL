import numpy as np
from scipy.integrate import solve_ivp, odeint
import scipy.constants as SI
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numba import prange, jit, njit
import numba as nb
from types import SimpleNamespace
import multiprocessing
import os
import time
from abel.utilities.statistics import weighted_cov
num_cores = multiprocessing.cpu_count()
os.environ['NUMEXPR_MAX_THREADS'] = f'{num_cores}'

re = SI.physical_constants['classical electron radius'][0]

@njit#(parallel = True)
def acc_func(ys, A, B, C, D):
    # ys =[[x0, x1, x2, ...           ]  0
    #     [y0, y1, y2, ...            ]  1
    #     [vx0, vx1, vx2, ...         ]  2
    #     [vy0, vy1, vy2, ...         ]  3
    #     [gamma0, gamma1, gamma2, ...]] 4
    ax = -(A/ys[4] + B)*ys[2] - C/ys[4]*ys[0]
    ay = -(A/ys[4] + B)*ys[3] - C/ys[4]*ys[1]
    dgamma = A - D*ys[4]**2*(ys[0]**2 + ys[1]**2)

    dy_dz = np.empty_like(ys, dtype = np.float64)
    dy_dz[0] = ys[2]
    dy_dz[1] = ys[3]
    dy_dz[2] = ax
    dy_dz[3] = ay
    dy_dz[4] = dgamma
    
    return dy_dz

@njit
# Get the second derivatives of position, velocitity and gamma (based on equations above)
def gs(y, a, A, B, C, D):
    # y = [x, y, vx, vy, gamma]
    # a = [vx, vy, ax, ay, d_gamma]
    g = np.empty_like(y, dtype = np.float64)
    
    g_gamma = -2*D*y[4]*a[4]*(y[0]**2+y[1]**2) - 2*D*y[4]**2*(y[0]*y[1]+y[2]*y[3])
    g_xy = a[2:4]
    g_v = A/y[4]**2*a[4]*a[:2] - (A/y[4] + B)*a[:2] + C/y[4]**2*y[:2] - C/y[4]*a[:2]
    
    g[0] = a[2]
    g[1] = a[3]
    g[2] = g_v[0]
    g[3] = g_v[1]
    g[4] = g_gamma
    
    return g

 
@njit(parallel = True)
def parallel_process(particle_list, A_list, Q_list, B, C, D, n, dz, n_cores, Q_tot, save_evolution):
    result = [np.empty_like(particle_list[0], dtype=np.float64) for _ in range(n_cores)]
    evolution = [np.zeros((10, n), dtype = np.float64) for _ in range(n_cores)]
    #Q_core = [np.sum(q) for q in Q_list]

    for j in prange(n_cores):
        mat = particle_list[j].copy()
        A = A_list[j]
        q = Q_list[j]
        Q_sum = np.sum(q)
        fs = [np.empty_like(mat, dtype=np.float64)for _ in range(2)]
        if save_evolution:      
            x_mean = np.sum(q*mat[0])/Q_sum
            y_mean = np.sum(q*mat[1])/Q_sum
            gamma_mean = np.sum(q*mat[4])/Q_sum
            xp_mean = np.sum(q*mat[2]*mat[4])/Q_sum
            yp_mean = np.sum(q*mat[3]*mat[4])/Q_sum
            # Sum of x, y
            evolution[j][0,0] = np.sum(q*mat[0])
            evolution[j][1,0] = np.sum(q*mat[1])
            
            # sum of (x-x_mean)**2, (y-y_mean)**2
            evolution[j][2,0] = np.sum(q*(mat[0]-x_mean)**2)
            evolution[j][3,0] = np.sum(q*(mat[1]-y_mean)**2)
            
            # Sum of gamma and (gamma-gamma_mean)**2
            evolution[j][4,0] = np.sum(q*mat[4])
            evolution[j][5,0] = np.sum(q*(mat[4]-gamma_mean)**2)
            
            #Sum of (xp-xp_mean)**2, (yp-yp_mean)**2
            evolution[j][6,0] = np.sum(q*(mat[2]*mat[4]-xp_mean)**2)
            evolution[j][7,0] = np.sum(q*(mat[3]*mat[4]-yp_mean)**2)
            
            # Sum of x-x_mean * xp-xp_mean (xp = vx*gamma = mat[2]*mat[4]), and same for y
            evolution[j][8,0] = np.sum(q*(mat[0]-x_mean)*q*(mat[2]*mat[4]-xp_mean))
            evolution[j][9,0] = np.sum(q*(mat[1]-y_mean)*q*(mat[3]*mat[4]-yp_mean))
            
            #Find use the derivatives at present, and future times
            a_present = acc_func(mat, A, B, C, D)
            fs[0] = a_present
            y_next = mat + a_present*dz
            a_next = acc_func(y_next, A, B, C, D)
            # Find acceleration at future time
            g = gs(y_next, a_next, A, B, C, D)
            #Use multistep formula with k=1 to evolve the particles to next time step.
            mat += dz*(2/3*a_next + 1/3*a_present) - 1/6*dz**2*g
            
            for i in range(1,n-1):
                # Get derivatives of present time
                a_present = acc_func(mat, A, B, C, D)
                fs[i%2] = a_present # store the present acceleration values for use in next timestep
                a_prev = fs[(i+1)%2] # get the accelerations from the previous timestep
                #get derivatives at next timestep
                y_next = mat + a_present*dz
                a_next = acc_func(y_next, A, B, C, D)
                # get accelerations at next timestep
                g = gs(y_next, a_next, A, B, C, D)
                # Use multistep formula with k=2 to evolve the particles
                mat += dz*(29/48*a_next + 5/12*a_present- 1/48*a_prev) - 1/8*dz**2*g               
                
                x_mean = np.sum(q*mat[0])/Q_sum
                y_mean = np.sum(q*mat[1])/Q_sum
                gamma_mean = np.sum(q*mat[4])/Q_sum
                xp_mean = np.sum(q*mat[2]*mat[4])/Q_sum
                yp_mean = np.sum(q*mat[3]*mat[4])/Q_sum
                # Sum of x, y
                evolution[j][0,i+1] = np.sum(q*mat[0])
                evolution[j][1,i+1] = np.sum(q*mat[1])
                
                # sum of (x-x_mean)**2, (y-y_mean)**2
                evolution[j][2,i+1] = np.sum(q*(mat[0]-x_mean)**2)
                evolution[j][3,i+1] = np.sum(q*(mat[1]-y_mean)**2)
            
                # Sum of gamma and (gamma-gamma_mean)**2
                evolution[j][4,i+1] = np.sum(q*mat[4])
                evolution[j][5,i+1] = np.sum(q*(mat[4]-gamma_mean)**2)
            
                #Sum of (xp-xp_mean)**2, (yp-yp_mean)**2
                evolution[j][6,i+1] = np.sum(q*(mat[2]*mat[4]-xp_mean)**2)
                evolution[j][7,i+1] = np.sum(q*(mat[3]*mat[4]-yp_mean)**2)
            
                # Sum of x-x_mean * xp-xp_mean (xp = vx*gamma = mat[2]*mat[4]), and same for y
                evolution[j][8,i+1] = np.sum(q*(mat[0]-x_mean)*q*(mat[2]*mat[4]-xp_mean))
                evolution[j][9,i+1] = np.sum(q*(mat[1]-y_mean)*q*(mat[3]*mat[4]-yp_mean))
            
        else:
            #Find use the derivatives at present, and future times
            a_present = acc_func(mat, A, B, C, D)
            fs[0] = a_present
            y_next = mat + a_present*dz
            a_next = acc_func(y_next, A, B, C, D)
            # Find acceleration at future time
            g = gs(y_next, a_next, A, B, C, D)
            #Use multistep formula with k=1 to evolve the particles to next time step.
            mat += dz*(2/3*a_next + 1/3*a_present) - 1/6*dz**2*g
            dz_sq = dz**2
            for i in range(1,n-1):
                # Get derivatives of present time
                a_present = acc_func(mat, A, B, C, D)
                fs[i%2] = a_present # store the present acceleration values for use in next timestep
                a_prev = fs[(i-1)%2] # get the accelerations from the previous timestep
                
                #get derivatives at next timestep
                y_next = mat + a_present*dz
                a_next = acc_func(y_next, A, B, C, D)
                
                # get accelerations at next timestep
                g = gs(y_next, a_next, A, B, C, D)
                
                # Use multistep formula with k=2 to evolve the particles            
                mat += dz*(29/48*a_next + 5/12*a_present- 1/48*a_prev) - 1/8*dz_sq*g
                
                
        result[j] = mat
        
    evolution_tot = evolution[0]
    finished_evolution = np.zeros((8, n), dtype = np.float64)
    if save_evolution:
        for i in range(n_cores-1):
            evolution_tot += evolution[i+1]
        # x and y offset
        finished_evolution[0] = evolution_tot[0]/Q_tot
        finished_evolution[1] = evolution_tot[1]/Q_tot
        
        # x and y beam size
        finished_evolution[2] = np.sqrt(evolution_tot[2]/Q_tot)
        finished_evolution[3] = np.sqrt(evolution_tot[3]/Q_tot)
        
        # Average energy and rel. energy spread
        finished_evolution[4] = evolution_tot[4]/Q_tot
        finished_evolution[5] = np.sqrt(evolution_tot[5]/Q_tot)
        
        # norm emittance in x and y
        finished_evolution[6] = abs(1/Q_tot)*np.sqrt(evolution_tot[2]*evolution_tot[6] - evolution_tot[8]**2)
        finished_evolution[7] = abs(1/Q_tot)*np.sqrt(evolution_tot[3]*evolution_tot[7] - evolution_tot[9]**2)

    
    return result, finished_evolution
          
def evolve_betatron_motion(qs, x0, y0, ux0, uy0, L, gamma, dgamma_ds, kp, save_evolution):
    # Constants
    #Ezs = Es/L #eV/m = J/e/m = V/m
    n_cores = max(1,min(16, round(len(x0)/10000)))
    print("Number of cores:", n_cores)
    
    #Ezs = dgamma_ds*SI.m_e*SI.c**2/SI.e #eV/m = J/e/m = V/m
    K_sq = kp**2/2
    
    C = K_sq # constant
    As = dgamma_ds # vector, dependant on Ez = E / L
    #Plasma constants
    
    #tau_r = 2*re/3/SI.c
    #Radiation reaction constants
    B = 2/3 * re* C # constant
    D = B * K_sq # constant

    #Find the smallest wavelength of oscillations to resolve
    beta_matched = np.sqrt(2*gamma)/kp # Vector
    lambda_beta = min(2*np.pi*beta_matched) # Vector
    n_per_beta = 150
    
    #Find the appropriate ammount of steps to resolve each oscillation    
    n = round(L/lambda_beta * n_per_beta)
    dz = L/(n-1)
    
    #velocities with respect to z instead of t
    vys = uy0/gamma/SI.c 
    vxs = ux0/gamma/SI.c

    Particles = np.zeros((5, len(x0)))
    Particles[0] = x0
    Particles[1] = y0
    Particles[2] = vxs
    Particles[3] = vys
    Particles[4] = gamma
    
    particle_list = np.array_split(Particles,n_cores, axis=1)
    Q_list = np.array_split(qs, n_cores)
    A_list = np.array_split(As, n_cores)
    Q_tot = np.sum(qs)

    #start = time.time()
    results, evolution = parallel_process(particle_list, A_list, Q_list, B, C, D, n, dz, n_cores, Q_tot, save_evolution)
    
    #end = time.time()
    #print('time = ', end-start, ' sec')
    if save_evolution:
        evolution[4] = evolution[4]*SI.m_e*SI.c**2/SI.e
        evolution[5] = evolution[5]*SI.m_e*SI.c**2/SI.e/evolution[4]
    location = np.linspace(0,L,n)
    
    
    solution = np.concatenate(results, axis=1)

    return solution[0], solution[1], solution[2] * solution[4]*SI.c, solution[3]* solution[4]*SI.c, solution[4]*SI.m_e*SI.c**2 / SI.e, evolution, location

"""
@njit(parallel = True)
def parallel_process(particle_list, A_list, Q_list, B, C, D, n, dz, n_cores, Q_tot, save_evolution):
    result = [np.empty_like(particle_list[0], dtype=np.float64) for _ in range(n_cores)]
    evolution = [np.zeros((10, n), dtype = np.float64) for _ in range(n_cores)]
    #Q_core = [np.sum(q) for q in Q_list]

    for j in prange(n_cores):
        mat = particle_list[j].copy()
        A = A_list[j]
        q = Q_list[j]
        Q_sum = np.sum(q)
        if save_evolution:      
            x_mean = np.sum(q*mat[0])/Q_sum
            y_mean = np.sum(q*mat[1])/Q_sum
            gamma_mean = np.sum(q*mat[4])/Q_sum
            xp_mean = np.sum(q*mat[2]*mat[4])/Q_sum
            yp_mean = np.sum(q*mat[3]*mat[4])/Q_sum
            # Sum of x, y
            evolution[j][0,0] = np.sum(q*mat[0])
            evolution[j][1,0] = np.sum(q*mat[1])
            
            # sum of (x-x_mean)**2, (y-y_mean)**2
            evolution[j][2,0] = np.sum(q*(mat[0]-x_mean)**2)
            evolution[j][3,0] = np.sum(q*(mat[1]-y_mean)**2)
            
            # Sum of gamma and (gamma-gamma_mean)**2
            evolution[j][4,0] = np.sum(q*mat[4])
            evolution[j][5,0] = np.sum(q*(mat[4]-gamma_mean)**2)
            
            #Sum of (xp-xp_mean)**2, (yp-yp_mean)**2
            evolution[j][6,0] = np.sum(q*(mat[2]*mat[4]-xp_mean)**2)
            evolution[j][7,0] = np.sum(q*(mat[3]*mat[4]-yp_mean)**2)
            
            # Sum of x-x_mean * xp-xp_mean (xp = vx*gamma = mat[2]*mat[4]), and same for y
            evolution[j][8,0] = np.sum(q*(mat[0]-x_mean)*q*(mat[2]*mat[4]-xp_mean))
            evolution[j][9,0] = np.sum(q*(mat[1]-y_mean)*q*(mat[3]*mat[4]-yp_mean))

            for i in range(n-1):
                k1 = acc_func(mat, A, B, C, D)
    
                k2 = acc_func(mat + k1 * dz/2, A, B, C, D)
                
                k3 = acc_func(mat + k2 * dz/2, A, B, C, D)
                
                k4 = acc_func(mat + k3 * dz, A, B, C, D)
                
                k_av = 1/6*(k1+2*k2+2*k3+k4)
                
                mat += k_av*dz                
                
                x_mean = np.sum(q*mat[0])/Q_sum
                y_mean = np.sum(q*mat[1])/Q_sum
                gamma_mean = np.sum(q*mat[4])/Q_sum
                xp_mean = np.sum(q*mat[2]*mat[4])/Q_sum
                yp_mean = np.sum(q*mat[3]*mat[4])/Q_sum
                # Sum of x, y
                evolution[j][0,i+1] = np.sum(q*mat[0])
                evolution[j][1,i+1] = np.sum(q*mat[1])
                
                # sum of (x-x_mean)**2, (y-y_mean)**2
                evolution[j][2,i+1] = np.sum(q*(mat[0]-x_mean)**2)
                evolution[j][3,i+1] = np.sum(q*(mat[1]-y_mean)**2)
            
                # Sum of gamma and (gamma-gamma_mean)**2
                evolution[j][4,i+1] = np.sum(q*mat[4])
                evolution[j][5,i+1] = np.sum(q*(mat[4]-gamma_mean)**2)
            
                #Sum of (xp-xp_mean)**2, (yp-yp_mean)**2
                evolution[j][6,i+1] = np.sum(q*(mat[2]*mat[4]-xp_mean)**2)
                evolution[j][7,i+1] = np.sum(q*(mat[3]*mat[4]-yp_mean)**2)
            
                # Sum of x-x_mean * xp-xp_mean (xp = vx*gamma = mat[2]*mat[4]), and same for y
                evolution[j][8,i+1] = np.sum(q*(mat[0]-x_mean)*q*(mat[2]*mat[4]-xp_mean))
                evolution[j][9,i+1] = np.sum(q*(mat[1]-y_mean)*q*(mat[3]*mat[4]-yp_mean))
            
        else:
            for i in range(n-1):
                k1 = acc_func(mat, A, B, C, D)
            
                k2 = acc_func(mat + k1 * dz/2, A, B, C, D)
                
                k3 = acc_func(mat + k2 * dz/2, A, B, C, D)
                
                k4 = acc_func(mat + k3 * dz, A, B, C, D)
                
                k_av = 1/6*(k1+2*k2+2*k3+k4)
                
                
                mat += k_av*dz
                
        result[j] = mat
        
    evolution_tot = evolution[0]
    finished_evolution = np.zeros((8, n), dtype = np.float64)
    if save_evolution:
        for i in range(n_cores-1):
            evolution_tot += evolution[i+1]
        # x and y offset
        finished_evolution[0] = evolution_tot[0]/Q_tot
        finished_evolution[1] = evolution_tot[1]/Q_tot
        
        # x and y beam size
        finished_evolution[2] = np.sqrt(evolution_tot[2]/Q_tot)
        finished_evolution[3] = np.sqrt(evolution_tot[3]/Q_tot)
        
        # Average energy and rel. energy spread
        finished_evolution[4] = evolution_tot[4]/Q_tot
        finished_evolution[5] = np.sqrt(evolution_tot[5]/Q_tot)
        
        # norm emittance in x and y
        finished_evolution[6] = abs(1/Q_tot)*np.sqrt(evolution_tot[2]*evolution_tot[6] - evolution_tot[8]**2)
        finished_evolution[7] = abs(1/Q_tot)*np.sqrt(evolution_tot[3]*evolution_tot[7] - evolution_tot[9]**2)

    
    return result, finished_evolution
          
def evolve_betatron_motion(qs, x0, y0, ux0, uy0, L, gamma, dgamma_ds, kp, save_evolution):
    # Constants
    #Ezs = Es/L #eV/m = J/e/m = V/m
    n_cores = max(1,min(16, round(len(x0)/10000)))
    print("Number of cores:", n_cores)
    
    #Ezs = dgamma_ds*SI.m_e*SI.c**2/SI.e #eV/m = J/e/m = V/m
    K_sq = kp**2/2
    
    C = K_sq # constant
    As = dgamma_ds # vector, dependant on Ez = E / L
    #Plasma constants
    
    #tau_r = 2*re/3/SI.c
    #Radiation reaction constants
    B = 2/3 * re* C # constant
    D = B * K_sq # constant

    #Find the smallest wavelength of oscillations to resolve
    beta_matched = np.sqrt(2*gamma)/kp # Vector
    lambda_beta = min(2*np.pi*beta_matched) # Vector
    n_per_beta = 200
    
    #Find the appropriate ammount of steps to resolve each oscillation    
    n = round(L/lambda_beta * n_per_beta)
    dz = L/(n-1)
    
    #velocities with respect to z instead of t
    vys = uy0/gamma/SI.c 
    vxs = ux0/gamma/SI.c

    Particles = np.zeros((5, len(x0)))
    Particles[0] = x0
    Particles[1] = y0
    Particles[2] = vxs
    Particles[3] = vys
    Particles[4] = gamma
    
    particle_list = np.array_split(Particles,n_cores, axis=1)
    Q_list = np.array_split(qs, n_cores)
    A_list = np.array_split(As, n_cores)
    Q_tot = np.sum(qs)

    #start = time.time()
    results, evolution = parallel_process(particle_list, A_list, Q_list, B, C, D, n, dz, n_cores, Q_tot, save_evolution)
    
    #end = time.time()
    #print('time = ', end-start, ' sec')
    if save_evolution:
        evolution[4] = evolution[4]*SI.m_e*SI.c**2/SI.e
        evolution[5] = evolution[5]*SI.m_e*SI.c**2/SI.e/evolution[4]
    location = np.linspace(0,L,n)
    
    
    solution = np.concatenate(results, axis=1)

    return solution[0], solution[1], solution[2] * solution[4]*SI.c, solution[3]* solution[4]*SI.c, solution[4]*SI.m_e*SI.c**2 / SI.e, evolution, location

"""




