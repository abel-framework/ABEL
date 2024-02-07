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

#For matrix of particles and coordinates
@njit("float64[:, :](float64[:, :], float64[::1], float64, float64, float64)")
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
    
@jit("Tuple((List(float64[:, ::1]), float64[:, ::1]))(List(float64[:, :], reflected=True), List(float64[::1], reflected=True), List(float64[::1], reflected=True), float64, float64, float64, int64, float64, int64, float64, boolean)", nopython=True, parallel=True)
def parallel_process(particle_list, A_list, Q_list, B, C, D, n, dz, n_cores, Q_tot, save_evolution):
    result = [np.empty_like(particle_list[0], dtype=np.float64) for _ in range(n_cores)]
    evolution = [np.zeros((10, n), dtype = np.float64) for _ in range(n_cores)]
    #Q_core = [np.sum(q) for q in Q_list]
    for j in prange(n_cores):
        mat = particle_list[j].copy()
        A = A_list[j]
        q = Q_list[j]
        if save_evolution:
            x_sum = np.sum(q*mat[0])
            y_sum = np.sum(q*mat[1])
            Q_sum = np.sum(q)
            
            # Sum of x, y and gamma
            evolution[j][0,0] = x_sum
            evolution[j][1,0] = y_sum
            evolution[j][2,0] = np.sum(q*mat[4])
            
            # Sum of (gamma-gamma_mean)**2, (x-x_mean)**2, (y-y_mean)**2
            evolution[j][3,0] = np.sum(q*(mat[4]-np.sum(q*mat[4])/Q_sum)**2)
            evolution[j][4,0] = np.sum(q*(mat[0]-x_sum/Q_sum)**2)
            evolution[j][5,0] = np.sum(q*(mat[1]-y_sum/Q_sum)**2)
            
            #Sum of (xp-xp_mean)**2, (yp-yp_mean)**2
            evolution[j][6,0] = np.sum(q*(mat[2]*mat[4]-np.sum(q*mat[2]*mat[4])/Q_sum)**2)
            evolution[j][7,0] = np.sum(q*(mat[3]*mat[4]-np.sum(q*mat[3]*mat[4])/Q_sum)**2)
            
            # Sum of x-x_mean * xp-xp_mean (xp = vx*gamma = mat[2]*mat[4]), and same for y
            evolution[j][8,0] = np.sum(q*(mat[0]-x_sum/Q_sum)*q*(mat[2]*mat[4]-np.sum(q*mat[2]*mat[4])/Q_sum))
            evolution[j][9,0] = np.sum(q*(mat[1]-y_sum/Q_sum)*q*(mat[3]*mat[4]-np.sum(q*mat[3]*mat[4])/Q_sum))

            for i in range(n-1):
                k1 = acc_func(mat, A, B, C, D)
    
                k2 = acc_func(mat + k1 * dz/2, A, B, C, D)
                
                k3 = acc_func(mat + k2 * dz/2, A, B, C, D)
                
                k4 = acc_func(mat + k3 * dz, A, B, C, D)
                
                k_av = 1/6*(k1+2*k2+2*k3+k4)
                
                mat += k_av*dz
 
                # Sum of x, y and gamma
                x_sum = np.sum(q*mat[0])
                y_sum = np.sum(q*mat[1])
                evolution[j][0,i+1] = x_sum
                evolution[j][1,i+1] = y_sum
                evolution[j][2,i+1] = np.sum(q*mat[4])
                
                # Sum of (gamma-gamma_mean)**2, (x-x_mean)**2, (y-y_mean)**2
                evolution[j][3,i+1] = np.sum(q*(mat[4]-np.sum(q*mat[4])/Q_sum)**2)
                evolution[j][4,i+1] = np.sum(q*(mat[0]-x_sum/Q_sum)**2)
                evolution[j][5,i+1] = np.sum(q*(mat[1]-y_sum/Q_sum)**2)
                
                #Sum of (xp-xp_mean)**2, (yp-yp_mean)**2
                evolution[j][6,i+1] = np.sum(q*(mat[2]*mat[4]-np.sum(q*mat[2]*mat[4])/Q_sum)**2)
                evolution[j][7,i+1] = np.sum(q*(mat[3]*mat[4]-np.sum(q*mat[3]*mat[4])/Q_sum)**2)
                
                # Sum of (x-x_mean) * (xp-xp_mean) (xp = vx*gamma = mat[2]*mat[4]), same for y
                evolution[j][8,i+1] = np.sum(q*(mat[0]-x_sum/Q_sum)*q*(mat[2]*mat[4]-np.sum(q*mat[2]*mat[4])/Q_sum))
                evolution[j][9,i+1] = np.sum(q*(mat[1]-y_sum/Q_sum)*q*(mat[3]*mat[4]-np.sum(q*mat[3]*mat[4])/Q_sum))
            
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
        # Average energy and rel. energy spread
        finished_evolution[2] = evolution_tot[2]/Q_tot
        finished_evolution[3] = np.sqrt(evolution_tot[3]/Q_tot)
        # norm emittance in x and y
        finished_evolution[4] = abs(1/Q_tot)*np.sqrt(evolution_tot[4]*evolution_tot[6] - evolution_tot[8]**2)
        finished_evolution[5] = abs(1/Q_tot)*np.sqrt(evolution_tot[5]*evolution_tot[7] - evolution_tot[9]**2)
        # Beam size in x and y
        finished_evolution[6] = np.sqrt(evolution_tot[4]/Q_tot)
        finished_evolution[7] = np.sqrt(evolution_tot[5]/Q_tot)
    
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

    start = time.time()
    results, evolution = parallel_process(particle_list, A_list, Q_list, B, C, D, n, dz, n_cores, Q_tot, save_evolution)
    end = time.time()
    print('time = ', end-start, ' sec')
    if save_evolution:
        evolution[2] = evolution[2]*SI.m_e*SI.c**2/SI.e
        evolution[3] = evolution[3]*SI.m_e*SI.c**2/SI.e/evolution[2]
    location = np.linspace(0,L,n)
    
    
    solution = np.concatenate(results, axis=1)

    return solution[0], solution[1], solution[2] * solution[4]*SI.c, solution[3]* solution[4]*SI.c, solution[4]*SI.m_e*SI.c**2 / SI.e, evolution, location
    
"""
# Calculate acceleration and change in gamma
def oscillator2d(t, x, A, B, C, D):
    # x= [x,vx, y, vy, gamma]
    x_, vx_, y_, vy_, gamma_ = x

    # Calculate acceleration in x and y
    a_x = -(A/gamma_ + B)*vx_ - C/gamma_*x_
    a_y = -(A/gamma_ + B)*vy_ - C/gamma_*y_

    # Calculate change in gamma per second
    d_gamma = A - D*gamma_**2 * (x_**2 + y_**2)
    
    return np.array([vx_, a_x, vy_, a_y, d_gamma])

# Assume x0 etc is a vector containing appropriate value for each particle
#def evolve_betatron_motion(x0, ux0, y0, uy0, L, gamma, kp, Es):
def evolve_betatron_motion(x0, y0, ux0, uy0, L, gamma, dgamma_ds, kp):
    # Constants
    #Ezs = Es/L #eV/m = J/e/m = V/m
    Ezs = dgamma_ds*SI.m_e*SI.c**2/SI.e #eV/m = J/e/m = V/m
    
    K = kp/ np.sqrt(2)
    C = SI.c**2 * K**2 # constant
    As = SI.e/SI.m_e/SI.c * Ezs # vector, dependant on Ez = E / L
    #Plasma constants
    tau_r = 2*re/3/SI.c
    B = tau_r * C # constant
    D = B * K**2 # constant


    #Find the smallest wavelength of oscillations to resolve
    beta_matched = np.sqrt(2*gamma)/kp # Vector
    lambda_beta = min(2*np.pi*beta_matched) # Vector
    n_per_beta = 200
    #Find the appropriate ammount of steps to resolve each oscillation
    T = L/SI.c    
    n = round(L/lambda_beta * n_per_beta)
    t = np.linspace(0,T,n)

    length = x0.size
    
    vys = uy0/gamma
    vxs = ux0/gamma
    n_cores = 24
    # Solve the equation of motion for each particle, and loop over As as well, as it is different for each particle
    def parallel_process(i):
        x_, vx_, y_, vy_, gamma_, A_ = x0[i], vxs[i], y0[i], vys[i], gamma[i], As[i]

        # Find initial velocity
        # u = p/SI.m_e, dividing by gamma gives velocity
        # Initial values
        sysinits = np.array([x_, vx_, y_, vy_, gamma_])
        # Solve the radiation reaction eqaution of motion
        solution = solve_ivp(fun = oscillator2d, y0 = sysinits, method='RK45', \
                             t_span = (0,T), t_eval = t, args = (A_, B, C, D))
    
        x_end = solution.y[0, -1]
        #xs[i] = x_end
        
        vx_end = solution.y[1, -1]
        
        y_end = solution.y[2, -1]
        #ys[i] = y_end
        
        vy_end = solution.y[3, -1]
        
        gamma_end = solution.y[4, -1]
        
        ux_end = vx_end * gamma_end
        #uxs[i] = ux_end
        
        uy_end = vy_end * gamma_end
        #uys[i] = uy_end
        
        #if i ==0:
        #    print(vx_end, vy_end)
        #    print(x_dot0, y_end)
        #    plt.plot(t, solution.y[0,:])
        #    plt.show()

        E = gamma_end*SI.m_e*SI.c**2 / SI.e # eV
        #Es[i] = E
        return x_end, ux_end, y_end, uy_end, E
    
    res = np.array(Parallel(n_jobs=n_cores)(delayed(parallel_process)(i) for i in range(length)))

    xs, uxs, ys, uys, Es = res[:,0], res[:,1], res[:,2], res[:,3], res[:,4]

    return xs, ys, uxs, uys, Es

"""









