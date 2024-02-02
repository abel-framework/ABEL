import numpy as np
from scipy.integrate import solve_ivp, odeint
import scipy.constants as SI
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numba import jit, prange
from types import SimpleNamespace
import multiprocessing
import os
import time
num_cores = multiprocessing.cpu_count()
os.environ['NUMEXPR_MAX_THREADS'] = f'{num_cores}'

re = SI.physical_constants['classical electron radius'][0]

#For matrix of particles and coordinates
@jit(nopython=True)
def acc_func(ys, A, B, C, D):
    # ys =[[x0, x1, x2, ...           ]
    #     [y0, y1, y2, ...            ]
    #     [vx0, vx1, vx2, ...         ]
    #     [vy0, vy1, vy2, ...         ]
    #     [gamma0, gamma1, gamma2, ...]]
    ax = -(A/ys[4] + B)*ys[2] - C/ys[4]*ys[0]
    ay = -(A/ys[4] + B)*ys[3] - C/ys[4]*ys[1]
    dgamma = A - D*ys[4]**2*(ys[0]**2 + ys[1]**2)

    result = np.empty_like(ys, dtype = np.float64)
    result[0] = ys[2]
    result[1] = ys[3]
    result[2] = ax
    result[3] = ay
    result[4] = dgamma
    
    return result
    
@jit(nopython=False, parallel=True)
def parallel_process(particle_list, A_list, Q_list, B, C, D, n, dz, n_cores, Q_tot, save_evolution = False):
    result = [np.empty_like(particle_list[0], dtype=np.float64) for _ in range(n_cores)]
    evolution = [np.zeros((4, n)) for _ in range(n_cores)]
    Q_core = [np.sum(q) for q in Q_list]
    for j in prange(n_cores):
        mat = particle_list[j].copy()
        A = A_list[j]
        q = Q_list[j]
        if save_evolution:
            evolution[j][0,0] = np.sum(q*mat[0])
            evolution[j][1,0] = np.sum(q*mat[1])
            evolution[j][2,0] = np.sum(q*mat[4])
            evolution[j][3,0] = np.sum(q*mat[4]**2)
            for i in range(n-1):
                k1 = acc_func(mat, A, B, C, D)
    
                k2 = acc_func(mat + k1 * dz/2, A, B, C, D)
                
                k3 = acc_func(mat + k2 * dz/2, A, B, C, D)
                
                k4 = acc_func(mat + k3 * dz, A, B, C, D)
                
                k_av = 1/6*(k1+2*k2+2*k3+k4)
                
                mat += k_av*dz
 
                evolution[j][0,i+1] = np.sum(q*mat[0])
                evolution[j][1,i+1] = np.sum(q*mat[1])
                evolution[j][2,i+1] = np.sum(q*mat[4])
                evolution[j][3,i+1] = np.sum(q*mat[4]**2)
        else:
            for i in range(n-1):
                k1 = acc_func(mat, A, B, C, D)
            
                k2 = acc_func(mat + k1 * dz/2, A, B, C, D)
                
                k3 = acc_func(mat + k2 * dz/2, A, B, C, D)
                
                k4 = acc_func(mat + k3 * dz, A, B, C, D)
                
                k_av = 1/6*(k1+2*k2+2*k3+k4)
                
                mat += k_av*dz
                
        result[j] = mat

    if save_evolution:
        evolution_tot = evolution[0]
        for i in range(n_cores-1):
            evolution_tot += evolution[i+1]
        evolution_tot[0] = evolution_tot[0]/Q_tot
        evolution_tot[1] = evolution_tot[1]/Q_tot
        evolution_tot[2] = evolution_tot[2]/Q_tot
        evolution_tot[3] = np.sqrt(evolution_tot[3])/(Q_tot-1)
    return result, evolution
    
#@jit(nopython=True)           
def evolve_betatron_motion(qs, x0, y0, ux0, uy0, L, gamma, dgamma_ds, kp, save_evolution = False):
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
    n_per_beta = 200
    
    #Find the appropriate ammount of steps to resolve each oscillation    
    n = round(L/lambda_beta * n_per_beta)
    dz = L/(n-1)
    
    vys = uy0/gamma/SI.c #velocity with respect to z instead of t
    vxs = ux0/gamma/SI.c

    Particles = np.zeros((5, len(x0)))#, n))
    Particles[:,:] = [x0, y0, vxs, vys, gamma]
    
    particle_list = np.array_split(Particles,n_cores, axis=1)#split_array, axis = 1)
    Q_list = np.array_split(qs, n_cores)
    A_list = np.array_split(As, n_cores)
    Q_tot = np.sum(qs)
    start = time.time()
    results, evolution = parallel_process(particle_list, A_list, Q_list, B, C, D, n, dz, n_cores, Q_tot, save_evolution = save_evolution)
    #print(results)
    end = time.time()
    print('time = ',end-start, ' sec')
    evolution = np.concatenate(evolution, axis =1)
    
    evolution_ = SimpleNamespace()
    evolution_.x = evolution[0]
    evolution_.y = evolution[1]
    evolution_.energy = evolution[2]*SI.m_e*SI.c**2/SI.e
    
    
    solution = np.concatenate(results, axis = 1)
    # xs, uxs, ys, uys, Es = solution[:,0,-1], solution[:,1,-1], solution[:,2,-1], solution[:,3,-1], solution[:,4,-1]
    return solution[0], solution[1], solution[2] * solution[4]*SI.c, solution[3]* solution[4]*SI.c, solution[4]*SI.m_e*SI.c**2 / SI.e, evolution_
    
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









