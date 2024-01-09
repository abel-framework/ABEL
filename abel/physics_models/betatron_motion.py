import numpy as np
from scipy.integrate import solve_ivp, odeint
import scipy.constants as SI
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import os

num_cores = multiprocessing.cpu_count()
os.environ['NUMEXPR_MAX_THREADS'] = f'{num_cores}'

re = SI.physical_constants['classical electron radius'][0]

# Calculate acceleration and change in gamma
def oscillator2d(t, x, A, B, C, D):
    # x= [x,vx, y, vy, gamma]
    x_, vx_, y_, vy_, gamma_ = x

    # Calculate acceleration in x and y
    a_x = -(C/gamma_ + A)*vx_ - B/gamma_*x_
    a_y = -(C/gamma_ + A)*vy_ - B/gamma_*y_

    # Calculate change in gamma per second
    d_gamma = C - D*gamma_**2 * (x_**2 + y_**2)
    
    return np.array([vx_, a_x, vy_, a_y, d_gamma])

# Assume x0 etc is a vector containing appropriate value for each particle
#def evolve_betatron_motion(x0, ux0, y0, uy0, L, gamma, kp, Es):
def evolve_betatron_motion(x0, ux0, y0, uy0, L, gamma, dgamma_ds, kp, enable_rr = True):
    # Constants
    #Ezs = Es/L #eV/m = J/e/m = V/m
    Ezs = dgamma_ds*SI.m_e*SI.c**2/SI.e #eV/m = J/e/m = V/m
    
    K = kp/ np.sqrt(2)
    B = SI.c**2 * K**2 # constant
    Cs = SI.e/SI.m_e/SI.c * Ezs # vector, dependant on Ez = E / L
    #Plasma constants
    if enable_rr:
        tau_r = 2*re/3/SI.c
        A = tau_r * SI.c**2 * K**2 # constant
        D = tau_r * SI.c**2 * K**4 # constant
    else:
        tau_r = 0
        A = 0
        D = 0


    #Find the smallest wavelength of oscillations to resolve
    beta_matched = np.sqrt(2*gamma)/kp # Vector
    lambda_beta = 2*np.pi*beta_matched # Vector

    #Find the appropriate ammount of steps to resolve each oscillation
    T = L/SI.c
    n_per_beta = 50

    length = len(x0)
    
    vys = uy0/gamma
    vxs = ux0/gamma    
    
    # Solve the equation of motion for each particle, and loop over Cs as well, as it is different for each particle
    
    def parallel_process(i):
        x_, vx_, y_, vy_, gamma_, C_, lambda_beta_ = x0[i], vxs[i], y0[i], vys[i], gamma[i], Cs[i], lambda_beta[i]
        
        n = round(L/lambda_beta_ * n_per_beta)
        t = np.linspace(0,T,n)
        # Find initial velocity
        # u = p/SI.m_e, dividing by gamma gives velocity
        # Initial values
        sysinits = np.array([x_, vx_, y_, vy_, gamma_])
        # Solve the radiation reaction eqaution of motion
        solution = solve_ivp(fun = oscillator2d, y0 = sysinits, method='RK45', \
                             t_span = (0,T), t_eval = t, args = (A, B, C_, D))
    
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

    res = np.array(Parallel(n_jobs=15)(delayed(parallel_process)(i) for i in range(length)))

    xs, uxs, ys, uys, Es = res[:,0], res[:,1], res[:,2], res[:,3], res[:,4]

    return xs, uxs, ys, uys, Es


