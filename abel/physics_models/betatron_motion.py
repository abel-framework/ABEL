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
    a_x = -(A/gamma_ + B)*vx_ - C/gamma_*x_
    a_y = -(A/gamma_ + B)*vy_ - C/gamma_*y_

    # Calculate change in gamma per second
    d_gamma = A - D*gamma_**2 * (x_**2 + y_**2)
    
    return np.array([vx_, a_x, vy_, a_y, d_gamma])

#For matrix of particles and coordinates
def acc_func(y, A, B, C, D):
    # y =[[x, y, vx, vy, gamma]
    #     [same for nesxt part] ...]
    dy = np.zeros(y.shape) # [[vx, vy, ax, xy, dgamma] ...]
    ax = -(A/y[:,-1] + B)*y[:,2] - C/y[:,-1]*y[:,0]
    ay = -(A/y[:,-1] + B)*y[:,3] - C/y[:,-1]*y[:,1]
    dgamma = A - D*y[:,-1]**2*(y[:,0]**2 + y[:,1]**2)
    return np.c_[y[:,2], y[:,3], ax, ay, dgamma]

"""
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
    n_per_beta = 500
    #Find the appropriate ammount of steps to resolve each oscillation
    T = L/SI.c    
    n = round(L/lambda_beta * n_per_beta)
    t = np.linspace(0,T,n)

    length = x0.size
    
    vys = uy0/gamma
    vxs = ux0/gamma
    
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

    res = np.array(Parallel(n_jobs=15)(delayed(parallel_process)(i) for i in range(length)))

    xs, uxs, ys, uys, Es = res[:,0], res[:,1], res[:,2], res[:,3], res[:,4]

    return xs, ys, uxs, uys, Es

"""
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
    n_per_beta = 500
    #Find the appropriate ammount of steps to resolve each oscillation
    T = L/SI.c    
    n = round(L/lambda_beta * n_per_beta)
    t, dt = np.linspace(0,T,n, retstep = True)
    
    vys = uy0/gamma
    vxs = ux0/gamma
    
    y = np.c_[x0, y0, vxs, vys, gamma]
    solution = np.zeros((y.shape[0], y.shape[1], t.size))
    solution[:,:,0] = y
    for i in range(t.size -1):
        y_ = solution[:,:,i]
        k1 = acc_func(y_, As, B, C, D)
        
        k2 = acc_func(y_ + k1 * dt/2, As, B, C, D)
        
        k3 = acc_func(y_ + k2 * dt/2, As, B, C, D)

        k4 = acc_func(y_ + k3 * dt, As, B, C, D)
        
        k_av = 1/6*(k1+2*k2+2*k3+k4)
        
        solution[:,:,i+1] = y_ + k_av*dt

    # xs, uxs, ys, uys, Es = solution[:,0,-1], solution[:,1,-1], solution[:,2,-1], solution[:,3,-1], solution[:,4,-1]
    return solution[:,0,-1], solution[:,1,-1], solution[:,2,-1] * solution[:,4,-1], \
    solution[:,3,-1]* solution[:,4,-1], solution[:,4,-1]*SI.m_e*SI.c**2 / SI.e

    





























