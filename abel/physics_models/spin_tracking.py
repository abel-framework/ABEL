from abel import *
import numpy as np
import scipy.constants as SI
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from abel.utilities.relativity import gamma2proper_velocity, gamma2energy
from abel.physics_models import ode_Hills as ode_hill_system

SI.r_e = SI.physical_constants['classical electron radius'][0]

"""
Simulates particle motion and tracks spin motion using the T-BMT equation
"""

#def ode_hill_system(u, s, gamma, kp):
    #Defines the differential system for transverse motion
#    x, ux = u #unpacks transverese position and velocity
#    return [ux / (gamma(s) * SI.c), -x * (SI.c / 2.0) * kp(s)**2] #returns time derivative of position and velocity using Hills eq

def evolve_hills_numeric(x0, ux0, L, gamma, kp, steps_per_meter=100):
    #simulates transverese particle motion through a focusing lattice
    Nstep = max(int(L * steps_per_meter), 10) #sets number of integration steps
    ss = np.linspace(0, L, Nstep) #creates array of longitudinal positions
    u0 = [x0, ux0] #initial condition vector
    sol = odeint(ode_hill_system, u0, ss, args=(u0, ss, gamma, kp)) #solves the ODE system over path s
    x = sol[:, 0]
    ux = sol[:, 1] #extracts position and velocity from solution
    return ss, x, ux #returns longitudinal positions, transverese posotions and velocities
    
def compute_fields(ss, x, ux, lattice):
    #Computes fields along the particle's path
    gamma_vals = np.array([lattice.gamma(si) for si in ss]) #evaluates relativistic gamma at each position s
    beta_x = ux / (gamma2proper_velocity(gamma_vals) * SI.c) #computes normalized transverese velocity betax
    total_beta2 = 1.0 - 1.0 / gamma_vals**2
    beta_z = np.sqrt(np.clip(total_beta2 - beta_x**2, 0.0, None)) #computes longitudinal velcoty B^2 = Bx^2 + Bz^2

    ds = np.diff(ss)
    dt_steps = ds / (beta_z[:-1] * SI.c)
    ts = np.concatenate(([0.0], np.cumsum(dt_steps))) #calculates time steps dt from distances and Bz, compute absolut time

    Es = []
    Bs = [] #initializes electric and magnetic field arrays
    for xi, si, ti in zip(x, ss, ts):
        r_vec = np.array([xi, 0.0, si])
        Es.append(lattice.E_field(r_vec, ti))
        Bs.append(lattice.B_field(r_vec, ti)) #loops through positions and times to calculate E and B fields using lattice
    return np.array(Es), np.array(Bs)

def tbmt_boris_spin_update(S, E, B, beta_vec, gamma, dt):
    #Updates spin vector s using T-BMT precession
    q = SI.e 
    m = SI.m_e
    term1 = B / gamma
    term2 = -np.cross(beta_vec, E) / (SI.c**2)
    term3 = (gamma / (gamma + 1.0)) * np.dot(beta_vec, B) * beta_vec
    Omega = - (q / m) * (term1 + term2 + term3) #Computes angular velocity of spin precession

    theta = np.linalg.norm(Omega) * dt
    if theta == 0.0:
        return S.copy() #determines rotation angle; exits early if no precession
    u = Omega / np.linalg.norm(Omega)
    S_par = np.dot(S, u) * u
    S_perp = S - S_par
    S_rot = S_perp * np.cos(theta) + np.cross(u, S) * np.sin(theta) + S_par #performs 3D rotation of spin vector around axis u
    print(f"Beta: {beta_vec}, E: {E}, B: {B}")
    print(f"Omega: {Omega}, theta: {theta}")
    return S_rot #returns updated spin vector

def track_spin(ss, x, ux, Es, Bs, lattice, S0):
    #tracks spin vector along the trajectory
    N = len(ss)
    S_hist = np.zeros((N, 3))
    S_hist[0] = S0 #initializes spin history and sets initial spin

    gamma_vals = np.array([lattice.gamma(si) for si in ss])
    beta_x = ux / (gamma2proper_velocity(gamma_vals) * SI.c)
    total_beta2 = 1.0 - 1.0 / gamma_vals**2
    beta_z = np.sqrt(np.clip(total_beta2 - beta_x**2, 0.0, None)) #calculates gamma, Bx, Bz

    ds = np.diff(ss)
    dt_steps = ds / (beta_z[:-1] * SI.c)
    dt_arr = np.concatenate(([dt_steps[0]], dt_steps)) #computes time steps

    for i in range(1, N):
        S_prev = S_hist[i-1]
        E = Es[i-1]
        B = Bs[i-1]
        beta_vec = np.array([beta_x[i-1], 0.0, beta_z[i-1]])
        gamma = gamma_vals[i-1]
        dt = dt_arr[i-1]
        S_hist[i] = tbmt_boris_spin_update(S_prev, E, B, beta_vec, gamma, dt) #iteratively updates spin vector using T-BMT boris alogorith
    return S_hist #returns complete spin trajectory


def plot_spin_tracking(all_spins, ss):
    plt.figure(figsize=(10, 6))
    for i, spins in enumerate(all_spins):
        plt.plot(ss, spins[:, 0], label=f"Spin X (Particle {i+1})")
        plt.plot(ss, spins[:, 1], label=f"Spin Y (Particle {i+1})")
        plt.plot(ss, spins[:, 2], label=f"Spin Z (Particle {i+1})")

    plt.title("Spin Tracking of Particles")
    plt.xlabel("Stage Length (m)")
    plt.ylabel("Spin Components")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


        