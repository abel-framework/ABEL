from abel import *
import numpy as np
import scipy.constants as SI
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from abel.physics_models import *
from abel.utilities.relativity import *
import scipy.special as scispec

SI.r_e = SI.physical_constants['classical electron radius'][0]

"""
Simulates particle motion and tracks spin motion using the T-BMT equation
"""


def evolve_hills_equation_analytic_evolution(x0, ux0, L, gamma0, dgamma_ds, kp=None, g=None, N=500):
    s = np.linspace(0, L, N)

    xp0 = ux0 / gamma2proper_velocity(gamma0)
    
    if dgamma_ds == 0:
        if g is None:
            g = kp ** 2 * SI.m_e * SI.c / (2 * SI.e)

        k = g * SI.c / gamma2energy(gamma0)

        if k == 0:
            x = x0 + xp0 * s
            xp = np.full_like(s, xp0)
        else:
            x = np.real(
                x0 * np.cos(k * s) + (xp0 / k) * np.sin(k * s)
            )
            xp = np.real(
                xp0 * np.cos(k * s) - x0 * k * np.sin(k * s)
            )

        gamma = np.full_like(s, gamma0)

    else:
        if kp is None:
            kp = np.sqrt(2 * g * SI.e / (SI.m_e * SI.c))

        gamma = gamma0 + dgamma_ds * s
        C = np.sqrt(2) * kp / dgamma_ds
        A0 = C * np.sqrt(gamma0)
        A = C * np.sqrt(gamma)

        Di = (kp**2 * x0 * scispec.iv(1, A0 * 1j) + A0 * 1j * dgamma_ds * xp0 * scispec.iv(0, A0 * 1j))
        Dk = (kp**2 * x0 * scispec.kv(1, A0 * 1j) - A0 * 1j * dgamma_ds * xp0 * scispec.kv(0, A0 * 1j))
        E = kp**2 * (scispec.iv(1, A0 * 1j) * scispec.kv(0, A0 * 1j) +
                     scispec.iv(0, A0 * 1j) * scispec.kv(1, A0 * 1j)) * 1j

        x = np.real(1j * (Di * scispec.kv(0, A * 1j) + Dk * scispec.iv(0, A * 1j)) / E)
        xp = -np.real((dgamma_ds * C**2 / (2 * A * E)) *
                      (Dk * scispec.iv(1, A * 1j) - Di * scispec.kv(1, A * 1j)))

    ux = xp * gamma2proper_velocity(gamma)
    
    return x, ux, gamma



def plasma_E_field(r_vec, t, k_p, Ez0=0.0):
    x, y, _ = r_vec
    Er_coeff = SI.m_e * (k_p**2) * SI.c**2 / (2*SI.e)
    Ex = Er_coeff * x
    Ey = Er_coeff * y
    Ez = Ez0  # set from the accelerating gradient below
    return np.array([Ex, Ey, Ez], dtype=float)

def plasma_B_field(r_vec, t, k_p, beta_z=1.0):
    x, y, _ = r_vec
    Br_coeff = (SI.m_e * k_p**2 * SI.c / (2*SI.e)) * beta_z
    Bx = -Br_coeff * y
    By =  Br_coeff * x
    Bz = 0.0
    return np.array([Bx, By, Bz], dtype=float)
    
def tbmt_boris_spin_update(S, E, B, beta_vec, gamma, dt):
    q = - SI.e
    m = SI.m_e
    a = 0.00115965218128

    # Ensure all inputs are proper scalars or 1D arrays
    dt = float(np.ravel(dt)[0])

    S = np.asarray(S, dtype=np.float64).reshape(3)
    E = np.asarray(E, dtype=np.float64).reshape(3)
    B = np.asarray(B, dtype=np.float64).reshape(3)
    beta_vec = np.asarray(beta_vec, dtype=np.float64).reshape(3)
    gamma = float(gamma)
    
    term1 = (a + 1.0 / gamma)*SI.c * B
    term2 = - (a + 1.0/(gamma+1.0)) * np.cross(beta_vec, E)
    term3 = - a*(gamma / (gamma + 1.0)) * (np.dot(beta_vec*SI.c , B) * beta_vec)
    Omega = - (q / m) * (term1 + term2 + term3)/SI.c

    w = np.linalg.norm(Omega)
    if w == 0 or dt == 0:
        return S.copy()

    n = Omega / w
    theta = w * dt
    c, s = np.cos(theta), np.sin(theta)

    # Rodrigues rotation of S about axis n by angle theta
    S_rot = S * c + np.cross(n, S) * s + n * np.dot(n, S) * (1 - c)
    return S_rot


    #return S+np.cross(Omega, S) * dt
"""
def plot_spin_tracking(all_spins, ss):
    plt.figure(figsize=(10, 6))
    for i, spins in enumerate(all_spins):
        plt.plot(ss, spins[:, 0], label=f"Spin X (Particle {i+1})")
        plt.plot(ss, spins[:, 1], label=f"Spin Y (Particle {i+1})")
        plt.plot(ss, spins[:, 2], label=f"Spin Z (Particle {i+1})")
        break

    plt.title("Spin Tracking of Particles")
    plt.xlabel("Stage Length (m)")
    plt.ylabel("Spin Components")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
"""
def plot_spin_tracking(all_spins, ss):
    if len(all_spins) == 0:
        return
    plt.figure(figsize=(10, 6))
    spins = all_spins[10]
    #plt.plot(ss, spins[:, 0], label="Spin X")
    #plt.plot(ss, spins[:, 1], label="Spin Y")
    plt.plot(ss, spins[:, 2], label="Spin Z")
    plt.title("Spin Tracking (first particle)")
    plt.xlabel("Stage Length (m)")
    plt.ylabel("Spin Components")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
    
def plot_spin_tracking_gamma(all_spins, gammas):
    plt.figure(figsize=(10, 6))
    for i, spins in enumerate(all_spins):
        #plt.plot(gammas, spins[:, 0], label=f"Spin X (Particle {i+1})")
        #plt.plot(gammas, spins[:, 1], label=f"Spin Y (Particle {i+1})")
        plt.plot(gammas, spins[:, 2], label=f"Spin Z (Particle {i+1})")
        break 

    plt.title("Spin Tracking of Particles Over Gamma")
    plt.xlabel("Lorentz Factor (γ)")
    plt.ylabel("Spin Components")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


        