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

def evolve_hills_equation_analytic(x0, ux0, L, gamma0, dgamma_ds, kp=None, g=None, N=500):
    s = np.linspace(0, L, N)

    xp0 = ux0 / gamma2proper_velocity(gamma0)

    if dgamma_ds == 0:
        if g is None:
            g = kp**2 * SI.m_e * SI.c / (2 * SI.e)

        k = g * SI.c / gamma2energy(gamma0)

        if k == 0:
            x = x0 + xp0 * s
            xp = np.full_like(s, xp0)
        else:
            x = np.real(x0 * np.cos(k * s) + (xp0 / k) * np.sin(k * s))
            xp = np.real(xp0 * np.cos(k * s) - x0 * k * np.sin(k * s))

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


def plasma_E_field(r_vec, t, k_p): 
    x, y, _ = r_vec
    Ex = SI.m_e * (k_p ** 2) * x / SI.e
    Ey = SI.m_e * (k_p ** 2) * y / SI.e
    return np.array([Ex, Ey, 0.0])

def plasma_B_field(r_vec, t, k_p):
    Ex, Ey, _ = plasma_E_field(r_vec, t, k_p)
    By = Ex / SI.c
    Bx = -Ey / SI.c
    return np.array([Bx, By, 0.0])

    
def tbmt_boris_spin_update(S, E, B, beta_vec, gamma, dt):
    q = SI.e
    m = SI.m_e

    # Ensure all inputs are proper scalars or 1D arrays
    dt = float(np.ravel(dt)[0])
    S = np.asarray(S, dtype=np.float64).reshape(3)
    E = np.asarray(E, dtype=np.float64).reshape(3)
    B = np.asarray(B, dtype=np.float64).reshape(3)
    beta_vec = np.asarray(beta_vec, dtype=np.float64).reshape(3)
    gamma = float(gamma)

    a = 0.00115965218128
    term1 = B / gamma
    term2 = -np.cross(beta_vec, E) / (SI.c ** 2)
    term3 = (gamma / (gamma + 1.0)) * (np.dot(beta_vec, B) * beta_vec)
    Omega = - (q / m) * (term1 + term2 + term3) * (1 + a)
   

    omega_mag = np.linalg.norm(Omega)
    if np.isclose(omega_mag, 0):
        return S.copy()

    theta = omega_mag * dt
    u = Omega / omega_mag #unit vector in the direction of omega

    S_par = np.dot(S, u) * u #component of s parallel to the rotation axis, projecting s onto u
    S_perp = S - S_par #component of s perpendicular to the rotation axis
    S_rot = S_perp * np.cos(theta) + np.cross(u, S) * np.sin(theta) + S_par
    #Rotating the perpendicular component by the angle theta, crossing to get the part of S that contributes to the rotation, parallel comp remains the same
    
    return S_rot / np.linalg.norm(S_rot)


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


        