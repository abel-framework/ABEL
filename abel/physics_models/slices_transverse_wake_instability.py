"""
Transverse wake instability model as described in thesis "Instability and Beam-Beam Study for Multi-TeV PWFA e+e- and gamma gamma Linear Colliders" (https://cds.cern.ch/record/2754022?ln=en).

Parameters
----------
plasma_density: [m^-3] float
    Plasma density.
    
E_z: [V/m] 1D float array
    Contains the gradient/longitudinal electric field in the region of interest.
    
bubble_radius: [m] 1D float array
    Contains the bubble radius in the region of interest.
    
num_profile: [1/m] 1D float array
    Contains the longitudinal number profile of the bunch.
    
initial_offsets: [m] 1D float array
    Contains the transverse (x/y) center position of each slice.

initial_angles: [rad] 1D float array
    Contains the angles (x'/y') of each slice.

initial_energies: [eV] 1D float array
    Contains the energies of each slice.
    
stage_length: [m] float
    Length of the plasma stage.

s_slices: [m] 1D float array
    Contains the propagation coordinate of each slice.

xi_slices: [m] 1D float array
    Contains the co-moving coordinate of each slice.
    

Returns
----------
    s_slices_table, x_slices_table, angle_slices_table, energy_slices


Ben Chen, 5 July 2023, University of Oslo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, epsilon_0 as eps0
from abel.utilities.other import find_closest_value_in_arr
from abel.utilities.relativity import energy2gamma
from tqdm import tqdm
import time

from abel.utilities.plasma_physics import k_p


def wakefunc_Stupakov(xi_lead, xi_ref, a):
    return 2/(np.pi*eps0*a**4)*np.abs(xi_lead - xi_ref)  # [V/Cm^2]
    

def transverse_wake_instability_slices(plasma_density, E_z, bubble_radius, num_profile, initial_offsets, initial_angles, initial_energies, stage_length, s_slices, xi_slices):

    x_slices = initial_offsets  # [m] the center x position of each slice.
    angle_slices = initial_angles  # [rad] x' for each beam slice.
    energy_slices = initial_energies  # [eV] single particle energy for each slice.
    initial_gammas = energy2gamma(initial_energies)  # Initial Lorentz factor for each slice.
    gamma0 = np.mean(initial_gammas)

    # Check whether particles have too large angles
    if np.max(np.abs(angle_slices)) > 50e-3:
        raise ValueError('Some angles are too large for small angle approximation.')

    # Check whether particles have too small energies
    v0 = c*np.sqrt(1-1/initial_gammas**2)  # [m/s] initial speed for each slice.
    if np.min(v0)/c < 0.99:
        raise ValueError('Velocity too low for speed of light approximation.')
    
    ve_xi = v0  # [m/s] longitudinal electron velocity of each slice.
    p_xi = initial_gammas*m_e*ve_xi  # [kg m/s] initial longitudinal momentum at each beam slice.
    p_x = angle_slices*p_xi  # [kg m/s] initial transverse momentum at each beam slice.
    ve_x = angle_slices*ve_xi  # [m/s] transverse velocity of each beam slice.
    F_x = np.zeros(len(x_slices))  # [N] transverse force on each beam slice.
    
    skin_depth = 1/k_p(plasma_density)  # [m] 1/kp, plasma skin depth.
    
    #beta_func = c/e*np.sqrt(2* gamma0 *eps0*m_e/plasma_density)  # [m] matched beta function.
    #beta_wave_length = 2*np.pi*beta_func  # [m] betatron wavelength.
    #time_step = 0.01*beta_wave_length/np.max(ve_xi)/2  # [s] beam time step.
    
    # Record s_slices, x_slices and angle_slices at each time step in tables
    s_slices_table = s_slices  # [m]
    x_slices_table = x_slices  # [m]
    angle_slices_table = angle_slices  # [rad]
    
    
    ############# Beam propagation through the plasma cell #############
    time_step_count = 0.0
    s_start = s_slices[-1]  # Set the current propagation length using the beam head.
    prop_length = 0.0

    # Progress bar
    #pbar = tqdm(total=100)
    #pbar.set_description('0%')

    while prop_length < stage_length:
        
        beta_func = c/e*np.sqrt(2* energy2gamma(np.mean(energy_slices)) *eps0*m_e/plasma_density)  # [m] matched beta function.
        beta_wave_length = 2*np.pi*beta_func  # [m] betatron wavelength.
        time_step = 0.05*beta_wave_length/np.max(ve_xi)/2  # [s] beam time step.
        
        # ============= Drift of beam =============
        x_slices = x_slices + p_x/p_xi*1/2*ve_xi*time_step
        s_slices = s_slices + 1/2*ve_xi*time_step
        time_step_count = time_step_count + 1/2

        
        # ============= Integrate the wake function and beam slices =============
        for idx_beam in range(len(x_slices)-1,-1,-1):  # Loops through all beam slices
            
            #E_x = 0.0
            #
            #for idx_convl in range(len(x_slices)-1,idx_beam,-1):  # Goes through all the preceding slices that generates a transverse wakefield that acts on slice idx_beam.
            #        
            #    a = bubble_radius[idx_convl] + 0.75*skin_depth
            #    incr = -e * wakefunc_Stupakov(xi_slices[idx_convl], xi_slices[idx_beam], a) * num_profile[idx_convl] * x_slices[idx_convl]
            #    E_x = E_x + incr  # Transverse electric field on beam slice at xi_slices[idx_beam] caused by the wake.
                
            a = bubble_radius[-1:idx_beam:-1] + 0.75*skin_depth
            xi_preceding_slices = xi_slices[-1:idx_beam:-1]
            xi_ref_slice = xi_slices[idx_beam]
            num_profile_preceding_slices = num_profile[-1:idx_beam:-1]
            x_preceding_slices = x_slices[-1:idx_beam:-1]
            contributions = -e * wakefunc_Stupakov(xi_preceding_slices, xi_ref_slice, a) * num_profile_preceding_slices * x_preceding_slices
            E_x = np.sum(contributions, axis=0)  # Sum the contributions from all preceding slices.
            
            F_x[idx_beam] = -e*(E_x + plasma_density*e*x_slices[idx_beam]/(2*eps0))  # Total transverse force on beam slice at xi_slices[idx_beam].

        # Update momentum and velocity
        p_x = p_x + F_x*time_step
        
        #E_z = -6.4e9  # Overload with constant field to see how this affects instability.
        p_xi = p_xi - e*E_z*time_step  # Electron longitudinal momentum at each slice. The signs of the charge and the gradient cancel each other.
        
        if np.max(np.abs(ve_x))>c or np.max(np.abs(ve_xi))>c:
            raise ValueError(f"\nError!\ntime_step_count = {time_step_count}\nidx_beam = {idx_beam}\nmax(abs(ve_x)) = {np.max(np.abs(ve_x))} m/s\nmax(abs(ve_xi)) = {np.max(np.abs(ve_xi))} m/s\n")
        
        # ============= Drift of beam =============
        angle_slices = p_x/p_xi
        
        # Check whether particles have too large angles
        if np.max(np.abs(angle_slices)) > 50e-3:
            raise ValueError('Some angles are too large for small angle approximation.')
        
        ve_x = angle_slices*ve_xi  # [m/s] transverse velocity of each beam slice.
        
        x_slices = x_slices + angle_slices*1/2*ve_xi*time_step
        s_slices = s_slices + 1/2*ve_xi*time_step
        time_step_count = time_step_count + 1/2

        # Check whether beam comes into contact with plasma bubble boundary
        if np.any(np.abs(x_slices) - bubble_radius >= 0):
            print('$s=$' f'{format(prop_length, ".2f")}' ' m')
            plt.figure()
            plt.plot(s_slices, x_slices*1e6, 'r', label='Transverse offset') 
            plt.plot(s_slices, bubble_radius*1e6, 'b', label='Bubble radius')
            plt.plot(s_slices, -bubble_radius*1e6, 'b')
            plt.xlabel(r'$s$ [$\mathrm{m}$]')
            plt.ylabel('Transverse axis $\mathrm{\mu}$m]')
            plt.legend()
            plt.title('Diagnostic plot for the moment beam coming into contact with bubble boundary')
            raise Exception('Beam came into contact with bubble boundary!')

        prop_length =  s_slices[-1] - s_start  # Set the current propagation length using the beam head.

        
        ############# Record some data #############
        energy_slices = np.sqrt((p_x**2+p_xi**2)*c**2+(m_e*c**2)**2)/e  # [eV] single particle energy for each slice.)
        # Record s_slices, x_slices and angle_slices at each time step in tables
        s_slices_table = np.vstack((s_slices_table, s_slices))  # [m]
        x_slices_table = np.vstack((x_slices_table, x_slices))  # [m]
        angle_slices_table = np.vstack((angle_slices_table, angle_slices))  # [rad]

        
        # Progress bar
        #pbar.update(prop_length/stage_length*100 - pbar.n)        
        #pbar.set_description(f"Instability tracking {round(prop_length/stage_length*100,2)}%")
        
        
    # =============  =============

    # Progress bar
    #pbar.close()
    
    return s_slices_table, x_slices_table, angle_slices_table, energy_slices