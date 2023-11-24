"""
2D particle version of transverse wake instability model as described in thesis "Instability and Beam-Beam Study for Multi-TeV PWFA e+e- and gamma gamma Linear Colliders" (https://cds.cern.ch/record/2754022?ln=en).

Currently do not support particles being deleted.

Parameters
----------
beam: Beam object

plasma_density: [m^-3] float
    Plasma density.
    
Ez_fit: [V/m] interpolation object?
    Contains the gradient/longitudinal electric field ...
    
rb_fit: [m] interpolation object?
    Contains the bubble radius ...
    
stage_length: [m] float
    Length of the plasma stage.

time_ste_mod: float
    Time step modifier in units of beta_wave_length/c.

get_centroids: bool

s_slices: [m] 1D float array
    Contains the propagation coordinate of each slice.

z_slices: [m] 1D float array
    Contains the co-moving coordinate of each slice.
    

Returns
----------
    offsets, tr_momenta, z_sorted, pzs_sorted, s_slices_table, offset_slices_table, angle_slices_table, y_slices_table, yp_slices_table


Ben Chen, 7 October 2023, University of Oslo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, epsilon_0 as eps0
from abel.classes.beam import *
#from abel.utilities.other import find_closest_value_in_arr
from abel.utilities.relativity import energy2gamma
from abel.utilities.plasma_physics import k_p
from tqdm import tqdm



def wakefunc_Stupakov(xi_lead, xi_ref, a):
    return 2/(np.pi*eps0*a**4)*np.abs(xi_lead - xi_ref)  # [V/Cm^2]


def twoD_transverse_wake_instability_particles(beam, offsets, tr_momenta, plasma_density, Ez_fit, rb_fit, stage_length, time_step_mod=0.05, get_centroids=False, s_slices=None, z_slices=None):

    # Check whether particles have too small energies
    energies = beam.Es()
    if np.min(energies) < 7.0*m_e*c**2/e:  # Corresponds to 0.99c.
        raise ValueError('Velocity too low for speed of light approximation.')

    # Check whether particles have too large angles
    pzs = beam.pzs()
    if np.max(np.abs(tr_momenta/pzs)) > 50e-3:
        raise ValueError('Some angles are too large for small angle approximation.')
        
    zs = beam.zs()
    weights = beam.weightings()

    # Sort the arrays based on zs
    indices = np.argsort(zs)
    zs_sorted = zs[indices]
    offsets = offsets[indices]
    tr_momenta = tr_momenta[indices]
    pzs_sorted = pzs[indices]
    weights_sorted = weights[indices]

    # Calculate Ez and rb based on interpolations of Ez and rb vs z
    Ez = Ez_fit(zs_sorted)
    bubble_radius = rb_fit(zs_sorted)
    
    skin_depth = 1/k_p(plasma_density)  # [m] 1/kp, plasma skin depth.

    if get_centroids is True:  # TODO
        if s_slices is None or z_slices is None:
            raise ValueError('s_slices or z_slices are not defined.')
        
        # Record s_slices, offset_slices and angle_slices at each time step in tables
        s_slices = s_slices + prop_length
        offset_slices = particles2slices(beam=beam, beam_quant=beam.xs(), z_slices=z_slices, make_plot=False)
        angle_slices = particles2slices(beam=beam, beam_quant=beam.xps(), z_slices=z_slices, make_plot=False)
        s_slices_table = s_slices  # [m]
        offset_slices_table = offset_slices  # [m]
        angle_slices_table = angle_slices  # [rad]
    else:
        s_slices_table = None
        offset_slices_table = None
        angle_slices_table = None
    
    
    ############# Beam propagation through the plasma cell #############
    time_step_count = 0.0
    prop_length = 0.0

    # Progress bar
    #pbar = tqdm(total=100)
    #pbar.set_description('0%')

    while prop_length < stage_length:

        tr_force = np.zeros(len(zs_sorted))  # [N] transverse force on each particle.
        
        # ============= Drift of beam =============
        gammas = energy2gamma(energies)  # Initial Lorentz factor for each particle.
        beta_func = c/e*np.sqrt(2* np.mean(gammas) *eps0*m_e/plasma_density)  # [m] matched beta function.
        beta_wave_length = 2*np.pi*beta_func  # [m] betatron wavelength.
        time_step = time_step_mod*beta_wave_length/c  # [s] beam time step.
        
        time_step_count = time_step_count + 1/2
        prop_length = prop_length + 1/2*c*time_step
        offsets = offsets + tr_momenta/pzs_sorted*1/2*c*time_step

        
        # ============= Integrate the wake function =============
        # Loop option 1
        #for idx_particle in range(len(zs_sorted)-1,-1,-1):  # Loops through all macroparticles
 #
        #    a = bubble_radius[-1:idx_particle:-1] + 0.75*skin_depth
        #    z_preceding_particles = zs_sorted[-1:idx_particle:-1]
        #    z_ref_particle = zs_sorted[idx_particle]
        #    weights_preceding_particles = weights_sorted[-1:idx_particle:-1]
        #    
        #    offset_preceding_particles = offsets[-1:idx_particle:-1]
        #    contributions = -e * wakefunc_Stupakov(z_preceding_particles, z_ref_particle, a) * weights_preceding_particles * offset_preceding_particles
        #    wakefield = np.sum(contributions, axis=0)  # Sum the contributions from all preceding slices.
        #    tr_force[idx_particle] = -e*(wakefield + plasma_density*e*offsets[idx_particle]/(2*eps0))  # Total transverse force on beam particle at zs_sorted[idx_particle].
 #
        ## Update momenta
        #tr_momenta = tr_momenta + tr_force*time_step


        # Loop option 2
        a = bubble_radius + 0.75*skin_depth
        # Assemble an array f used for the dot product (Based on Stupakov's wake function)
        f = -e*2/(np.pi*eps0*a**4)*weights_sorted*offsets
        # Calculate the wakefield on each macroparticle
        wakefield = np.zeros(len(zs_sorted))
        for idx_particle in range(len(zs_sorted)):
            wakefield[idx_particle] = np.sum((zs_sorted[idx_particle+1:] - zs_sorted[idx_particle]) * f[idx_particle+1:])
        # Calculate the total transverse force on macroparticles
        tr_force = -e*(wakefield + plasma_density*e*offsets/(2*eps0))
        # Update momenta
        tr_momenta = tr_momenta + tr_force*time_step


        # Matrix method
        # Create a two-dimensional array where each element (i, j) represents (z[j] - z[i])
        #z_diff_mat = (zs_sorted - zs_sorted[:, None])
        #
        ## Create an upper triangular matrix to ensure that only the elements where j > i are included
        #z_diff_mat_filtered = np.triu(z_diff_mat)  # Does not require a sorted z-array.
    #
        ## Assemble an array f used for the dot product (Based on Stupakov's wake function)
        #a = bubble_radius + 0.75*skin_depth
        #f = -e*2/(np.pi*eps0*a**4)*weights_sorted*offsets
    #
        ## Calculate the wakefield on each macroparticle
        #wakefield = np.dot(z_diff_mat_filtered, f)
        ##wakefield = z_diff_mat_filtered @ f
        ## Since z_diff_mat_filtered[i][j]=z[j]-z[i] then wakefield=np.dot(z_diff_mat_filtered, f) => wakefield[i]=sum_j((z[j]-z[i])*f[j]).
    #
        ## Calculate the total transverse force on macroparticles
        #tr_force = -e*(wakefield + plasma_density*e*offsets/(2*eps0))
    #
        ## Update momenta
        #tr_momenta = tr_momenta + tr_force*time_step


        pzs_sorted = pzs_sorted - e*Ez*time_step  # Update longitudinal momenta.
        
        
        # ============= Drift of beam =============
        # Check whether particles have too large angles
        if np.max(np.abs(tr_momenta/pzs)) > 50e-3:
            raise ValueError('Some angles are too large for small angle approximation.')
        
        time_step_count = time_step_count + 1/2
        prop_length = prop_length + 1/2*c*time_step
        offsets = offsets + tr_momenta/pzs_sorted*1/2*c*time_step

        # Check whether beam comes into contact with plasma bubble boundary
        if np.any(np.abs(offsets) - bubble_radius >= 0):
            print('$s=$' f'{format(prop_length, ".2f")}' ' m')
            plt.figure()
            plt.scatter(zs_sorted, offsets*1e6, 'r', label='Transverse offset') 
            plt.plot(zs_sorted, bubble_radius*1e6, 'b', label='Bubble radius')
            plt.plot(zs_sorted, -bubble_radius*1e6, 'b')
            plt.xlabel(r'$s$ [$\mathrm{m}$]')
            plt.ylabel('Transverse axis $\mathrm{\mu}$m]')
            plt.legend()
            plt.title('Diagnostic plot for the moment beam coming into contact with bubble boundary')
            raise Exception('Beam came into contact with bubble boundary!')
        

        #============= Add some diagnostics =============
        if get_centroids is True:  # TODO
            # Record s_slices, offset_slices and angle_slices at each time step in tables
            s_slices = s_slices + prop_length
            offset_slices = particles2slices(beam=beam, beam_quant=beam.xs(), z_slices=z_slices, make_plot=False)
            angle_slices = particles2slices(beam=beam, beam_quant=beam.xps(), z_slices=z_slices, make_plot=False)
            
            s_slices_table = np.vstack((s_slices_table, s_slices))  # [m]
            offset_slices_table = np.vstack((offset_slices_table, offset_slices))  # [m]
            angle_slices_table = np.vstack((angle_slices_table, angle_slices))  # [rad]
        

        # Progress bar
        #pbar.update(prop_length/stage_length*100 - pbar.n)        
        #pbar.set_description(f"Instability tracking {round(prop_length/stage_length*100,2)}%")
        
        
    #=============  =============
    
    #pbar.close()
    
    return offsets, tr_momenta, zs_sorted, pzs_sorted, weights_sorted, s_slices_table, offset_slices_table, angle_slices_table