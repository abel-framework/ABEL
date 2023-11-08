"""
Transverse wake instability model as described in thesis "Instability and Beam-Beam Study for Multi-TeV PWFA e+e- and gamma gamma Linear Colliders" (https://cds.cern.ch/record/2754022?ln=en).

Parameters
----------
beam: Beam object

plasma_density: [m^-3] float
    Plasma density.
    
Ez_fit: [V/m] interpolation object?
    Contains the gradient/longitudinal electric field ...
    
rb_fit: [m] interpolation object?
    Contains the bubble radius ...
    
#num_profile: [1/m] 1D float array
    Contains the longitudinal number profile of the bunch.
    
#initial_offsets: [m] 1D float array
    Contains the transverse (x/y) center position of each slice.

#initial_angles: [rad] 1D float array
    Contains the angles (x'/y') of each slice.

#initial_energies: [eV] 1D float array
    Contains the energies of each slice.
    
stage_length: [m] float
    Length of the plasma stage.

time_ste_mod: float
    Time step modifier in units of beta_wave_length/c.

get_centroids: bool
    TODO

s_slices: [m] 1D float array
    Contains the propagation coordinate of each slice.

z_slices: [m] 1D float array
    Contains the co-moving coordinate of each slice.
    

Returns
----------
    beam_out, s_slices_table, x_slices_table, xp_slices_table, y_slices_table, yp_slices_table


Ben Chen, 5 October 2023, University of Oslo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, epsilon_0 as eps0
#from opal.classes.stage.impl.stage_particle_transverse_wake_instability import particles2slices
from opal.classes.beam import *
#from opal.utilities.other import find_closest_value_in_arr
from opal.utilities.relativity import energy2gamma
from tqdm import tqdm
import time
from joblib import Parallel, delayed  # Parallel tracking
from joblib_progress import joblib_progress

from opal.utilities.plasma_physics import k_p


#def wakefunc_Stupakov(xi_lead, xi_ref, a):
#    return 2/(np.pi*eps0*a**4)*np.abs(xi_lead - xi_ref)  # [V/Cm^2]
#
#
## ==================================================
#def integrate_wake_func(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets, tr_momenta):
#    tr_force = np.zeros(len(zs_sorted))  # [N] transverse force on each particle.
#    for idx_particle in range(len(zs_sorted)-1,-1,-1):  # Loops through all macro particles
#
#        a = bubble_radius[-1:idx_particle:-1] + 0.75*skin_depth
#        z_preceding_particles = zs_sorted[-1:idx_particle:-1]
#        z_ref_particle = zs_sorted[idx_particle]
#        weights_preceding_particles = weights_sorted[-1:idx_particle:-1]
#        
#        offsets_preceding_particles = offsets[-1:idx_particle:-1]
#        contributions = -e * wakefunc_Stupakov(z_preceding_particles, z_ref_particle, a) * weights_preceding_particles * offsets_preceding_particles
#        E_field = np.sum(contributions, axis=0)  # Sum the contributions from all preceding slices.
#        tr_force[idx_particle] = -e*(E_field + plasma_density*e*offsets[idx_particle]/(2*eps0))  # Total transverse force on beam particle at zs_sorted[idx_particle].
#        
#    # Update momenta
#    tr_momenta = tr_momenta + tr_force*time_step
#    return tr_momenta


# ==================================================
# Simplified loop
def integrate_wake_func(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets, tr_momenta):
    
    a = bubble_radius + 0.75*skin_depth
    
    # Assemble an array f used for the dot product (Based on Stupakov's wake function)
    f = -e*2/(np.pi*eps0*a**4)*weights_sorted*offsets
    
    # Calculate the wakefield on each macro particle
    wakefield = np.zeros(len(zs_sorted))
    for idx_particle in range(len(zs_sorted)):
        wakefield[idx_particle] = np.sum((zs_sorted[idx_particle+1:] - zs_sorted[idx_particle]) * f[idx_particle+1:])
    
    # Calculate the total transverse force on macro particles
    tr_force = -e*(wakefield + plasma_density*e*offsets/(2*eps0))
    
    # Update momenta
    tr_momenta = tr_momenta + tr_force*time_step
    return tr_momenta
    

# ==================================================
# Using matrix multiplication
#def integrate_wake_func(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets, tr_momenta):
#    
#    # Create a two-dimensional array where each element (i, j) represents (z[j] - z[i])
#    z_diff_mat = (zs_sorted - zs_sorted[:, None])
#    
#    # Create an upper triangular matrix to ensure that only the elements where j > i are included
#    z_diff_mat_filtered = np.triu(z_diff_mat)  # Does not require a sorted z-array.
#
#    # Assemble an array f used for the dot product (Based on Stupakov's wake function)
#    a = bubble_radius + 0.75*skin_depth
#    f = -e*2/(np.pi*eps0*a**4)*weights_sorted*offsets
#
#    # Calculate the wakefield on each macro particle
#    wakefield = np.dot(z_diff_mat_filtered, f)
#    ##wakefield = z_diff_mat_filtered @ f
#    # Since z_diff_mat_filtered[i][j]=z[j]-z[i] then wakefield=np.dot(z_diff_mat_filtered, f) => wakefield[i]=sum_j((z[j]-z[i])*f[j]).
#
#    # Calculate the total transverse force on macro particles
#    tr_force = -e*(wakefield + plasma_density*e*offsets/(2*eps0))
#
#    # Update momenta
#    tr_momenta = tr_momenta + tr_force*time_step
#    return tr_momenta


# ==================================================
#def doffset_dt(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets, tr_momenta, Ez, pzs_sorted):
#    tr_force = np.zeros(len(zs_sorted))  # [N] transverse force on each particle.
#    for idx_particle in range(len(zs_sorted)-1,-1,-1):  # Loops through all macro particles
#
#        a = bubble_radius[-1:idx_particle:-1] + 0.75*skin_depth
#        z_preceding_particles = zs_sorted[-1:idx_particle:-1]
#        z_ref_particle = zs_sorted[idx_particle]
#        weights_preceding_particles = weights_sorted[-1:idx_particle:-1]
#        
#        offsets_preceding_particles = offsets[-1:idx_particle:-1]
#        contributions = -e * wakefunc_Stupakov(z_preceding_particles, z_ref_particle, a) * weights_preceding_particles * offsets_preceding_particles
#        E_field = np.sum(contributions, axis=0)  # Sum the contributions from all preceding slices.
#        tr_force[idx_particle] = -e*(E_field + plasma_density*e*offsets[idx_particle]/(2*eps0))  # Total transverse force on beam particle at zs_sorted[idx_particle].
#        
#    # Update momenta
#    tr_momenta = tr_momenta + tr_force*time_step
#    pzs_sorted = pzs_sorted - e*Ez*time_step
#    
#    return tr_momenta/pzs_sorted*c
#
#
# ==================================================
#def RK4(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets, tr_momenta, Ez, pzs_sorted):
#    #ys_sorted = ys_sorted + pys_sorted/pzs_sorted*1/2*c*time_step
#    #
#    #pxs_sorted = pxs_sorted + Fx*time_step
#    #tr_momenta = tr_momenta + Fy*time_step   
#    #pzs_sorted = pzs_sorted - e*Ez*time_step
#    #
#    #dydt = tr_momenta/pzs_sorted*c
##
#    #pzs_sorted = pzs_sorted - e*Ez*time_step
#    #dydt = integrate_wake_func(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets=ys_sorted, tr_momenta=tr_momenta)/pzs_sorted*c
#    
#    k1 = time_step*tr_momenta/pzs_sorted*c
#    k2 = time_step * doffset_dt(skin_depth, plasma_density, time_step/2, zs_sorted, bubble_radius, weights_sorted, offsets+k1/2, tr_momenta, Ez, pzs_sorted)
#    k3 = time_step * doffset_dt(skin_depth, plasma_density, time_step/2, zs_sorted, bubble_radius, weights_sorted, offsets+k2/2, tr_momenta, Ez, pzs_sorted)
#    k4 = time_step * doffset_dt(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets+k3, tr_momenta, Ez, pzs_sorted)
#    
#    offsets = offsets + k1/6 + k2/3 + k3/3 + k4/6
#    return offsets


# ==================================================
def transverse_wake_instability_particles(beam, plasma_density, Ez_fit, rb_fit, stage_length, time_step_mod=0.05, get_centroids=False, s_slices=None, z_slices=None):
    
    energies = beam.Es()
    xs = beam.xs()
    ys = beam.ys()
    zs = beam.zs()
    pxs = beam.pxs()
    pys = beam.pys()
    pzs = beam.pzs()
    weights = beam.weightings()

    # Sort the arrays based on zs
    indices = np.argsort(zs)
    zs_sorted = zs[indices]
    xs_sorted = xs[indices]
    ys_sorted = ys[indices]
    pxs_sorted = pxs[indices]
    pys_sorted = pys[indices]
    pzs_sorted = pzs[indices]
    energs_sorted = energies[indices]
    weights_sorted = weights[indices]

    # Filter out particles that have too small energies
    bool_indices = (energs_sorted > 7.0*m_e*c**2/e)  # Corresponds to 0.99c.
    zs_sorted = zs_sorted[bool_indices]
    xs_sorted = xs_sorted[bool_indices]
    ys_sorted = ys_sorted[bool_indices]
    pxs_sorted = pxs_sorted[bool_indices]
    pys_sorted = pys_sorted[bool_indices]
    pzs_sorted = pzs_sorted[bool_indices]
    #energs_sorted = energs_sorted[bool_indices]
    weights_sorted = weights_sorted[bool_indices]

    # Calculate Ez and rb based on interpolations of Ez and rb vs z
    Ez = Ez_fit(zs_sorted)
    bubble_radius = rb_fit(zs_sorted)
    
    skin_depth = 1/k_p(plasma_density)  # [m] 1/kp, plasma skin depth.

    if get_centroids is True:  # TODO
        if s_slices is None or z_slices is None:
            raise ValueError('s_slices or z_slices are not defined.')
        
        # Record s_slices, x_slices and xp_slices at each time step in tables
        s_slices = s_slices + prop_length
        x_slices = particles2slices(beam=beam, beam_quant=beam.xs(), z_slices=z_slices, make_plot=False)
        xp_slices = particles2slices(beam=beam, beam_quant=beam.xps(), z_slices=z_slices, make_plot=False)
        y_slices = particles2slices(beam=beam, beam_quant=beam.ys(), z_slices=z_slices, make_plot=False)
        yp_slices = particles2slices(beam=beam, beam_quant=beam.yps(), z_slices=z_slices, make_plot=False)
        
        #s_start = beam.location  # Set the current propagation distance.
        #s_slices = z_slices + beam0.location
        s_slices_table = s_slices  # [m]
        x_slices_table = x_slices  # [m]
        xp_slices_table = xp_slices  # [rad]
        y_slices_table = y_slices  # [m]
        yp_slices_table = yp_slices  # [rad]
    else:
        s_slices_table = None
        x_slices_table = None
        xp_slices_table = None
        y_slices_table = None
        yp_slices_table = None
    
    
    ############# Beam propagation through the plasma cell #############
    time_step_count = 0.0
    prop_length = 0.0

    # Progress bar
    pbar = tqdm(total=100)
    pbar.set_description('0%')

    while prop_length < stage_length:

        # ============= Apply filters =============
        # Filter out particles that diverge too much for small angle approximation
        bool_indices = (np.abs(pxs_sorted/pzs_sorted) < 50e-3)
        zs_sorted = zs_sorted[bool_indices]
        xs_sorted = xs_sorted[bool_indices]
        ys_sorted = ys_sorted[bool_indices]
        pxs_sorted = pxs_sorted[bool_indices]
        pys_sorted = pys_sorted[bool_indices]
        pzs_sorted = pzs_sorted[bool_indices]
        #energs_sorted = energs_sorted[bool_indices]
        weights_sorted = weights_sorted[bool_indices]
        Ez = Ez[bool_indices]
        bubble_radius = bubble_radius[bool_indices]

        bool_indices = (np.abs(pys_sorted/pzs_sorted) < 50e-3)
        zs_sorted = zs_sorted[bool_indices]
        xs_sorted = xs_sorted[bool_indices]
        ys_sorted = ys_sorted[bool_indices]
        pxs_sorted = pxs_sorted[bool_indices]
        pys_sorted = pys_sorted[bool_indices]
        pzs_sorted = pzs_sorted[bool_indices]
        #energs_sorted = energs_sorted[bool_indices]
        weights_sorted = weights_sorted[bool_indices]
        Ez = Ez[bool_indices]
        bubble_radius = bubble_radius[bool_indices]

        Fx = np.zeros(len(zs_sorted))  # [N] transverse force on each particle.
        Fy = np.zeros(len(zs_sorted))  # [N] transverse force on each particle.
        
        
        # ============= Drift of beam =============
        gammas = energy2gamma(energies)  # Initial Lorentz factor for each particle.
        beta_func = c/e*np.sqrt(2* np.mean(gammas) *eps0*m_e/plasma_density)  # [m] matched beta function.
        beta_wave_length = 2*np.pi*beta_func  # [m] betatron wavelength.
        time_step = time_step_mod*beta_wave_length/c  # [s] beam time step.

        # Leapfrog
        time_step_count = time_step_count + 1/2
        prop_length = prop_length + 1/2*c*time_step
        xs_sorted = xs_sorted + pxs_sorted/pzs_sorted*1/2*c*time_step
        ys_sorted = ys_sorted + pys_sorted/pzs_sorted*1/2*c*time_step

        
        # ============= Integrate the wake function =============
        #for idx_particle in range(len(zs_sorted)-1,-1,-1):  # Loops through all macro particles
#
        #    a = bubble_radius[-1:idx_particle:-1] + 0.75*skin_depth
        #    z_preceding_particles = zs_sorted[-1:idx_particle:-1]
        #    z_ref_particle = zs_sorted[idx_particle]
        #    weights_preceding_particles = weights_sorted[-1:idx_particle:-1]
        #    
        #    x_preceding_particles = xs_sorted[-1:idx_particle:-1]
        #    contributions = -e * wakefunc_Stupakov(z_preceding_particles, z_ref_particle, a) * weights_preceding_particles * x_preceding_particles
        #    Ex = np.sum(contributions, axis=0)  # Sum the contributions from all preceding slices.
        #    Fx[idx_particle] = -e*(Ex + plasma_density*e*xs_sorted[idx_particle]/(2*eps0))  # Total transverse force on beam particle at zs_sorted[idx_particle].
#
        #    y_preceding_particles = ys_sorted[-1:idx_particle:-1]
        #    contributions = -e * wakefunc_Stupakov(z_preceding_particles, z_ref_particle, a) * weights_preceding_particles * y_preceding_particles
        #    Ey = np.sum(contributions, axis=0)  # Sum the contributions from all preceding slices.
        #    Fy[idx_particle] = -e*(Ey + plasma_density*e*ys_sorted[idx_particle]/(2*eps0))  # Total transverse force on beam particle at zs_sorted[idx_particle].
#
        ## Update momenta
        #pxs_sorted = pxs_sorted + Fx*time_step
        #pys_sorted = pys_sorted + Fy*time_step


        # Parallel tracking
        results = Parallel(n_jobs=2)([
            delayed(integrate_wake_func)(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets=xs_sorted, tr_momenta=pxs_sorted),
            delayed(integrate_wake_func)(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets=ys_sorted, tr_momenta=pys_sorted)
        ])
        # Update momenta
        pxs_sorted = results[0]
        pys_sorted = results[1]

        #Ez = -6.4e9*np.ones(len(pzs_sorted))  # Overload with constant field to see how this affects instability. #############################################################################
        pzs_sorted = pzs_sorted - e*Ez*time_step  # Update longitudinal momenta.
        
        
        # ============= Drift of beam =============
        # Leapfrog
        xs_sorted = xs_sorted + pxs_sorted/pzs_sorted*1/2*c*time_step
        ys_sorted = ys_sorted + pys_sorted/pzs_sorted*1/2*c*time_step
        time_step_count = time_step_count + 1/2
        prop_length = prop_length + 1/2*c*time_step

        # RK4
        #xs_sorted = RK4(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, xs_sorted, pxs_sorted, Ez, pzs_sorted)
        #ys_sorted = RK4(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, ys_sorted, pys_sorted, Ez, pzs_sorted)
        #time_step_count = time_step_count + 1
        #prop_length = prop_length + c*time_step

        
        # ============= Filter out particles that collide into bubble =============
        bool_indices = (np.abs(xs_sorted) - bubble_radius < 0)
        zs_sorted = zs_sorted[bool_indices]
        xs_sorted = xs_sorted[bool_indices]
        ys_sorted = ys_sorted[bool_indices]
        pxs_sorted = pxs_sorted[bool_indices]
        pys_sorted = pys_sorted[bool_indices]
        pzs_sorted = pzs_sorted[bool_indices]
        weights_sorted = weights_sorted[bool_indices]
        Ez = Ez[bool_indices]
        bubble_radius = bubble_radius[bool_indices]

        bool_indices = (np.abs(ys_sorted) - bubble_radius < 0)
        zs_sorted = zs_sorted[bool_indices]
        xs_sorted = xs_sorted[bool_indices]
        ys_sorted = ys_sorted[bool_indices]
        pxs_sorted = pxs_sorted[bool_indices]
        pys_sorted = pys_sorted[bool_indices]
        pzs_sorted = pzs_sorted[bool_indices]
        weights_sorted = weights_sorted[bool_indices]
        Ez = Ez[bool_indices]
        bubble_radius = bubble_radius[bool_indices] 

        #beam.set_xs(xs_sorted)
        #beam.set_ys(ys_sorted)
        #beam.set_zs(zs_sorted)
        #uxs = momentum2proper_velocity(pxs_sorted)
        #beam.set_uxs(uxs)
        #uys = momentum2proper_velocity(pys_sorted)
        #beam.set_uys(uys)
        #uzs = momentum2proper_velocity(pzs_sorted)
        #beam.set_uzs(uzs)
        #beam.set_qs(-e*weights_sorted)

        # Initialise OPAL Beam object
        beam_out = Beam()
        
        # set the phase space of the OPAL beam
        beam_out.set_phase_space(Q=np.sum(weights_sorted)*-e,
                             xs=xs_sorted,
                             ys=ys_sorted,
                             zs=zs_sorted, 
                             pxs=pxs_sorted,
                             pys=pys_sorted,
                             pzs=pzs_sorted)
        

        #============= Add some diagnostics =============
        if get_centroids is True:  # TODO
            # Record s_slices, x_slices and xp_slices at each time step in tables
            s_slices = s_slices + prop_length
            x_slices = particles2slices(beam=beam, beam_quant=beam.xs(), z_slices=z_slices, make_plot=False)
            xp_slices = particles2slices(beam=beam, beam_quant=beam.xps(), z_slices=z_slices, make_plot=False)
            y_slices = particles2slices(beam=beam, beam_quant=beam.ys(), z_slices=z_slices, make_plot=False)
            yp_slices = particles2slices(beam=beam, beam_quant=beam.yps(), z_slices=z_slices, make_plot=False)
            #energy_slices = particles2slices(beam=beam, beam_quant=beam.Es(), z_slices=z_slices, make_plot=False)
            
            s_slices_table = np.vstack((s_slices_table, s_slices))  # [m]
            x_slices_table = np.vstack((x_slices_table, x_slices))  # [m]
            xp_slices_table = np.vstack((xp_slices_table, xp_slices))  # [rad]
            y_slices_table = np.vstack((y_slices_table, y_slices))  # [m]
            yp_slices_table = np.vstack((yp_slices_table, yp_slices))  # [rad]
        

        # Progress bar
        pbar.update(prop_length/stage_length*100 - pbar.n)        
        pbar.set_description(f"Instability tracking {round(prop_length/stage_length*100,2)}%")
        
        
    #=============  =============
    
    pbar.close()
    
    return beam_out, s_slices_table, x_slices_table, xp_slices_table, y_slices_table, yp_slices_table