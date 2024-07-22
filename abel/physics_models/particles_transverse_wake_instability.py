"""
Transverse wake instability model as described in thesis "Instability and Beam-Beam Study for Multi-TeV PWFA e+e- and gamma gamma Linear Colliders" (https://cds.cern.ch/record/2754022?ln=en).

Parameters
----------
beam: Beam object

plasma_density: [m^-3] float
    Plasma density.
    
Ez_fit_obj: [V/m] interpolation object
    1D interpolation object of longitudinal E-field fitted to axial E-field using a selection of zs along the main beam. Used to determine the value of the longitudinal E-field for all beam zs.
    
rb_fit_obj: [m] interpolation object
    1D interpolation object of plasma bubble radius fitted to axial bubble radius using a selection of zs along the main beam. Used to determine the value of the bubble radius for all beam zs.
    
stage_length: [m] float
    Length of the plasma stage.

time_step_mod: float
    Determines the time step of the instability tracking in units of beta_wave_length/c.

enable_radiation_reaction: bool
    Flag for enabling radiation reactions.

show_prog_bar: bool
    Flag for displaying the progress bar.

...

    
Returns
----------
beam_out


Ben Chen, 5 October 2023, University of Oslo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, epsilon_0 as eps0

from tqdm import tqdm
import time
from joblib import Parallel, delayed  # Parallel tracking
from joblib_progress import joblib_progress

from abel.classes.beam import *
#from abel.utilities.relativity import energy2gamma
from abel.utilities.relativity import momentum2gamma, velocity2gamma
from abel.utilities.plasma_physics import k_p
from abel.apis.rf_track.rf_track_api import rft_beam_fields


#def wakefunc_Stupakov(xi_lead, xi_ref, a):
#    return 2/(np.pi*eps0*a**4)*np.abs(xi_lead - xi_ref)  # [V/Cm^2]



# ==================================================
# Simplified loop integration option
def integrate_wake_func(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets, tr_momenta):
    
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
    return tr_momenta



# ==================================================
# Single pass integration option (Stupakov's wake function)
def single_pass_integrate_wake_func(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets, tr_momenta):
    
    a = bubble_radius + 0.75*skin_depth
    zzs = np.diff(zs_sorted)  # [m] z distance between neighbouring particles.

    # Calculate the derivative of the wakefield (Stupakov's wake function)
    dwakefields_dz_contribs = np.flip(2 / (np.pi * eps0 * a**4) * -e * weights_sorted * offsets)
    dwakefields_dz_contribs[0] = 0
    dwakefields_dz = np.cumsum(dwakefields_dz_contribs)  # Cumulative sum from 1st to last element.
    dwakefields_dz = np.flip(dwakefields_dz)
    
    # Calculate the wakefield on each macroparticle
    wakefields = np.zeros(len(zs_sorted))
    
    for idx_particle in range(len(zs_sorted)-2, -1, -1):
        wakefields[idx_particle] = wakefields[idx_particle+1] + zzs[idx_particle] * dwakefields_dz[idx_particle+1]

    # Calculate the total transverse force on macroparticles
    tr_force = -e*(wakefields + plasma_density*e*offsets/(2*eps0))
    
    # Update momenta
    tr_momenta = tr_momenta + tr_force*time_step
    return tr_momenta



# ==================================================
# Single pass integration of (Stupakov's wake function)
def calc_tr_momenta(beam, skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets, tot_offsets_sqr, tr_momenta, gammas, tr_direction='x', enable_tr_instability=True, enable_radiation_reaction=True, enable_ion_motion=False, ion_charge_num=1.0, ion_mass=None, sort_zs=False):
    
    a = bubble_radius + 0.75*skin_depth
    zzs = np.diff(zs_sorted)  # [m] z distance between neighbouring particles. zs_sorted[i+1]-zs_sorted[i]. zs_sorted increases towards larger indicies. I.e. coincide with beam head at the end of the array.
    

    # ============= Calculate the transverse wakefield =============
    if enable_tr_instability:  # TODO: Calculate the transverse wakefield using the split integral method instead.

        # ------------- Calculate the derivative of the wakefield (Stupakov's wake function) -------------
        dwakefields_dz_contribs = 2 / (np.pi * eps0 * a**4) * -e * weights_sorted * offsets
        dwakefields_dz_contribs[-1] = 0  # Set the contribution from the beam head to 0.
    
        # Cumulative sum from last to first element, i.e. from beam head to beam tail. This results in a reversed order where contributions from beam head are placed at the start of the dwakefields_dz array
        dwakefields_dz = np.cumsum(dwakefields_dz_contribs[::-1])
        
        # Flip the array to coincide with beam head facing the end of the array like all the other beam arrays
        dwakefields_dz = np.flip(dwakefields_dz)
        
        # ------------- Calculate the wakefield on each macroparticle -------------
        # Cumulative sum from last to first element, i.e. from beam head to beam tail. This results in a reversed order where contributions from beam head are placed at the start of the array
        wakefields = np.cumsum( (zzs * dwakefields_dz[:-1])[::-1] )

        # Flip the array to coincide with beam head facing the end of the array like all the other beam arrays
        wakefields = np.flip(wakefields)

        # Insert 0 at beam head wakefield
        wakefields = np.insert(arr=wakefields, obj=-1, values=0.0)
    else:
        wakefields = np.zeros(len(offsets))
        

    # ============= Calculate the effects of ion motion =============
    if enable_ion_motion:
        num_z_cells_rft = round(np.sqrt(len(offsets))/2)
        rft_z_grid_size = (beam.zs().max()-beam.zs().min())/num_z_cells_rft
        
        E_fields_beam, _, _, _, _ = rft_beam_fields(beam, num_x_cells=60, num_y_cells=60, num_z_cells=num_z_cells_rft, num_t_bins=4, sort_zs=sort_zs)  # beam needs to be sorted according to z. E_fields_beam is then alseo sorted according to z.
        
        if tr_direction == 'x':
            E_fields_comp = E_fields_beam[:,0]
        elif tr_direction == 'y':
            E_fields_comp = E_fields_beam[:,1]

        #print(len(beam), len(offsets), len(E_fields_comp))

        if ion_mass is None:
            ion_mass = 39.95*SI.physical_constants['atomic mass constant'][0]  # [kg], Ar mass.

        dionfields_dz = np.cumsum(E_fields_comp[::-1])
        dionfields_dz = np.flip(dionfields_dz)
        ion_wakefields = ion_charge_num * m_e/ion_mass/skin_depth**2 * np.cumsum( (zzs * dionfields_dz[:-1])[::-1] )*rft_z_grid_size
        ion_wakefields = np.flip(ion_wakefields)
        ion_wakefields = np.insert(arr=ion_wakefields, obj=-1, values=0.0)
        
        background_fields = plasma_density*e/(2*eps0)*offsets + ion_wakefields
        
    else:
        background_fields = plasma_density*e/(2*eps0)*offsets
        
    
    # ============= Calculate the total transverse force on macroparticles =============
    tr_force = -e*(wakefields + background_fields)
    
    
    # ============= Include radiation reaction if chosen and update momenta =============
    if enable_radiation_reaction:
        # Backward differentiation option (implicit method)
        denominators = 1 + c*1.87863e-15 * time_step * (1/skin_depth)**2/2 * (1+(1/skin_depth)**2/2*gammas*tot_offsets_sqr)
        tr_momenta = (tr_momenta + tr_force*time_step)/denominators
        
        # Forward differentiation option (direct method)
        #tr_momenta = tr_momenta + tr_force*time_step - c*1.87863e-15*(1/skin_depth)**2/2*tr_momenta*(1+(1/skin_depth)**2/2*gammas*tot_offsets_sqr)*time_step
    else:
        tr_momenta = tr_momenta + tr_force*time_step
    return tr_momenta


# ==================================================
#def doffset_dt(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets, tr_momenta, Ez, pzs_sorted):
#    tr_force = np.zeros(len(zs_sorted))  # [N] transverse force on each particle.
#    for idx_particle in range(len(zs_sorted)-1,-1,-1):  # Loops through all macroparticles
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
def transverse_wake_instability_particles(beam, plasma_density, Ez_fit_obj, rb_fit_obj, stage_length, time_step_mod=0.05, enable_tr_instability=True, enable_radiation_reaction=True, enable_ion_motion=False, ion_charge_num=1.0, ion_mass=None, show_prog_bar=True):
    
    xs = beam.xs()
    ys = beam.ys()
    zs = beam.zs()
    pxs = beam.pxs()
    pys = beam.pys()
    pzs = beam.pzs()
    weights = beam.weightings()
    particle_mass = beam.particle_mass

    # Check if the arrays are sorted based on zs
    if np.all(np.diff(zs) >= 0):
        zs_sorted = zs
        xs_sorted = xs
        ys_sorted = ys
        pxs_sorted = pxs
        pys_sorted = pys
        pzs_sorted = pzs
        weights_sorted = weights
    else:
        # Sort the arrays based on zs. TODO: checking will not pay off if the array is often unsorted.
        indices = np.argsort(zs)
        zs_sorted = zs[indices]
        xs_sorted = xs[indices]
        ys_sorted = ys[indices]
        pxs_sorted = pxs[indices]
        pys_sorted = pys[indices]
        pzs_sorted = pzs[indices]
        weights_sorted = weights[indices]

    # Filter out particles that have too small energies
    bool_indices = (pzs_sorted > velocity2gamma(0.99*c)*m_e*0.99*c)  # Corresponds to v=0.99c.
    zs_sorted = zs_sorted[bool_indices]
    xs_sorted = xs_sorted[bool_indices]
    ys_sorted = ys_sorted[bool_indices]
    pxs_sorted = pxs_sorted[bool_indices]
    pys_sorted = pys_sorted[bool_indices]
    pzs_sorted = pzs_sorted[bool_indices]
    weights_sorted = weights_sorted[bool_indices]

    # Calculate Ez and rb based on interpolations of Ez and rb vs z
    Ez = Ez_fit_obj(zs_sorted)
    bubble_radius = rb_fit_obj(zs_sorted)
    
    skin_depth = 1/k_p(plasma_density)  # [m] 1/kp, plasma skin depth.
    beta_func = c/e*np.sqrt(2* beam.gamma() * eps0*m_e/plasma_density)  # [m] matched beta function.
    beta_wave_length = 2*np.pi*beta_func  # [m] betatron wavelength.
    time_step = time_step_mod*beta_wave_length/c  # [s] beam time step.
    num_time_steps = np.ceil(stage_length/(c*time_step))
    time_step = stage_length/(c*num_time_steps)
    
    #print(time_step)
    #print(num_time_steps*c*time_step)
    #print(c*time_step)
    #print(stage_length-c*time_step)
    #print(stage_length)
    
    
    ############# Beam propagation through the plasma cell #############
    #time_step_count = 0.0
    prop_length = 0.0

    # Progress bar
    if show_prog_bar is True:
        pbar = tqdm(total=100)
        pbar.set_description('0%')

    while prop_length < stage_length-0.5*c*time_step:

        # ============= Apply filters =============
        # Filter out particles that have too small energies
        bool_indices = (pzs_sorted > velocity2gamma(0.99*c)*m_e*c*0.99)  # Corresponds to 0.99c.
        zs_sorted, xs_sorted, ys_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius = bool_indices_filter(bool_indices, zs_sorted, xs_sorted, ys_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius)
        
        # Filter out particles that diverge too much for applying small angle approximation
        bool_indices = (np.abs(pxs_sorted/pzs_sorted) < 50e-3)
        zs_sorted, xs_sorted, ys_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius = bool_indices_filter(bool_indices, zs_sorted, xs_sorted, ys_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius)

        bool_indices = (np.abs(pys_sorted/pzs_sorted) < 50e-3)
        zs_sorted, xs_sorted, ys_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius = bool_indices_filter(bool_indices, zs_sorted, xs_sorted, ys_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius)

        # Filter out particles that collide into bubble
        tot_offsets_sqr = xs_sorted**2 + ys_sorted**2
        bool_indices = (np.sqrt(tot_offsets_sqr) - bubble_radius <= 0)
        zs_sorted, xs_sorted, ys_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius = bool_indices_filter(bool_indices, zs_sorted, xs_sorted, ys_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius)
        
        
        # ============= Drift of beam =============
        # Leapfrog
        #time_step_count = time_step_count + 1/2
        prop_length = prop_length + 1/2*c*time_step
        xs_sorted = xs_sorted + pxs_sorted/pzs_sorted*1/2*c*time_step
        ys_sorted = ys_sorted + pys_sorted/pzs_sorted*1/2*c*time_step

        
        # ============= Integrate the wake function =============
        gammas = momentum2gamma(pzs_sorted)  # Lorentz factor for each particle.
        tot_offsets_sqr = xs_sorted**2 + ys_sorted**2
        filtered_beam = Beam()
        
        # set the phase space of the ABEL beam
        filtered_beam.set_phase_space(Q=np.sum(weights_sorted)*beam.charge_sign()*e,
                                      xs=xs_sorted,
                                      ys=ys_sorted,
                                      zs=zs_sorted, 
                                      pxs=pxs_sorted,
                                      pys=pys_sorted,
                                      pzs=pzs_sorted,
                                      weightings=weights_sorted,
                                      particle_mass=particle_mass)

        #calc_tr_momenta(filtered_beam, skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets=xs_sorted, tot_offsets_sqr=tot_offsets_sqr, tr_momenta=pxs_sorted, gammas=gammas, enable_tr_instability=enable_tr_instability, enable_radiation_reaction=enable_radiation_reaction, enable_ion_motion=enable_ion_motion, ion_charge_num=ion_charge_num, ion_mass=ion_mass)
        
        # Parallel tracking
        results = Parallel(n_jobs=2)([
            #delayed(integrate_wake_func)(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets=xs_sorted, tr_momenta=pxs_sorted),
            #delayed(integrate_wake_func)(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets=ys_sorted, tr_momenta=pys_sorted)
            #delayed(single_pass_integrate_wake_func)(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets=xs_sorted, tr_momenta=pxs_sorted),
            #delayed(single_pass_integrate_wake_func)(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets=ys_sorted, tr_momenta=pys_sorted)
            
            delayed(calc_tr_momenta)(filtered_beam, skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets=xs_sorted, tot_offsets_sqr=tot_offsets_sqr, tr_momenta=pxs_sorted, gammas=gammas, tr_direction='x', enable_tr_instability=enable_tr_instability, enable_radiation_reaction=enable_radiation_reaction, enable_ion_motion=enable_ion_motion, ion_charge_num=ion_charge_num, ion_mass=ion_mass),
            
            delayed(calc_tr_momenta)(filtered_beam, skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets=ys_sorted, tot_offsets_sqr=tot_offsets_sqr, tr_momenta=pys_sorted, gammas=gammas, tr_direction='y', enable_tr_instability=enable_tr_instability, enable_radiation_reaction=enable_radiation_reaction, enable_ion_motion=enable_ion_motion, ion_charge_num=ion_charge_num, ion_mass=ion_mass, sort_zs=False)
        ])
        # Update momenta
        pxs_sorted = results[0]
        pys_sorted = results[1]

        #Ez = -3.35e9*np.ones(len(pzs_sorted))  # [V/m] Overload with constant field to see how this affects instability. # <- ###########################
        #Ez = -3.20e9*np.ones(len(pzs_sorted))  # [V/m] Overload with constant field to see how this affects instability. # <- ###########################
        #Ez = -2.0e9*np.ones(len(pzs_sorted))  # [V/m] Overload with constant field to see how this affects instability. # <- ######################
        
        # Update longitudinal momenta.
        if enable_radiation_reaction:
            pzs_sorted = pzs_sorted - (e*Ez + m_e*c**2 * 1.87863e-15 * (1/skin_depth)**4/4 * gammas**2 * tot_offsets_sqr)*time_step
        else:
            pzs_sorted = pzs_sorted - e*Ez*time_step
        
        
        # ============= Drift of beam =============
        # Leapfrog
        xs_sorted = xs_sorted + pxs_sorted/pzs_sorted*1/2*c*time_step
        ys_sorted = ys_sorted + pys_sorted/pzs_sorted*1/2*c*time_step
        #time_step_count = time_step_count + 1/2
        prop_length = prop_length + 1/2*c*time_step

        # RK4
        #xs_sorted = RK4(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, xs_sorted, pxs_sorted, Ez, pzs_sorted)
        #ys_sorted = RK4(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, ys_sorted, pys_sorted, Ez, pzs_sorted)
        #time_step_count = time_step_count + 1
        #prop_length = prop_length + c*time_step
        
        
        # Progress bar
        if show_prog_bar is True:
            pbar.update(prop_length/stage_length*100 - pbar.n)        
            pbar.set_description(f"Instability tracking {round(prop_length/stage_length*100,2)}%")
        
        
    ############# End of loop #############
    
    # ============= Filter out particles that collide into bubble =============
    tot_offsets_sqr = xs_sorted**2 + ys_sorted**2
    bool_indices = (np.sqrt(tot_offsets_sqr) - bubble_radius <= 0)
    zs_sorted, xs_sorted, ys_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, _, _ = bool_indices_filter(bool_indices, zs_sorted, xs_sorted, ys_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius)

    # Progress bar
    if show_prog_bar is True:
        pbar.close()
        
    # Initialise ABEL Beam object
    beam_out = Beam()
        
    # set the phase space of the ABEL beam
    beam_out.set_phase_space(Q=np.sum(weights_sorted)*beam.charge_sign()*e,
                             xs=xs_sorted,
                             ys=ys_sorted,
                             zs=zs_sorted, 
                             pxs=pxs_sorted,  # Always use single particle momenta?
                             pys=pys_sorted,
                             pzs=pzs_sorted,
                             weightings=weights_sorted,
                             particle_mass=particle_mass)
    
    return beam_out



# ==================================================
def bool_indices_filter(bool_indices, zs_sorted, xs_sorted, ys_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius):
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

    return zs_sorted, xs_sorted, ys_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius