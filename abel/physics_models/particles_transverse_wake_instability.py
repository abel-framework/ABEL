"""
Transverse wake instability model as described in thesis "Instability and Beam-Beam Study for Multi-TeV PWFA e+e- and gamma gamma Linear Colliders" (https://cds.cern.ch/record/2754022?ln=en).

Ben Chen, 5 October 2023, University of Oslo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e, epsilon_0 as eps0

from tqdm import tqdm
import time
from joblib import Parallel, delayed  # Parallel tracking
from joblib_progress import joblib_progress
import csv, os
#from types import SimpleNamespace

from abel.classes.beam import *  # TODO: no need to import everything.
from abel.utilities.relativity import momentum2gamma, velocity2gamma
from abel.utilities.plasma_physics import k_p
from abel.utilities.statistics import weighted_mean, weighted_std
from abel.physics_models.ion_motion_wakefield_perturbation import IonMotionConfig, probe_driver_beam_field, assemble_main_sc_fields_obj, probe_main_beam_field, ion_wakefield_perturbation, intplt_ion_wakefield_perturbation


class PrtclTransWakeConfig():
# Stores configuration for the transverse wake instability calculations.

    # =============================================
    def __init__(self, plasma_density, stage_length, drive_beam=None, main_beam=None, time_step_mod=0.05, show_prog_bar=False, probe_evolution=False, probe_every_nth_time_step=1, make_animations=False, tmpfolder=None, shot_path=None, stage_num=None, enable_tr_instability=True, enable_radiation_reaction=True, enable_ion_motion=False, ion_charge_num=1.0, ion_mass=None, num_z_cells_main=None, num_x_cells_rft=50, num_y_cells_rft=50, num_xy_cells_probe=41, uniform_z_grid=False, driver_x_jitter=0.0, driver_y_jitter=0.0, update_factor=None, update_ion_wakefield=False):
        
        self.plasma_density = plasma_density  # [m^-3]
        self.stage_length = stage_length
        self.time_step_mod = time_step_mod
        self.enable_tr_instability = enable_tr_instability
        self.enable_radiation_reaction = enable_radiation_reaction
        self.enable_ion_motion = enable_ion_motion
        self.show_prog_bar = show_prog_bar

        self.make_animations = make_animations
        self.probe_evolution = probe_evolution
        if probe_evolution:
            if probe_every_nth_time_step <= 0 or isinstance(probe_every_nth_time_step, int) == False:
                raise ValueError('probe_every_nth_time_step has to be an integer larger than 0')
            self.probe_every_nth_time_step = probe_every_nth_time_step
            self.tmpfolder = tmpfolder
            self.shot_path = shot_path
            self.stage_num = stage_num

        if enable_ion_motion:
            if update_factor is None:
                update_factor=time_step_mod  # The default is to update the ion wakefield perturbation at every time step.
                
            self.ion_motion_config = IonMotionConfig(
                drive_beam=drive_beam, 
                main_beam=main_beam, 
                plasma_ion_density=plasma_density, 
                ion_charge_num=ion_charge_num, 
                ion_mass=ion_mass, 
                num_z_cells_main=num_z_cells_main, 
                num_x_cells_rft=num_x_cells_rft, 
                num_y_cells_rft=num_y_cells_rft, 
                num_xy_cells_probe=num_xy_cells_probe, 
                uniform_z_grid=uniform_z_grid, 
                driver_x_jitter=driver_x_jitter, 
                driver_y_jitter=driver_y_jitter, 
                update_factor=update_factor, 
                update_ion_wakefield=update_ion_wakefield
            )



###################################################
def calc_tr_instability_wakefield(skin_depth, bubble_radius, zs_sorted, weights_sorted, offsets):
    """
    Single pass integration of intra-beam wakefield (Stupakov's wake function).
    """
    
    a = bubble_radius + 0.75*skin_depth
    zzs = np.diff(zs_sorted)  # [m] z distance between neighbouring particles. zs_sorted[i+1]-zs_sorted[i]. zs_sorted increases towards larger indicies. I.e. coincide with beam head at the end of the array.
    
    # TODO: Calculate the transverse wakefield using the split integral method instead.

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

    return wakefields



###################################################
def calc_ion_wakefield_perturbation(beam, drive_beam, trans_wake_config):
    
    if trans_wake_config.enable_ion_motion:
        ion_motion_config = trans_wake_config.ion_motion_config
        
        # Set the coordinates used to probe beam electric fields from RF-Track
        ion_motion_config.set_probing_coordinates(drive_beam, main_beam=beam, set_driver_sc_coords=False)
        
        #ion_motion_config.update_ion_wakefield = True #######<- Override

        # Check if ion wakefield perturbation should be updated for the current time step
        if ion_motion_config.update_ion_wakefield:

            # Extract drive beam RF-Track SpaceCharge_Field object
            driver_sc_fields_obj = ion_motion_config.driver_sc_fields_obj
    
            # Probe drive beam E-field component in 3D
            driver_Exs_3d, driver_Eys_3d = probe_driver_beam_field(ion_motion_config, driver_sc_fields_obj)
            
            # Update the RF-Track SpaceCharge_Field object for the main beam
            sc_fields_obj = assemble_main_sc_fields_obj(ion_motion_config, main_beam=beam)
            
            # Probe main beam field component in 3D
            main_Exs_3d, main_Eys_3d = probe_main_beam_field(ion_motion_config, sc_fields_obj)
            
            # Calculate the ion wakefield perturbation
            Wx_perts, Wy_perts = ion_wakefield_perturbation(ion_motion_config, main_Exs_3d, main_Eys_3d, driver_Exs_3d, driver_Eys_3d)  # [V/m], 3D array
    
            # Interpolate the ion wakefield perturbation to macroparticle positions
            intpl_Wx_perts, _ = intplt_ion_wakefield_perturbation(beam, Wx_perts, ion_motion_config, intplt_beam_region_only=True)  # [V/m], 1D array
            intpl_Wy_perts, _ = intplt_ion_wakefield_perturbation(beam, Wy_perts, ion_motion_config, intplt_beam_region_only=True)  # [V/m], 1D array
    
            # Save Wx_perts and Wy_perts for time steps that skip calculating the wakefield
            trans_wake_config.ion_motion_config.Wx_perts = Wx_perts
            trans_wake_config.ion_motion_config.Wy_perts = Wy_perts
        
        else:
            Wx_perts = ion_motion_config.Wx_perts
            intpl_Wx_perts, _ = intplt_ion_wakefield_perturbation(beam, Wx_perts, ion_motion_config, intplt_beam_region_only=True)  # [V/m], 1D array
            Wy_perts = ion_motion_config.Wy_perts
            intpl_Wy_perts, _ = intplt_ion_wakefield_perturbation(beam, Wy_perts, ion_motion_config, intplt_beam_region_only=True)  # [V/m], 1D array
            
    else:  # No ion motion
        intpl_Wx_perts = np.zeros_like(beam.zs())
        intpl_Wy_perts = np.zeros_like(beam.zs())
            
    return intpl_Wx_perts, intpl_Wy_perts
    


###################################################
def calc_tr_momenta_comp(trans_wake_config, skin_depth, plasma_density, time_step, bubble_radius, zs_sorted, weights_sorted, gammas, tot_offsets_sqr, offsets, tr_momenta_comp, ion_wakefield_perts):
    """
    Calculates the transverse momentum component acting on each beam macroparticle.
    """
    
    enable_tr_instability = trans_wake_config.enable_tr_instability
    enable_radiation_reaction = trans_wake_config.enable_radiation_reaction
    
    # ============= Calculate the transverse intra-beam wakefield =============
    if enable_tr_instability:
        intra_beam_wakefields = calc_tr_instability_wakefield(skin_depth, bubble_radius, zs_sorted, weights_sorted, offsets)
    
    else:  # No intra-beam transverse wake instability
        intra_beam_wakefields = np.zeros_like(zs_sorted)
    
    
    # ============= Calculate the total transverse force component on macroparticles =============
    
    # Calculate the perturbed ion wakefield at macroparticle positions
    ion_wakefields = plasma_density*e/(2*eps0)*offsets - ion_wakefield_perts  # [V/m], 1D array
    
    # Calculate the transverse force component on macroparticles
    tr_forces_comp = -e*(intra_beam_wakefields + ion_wakefields)  # [N], 1D array
    
    
    # ============= Include radiation reaction if chosen and update momenta =============
    if enable_radiation_reaction:
        # Backward differentiation option (implicit method)
        denominators = 1 + c*1.87863e-15 * time_step * (1/skin_depth)**2/2 * (1+(1/skin_depth)**2/2*gammas*tot_offsets_sqr)
        tr_momenta_comp = (tr_momenta_comp + tr_forces_comp * time_step) / denominators
        
        # Forward differentiation option (direct method)
        #tr_momenta_comp = tr_momenta_comp + tr_forces_comp*time_step - c*1.87863e-15*(1/skin_depth)**2/2*tr_momenta_comp*(1+(1/skin_depth)**2/2*gammas*tot_offsets_sqr)*time_step
    
    else:  # No radiation reaction
        tr_momenta_comp = tr_momenta_comp + tr_forces_comp*time_step
    
    return tr_momenta_comp



###################################################
def transverse_wake_instability_particles(beam, drive_beam, Ez_fit_obj, rb_fit_obj, trans_wake_config):
    """
    Parameters
    ----------
    beam: Beam object
    
    #plasma_density: [m^-3] float
        Plasma density.
        
    #Ez_fit_obj: [V/m] interpolation object
        1D interpolation object of longitudinal E-field fitted to axial E-field using a selection of zs along the main beam. Used to determine the value of the longitudinal E-field for all beam zs.
        
    #rb_fit_obj: [m] interpolation object
        1D interpolation object of plasma bubble radius fitted to axial bubble radius using a selection of zs along the main beam. Used to determine the value of the bubble radius for all beam zs.
        
    #stage_length: [m] float
        Length of the plasma stage.
    
    #time_step_mod: float
        Determines the time step of the instability tracking in units of beta_wave_length/c.
    
    #enable_radiation_reaction: bool
        Flag for enabling radiation reactions.
    
    #show_prog_bar: bool
        Flag for displaying the progress bar.

    trans_wake_config: ...
        ...
    
    ...
    
        
    Returns
    ----------
    beam_out: Beam object
        ...
    
    evolution: ...
        ...
    """
    
    plasma_density = trans_wake_config.plasma_density
    stage_length = trans_wake_config.stage_length
    time_step_mod = trans_wake_config.time_step_mod
    enable_radiation_reaction = trans_wake_config.enable_radiation_reaction
    enable_ion_motion = trans_wake_config.enable_ion_motion
    show_prog_bar = trans_wake_config.show_prog_bar

    skin_depth = 1/k_p(plasma_density)  # [m] 1/kp, plasma skin depth.
    beta_func = c/e*np.sqrt(2* beam.gamma() * eps0*m_e/plasma_density)  # [m] matched beta function.
    beta_wave_length = 2*np.pi*beta_func  # [m] betatron wavelength.
    time_step = time_step_mod*beta_wave_length/c  # [s] beam time step.
    num_time_steps = int(np.ceil(stage_length/(c*time_step)))
    time_step = stage_length/(c*num_time_steps)
    
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

    # Determine the propagation axis
    x_axis = drive_beam.x_offset()
    y_axis = drive_beam.y_offset()

    # Calculate Ez and rb based on interpolations of Ez and rb vs z
    Ez = Ez_fit_obj(zs_sorted)
    bubble_radius = rb_fit_obj(zs_sorted)

    if enable_ion_motion:
        ion_motion_config = trans_wake_config.ion_motion_config
        update_factor = np.max([time_step_mod, ion_motion_config.update_factor])
        ion_motion_config.update_ion_wakefield = True  # Need to be initially true for the first ion wakefield calculation.
        update_freq = int(update_factor/time_step_mod)
        #driver_sc_fields_obj = ion_motion_config.assemble_driver_sc_fields_obj()
        #ion_motion_config.driver_sc_fields_obj = driver_sc_fields_obj
        
    
    ############# Beam propagation through the plasma cell #############
    time_step_count = 0
    prop_length = 0.0
    filtered_beam = Beam()

    # Progress bar
    if show_prog_bar is True:
        pbar = tqdm(total=100, bar_format='{desc} {percentage:3.1f}%|{bar}| [{elapsed}, {rate_fmt}{postfix}]')
        

    # ============= Save evolution =============
    if trans_wake_config.probe_evolution:
        if trans_wake_config.probe_every_nth_time_step > num_time_steps:
            probe_data_freq = num_time_steps
        else:
            probe_data_freq = trans_wake_config.probe_every_nth_time_step
        
        current_beam = Beam()
        evolution = Evolution( data_length=1+int(np.ceil( (num_time_steps - 1)/probe_data_freq ) ) )
            
    else:
        evolution = Evolution( data_length=0 )

    
    while prop_length < stage_length-0.5*c*time_step:

        
         # ============= Save evolution =============
        if trans_wake_config.probe_evolution and time_step_count % probe_data_freq == 0:
            current_beam.set_phase_space(Q=np.sum(weights_sorted)*beam.charge_sign()*e,
                                         xs=xs_sorted,
                                         ys=ys_sorted,
                                         zs=zs_sorted,
                                         pxs=pxs_sorted,
                                         pys=pys_sorted,
                                         pzs=pzs_sorted,
                                         weightings=weights_sorted,
                                         particle_mass=particle_mass)
            
            evolution.save_evolution(prop_length, current_beam, drive_beam, clean=False)
            
            if trans_wake_config.make_animations:
                save_beam(current_beam, trans_wake_config.tmpfolder, trans_wake_config.stage_num, time_step_count, num_time_steps)

        
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

        
        # ============= Beam kick =============
        # Set the phase space of the filtered ABEL beam
        #filtered_beam = Beam()
        filtered_beam.set_phase_space(Q=np.sum(weights_sorted)*beam.charge_sign()*e,
                                      xs=xs_sorted,
                                      ys=ys_sorted,
                                      zs=zs_sorted, 
                                      pxs=pxs_sorted,
                                      pys=pys_sorted,
                                      pzs=pzs_sorted,
                                      weightings=weights_sorted,
                                      particle_mass=particle_mass)
        
        #gammas = momentum2gamma(pzs_sorted)  # Lorentz factor for each particle.
        gammas = momentum2gamma( np.sqrt(pxs_sorted**2 + pys_sorted**2 + pzs_sorted**2) )  # Lorentz factor for each particle.
        tot_axis_offsets_sqr = (xs_sorted - x_axis)**2 + (ys_sorted - y_axis)**2

        #ion_start_time = time.time()

        # Calculate the ion wakefield perturbations
        intpl_Wx_perts, intpl_Wy_perts = calc_ion_wakefield_perturbation(filtered_beam, drive_beam, trans_wake_config)
        
        #ion_end_time = time.time()
        #print('Ion wake calc time taken:', ion_end_time - ion_start_time, 'seconds')
        

        # Update the transverse momenta components
        pxs_sorted = calc_tr_momenta_comp(trans_wake_config, skin_depth, plasma_density, time_step, bubble_radius, zs_sorted, weights_sorted, gammas, tot_axis_offsets_sqr, offsets=xs_sorted-x_axis, tr_momenta_comp=pxs_sorted, ion_wakefield_perts=intpl_Wx_perts)
            
        pys_sorted = calc_tr_momenta_comp(trans_wake_config, skin_depth, plasma_density, time_step, bubble_radius, zs_sorted, weights_sorted, gammas, tot_axis_offsets_sqr, offsets=ys_sorted-y_axis, tr_momenta_comp=pys_sorted, ion_wakefield_perts=intpl_Wy_perts)
        
        
        #Ez = -3.35e9*np.ones(len(pzs_sorted))  # [V/m] Overload with constant field to see how this affects instability. # <- ###########################
        #Ez = -3.20e9*np.ones(len(pzs_sorted))  # [V/m] Overload with constant field to see how this affects instability. # <- ###########################
        #Ez = -2.0e9*np.ones(len(pzs_sorted))  # [V/m] Overload with constant field to see how this affects instability. # <- ######################

        
        # Update longitudinal momenta.
        if enable_radiation_reaction:
            pzs_sorted = pzs_sorted - (e*Ez + m_e*c**2 * 1.87863e-15 * (1/skin_depth)**4/4 * gammas**2 * tot_axis_offsets_sqr)*time_step
        else:
            pzs_sorted = pzs_sorted - e*Ez*time_step
            
        # Save data
        #if time_step_count == 0 and trans_wake_config.probe_evolution:
            #file_path = trans_wake_config.shot_path[0:-1] + '_tr_wake_data' + os.sep + str(trans_wake_config.stage_num).zfill(3) + '_' + str(time_step_count).zfill(len(str(int(num_time_steps)))) + '.csv'
            #save_time_step([xs_sorted, ys_sorted, zs_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius, intpl_Wx_perts, intpl_Wy_perts], file_path)

        
        # ============= Drift of beam =============
        # Leapfrog
        x_angles = pxs_sorted/pzs_sorted
        xs_sorted = xs_sorted + x_angles*1/2*c*time_step
        y_angles = pys_sorted/pzs_sorted
        ys_sorted = ys_sorted + y_angles*1/2*c*time_step
        #time_step_count = time_step_count + 1/2
        prop_length = prop_length + 1/2*c*time_step
        
        time_step_count = time_step_count + 1

        if enable_ion_motion:
            if time_step_count % update_freq == 0:
                ion_motion_config.update_ion_wakefield = True
            else:
                ion_motion_config.update_ion_wakefield = False

        ## ============= Save evolution =============
        #if trans_wake_config.probe_evolution and time_step_count % probe_data_freq == 0:
            #file_path = trans_wake_config.shot_path[0:-1] + '_tr_wake_data' + os.sep + str(trans_wake_config.stage_num).zfill(3) + '_' + str(time_step_count).zfill(len(str(int(num_time_steps)))) + '.csv'
            #save_time_step([xs_sorted, ys_sorted, zs_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius, intpl_Wx_perts, intpl_Wy_perts], file_path)

        
        
        
                
        # RK4
        #xs_sorted = RK4(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, xs_sorted, pxs_sorted, Ez, pzs_sorted)
        #ys_sorted = RK4(skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, ys_sorted, pys_sorted, Ez, pzs_sorted)
        #time_step_count = time_step_count + 1
        #prop_length = prop_length + c*time_step
        
        
        # Progress bar
        if show_prog_bar is True:
            pbar.update(prop_length/stage_length*100 - pbar.n)
            pbar.set_description(f"\tInstability stage tracking")

        #print(time_step_count)
        #break #####################< override
        
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
        
    # Set the phase space of the ABEL beam
    beam_out.set_phase_space(Q=np.sum(weights_sorted)*beam.charge_sign()*e,
                             xs=xs_sorted,
                             ys=ys_sorted,
                             zs=zs_sorted, 
                             pxs=pxs_sorted,  # Always use single particle momenta?
                             pys=pys_sorted,
                             pzs=pzs_sorted,
                             weightings=weights_sorted,
                             particle_mass=particle_mass)
    
    
    # ============= Save evolution =============
    if trans_wake_config.probe_evolution:
        evolution.beam.plasma_density = plasma_density*np.ones_like(evolution.beam.location)
        evolution.driver.plasma_density = plasma_density*np.ones_like(evolution.driver.location)

        if evolution.index < len(evolution.beam.location):
        # Last step in the evolution arrays already written to if num_time_steps % probe_data_freq != 0.
            evolution.save_evolution(prop_length, beam_out, drive_beam, clean=False)
        
    if trans_wake_config.make_animations:
        save_beam(beam_out, trans_wake_config.tmpfolder, trans_wake_config.stage_num, time_step_count, num_time_steps)
            
    return beam_out, evolution



###################################################
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



###################################################
class Evolution:
    
    # =============================================
    def __init__(self, data_length):
        self.index = 0  # Keeping track of the current index.
        
        self.beam = SimpleNamespace()
        self.driver = SimpleNamespace()
        
        self.beam.location = np.empty(data_length)
        self.beam.x = np.empty(data_length)
        self.beam.y = np.empty(data_length)
        self.beam.z = np.empty(data_length)
        self.beam.energy = np.empty(data_length)
        self.beam.x_angle = np.empty(data_length)
        self.beam.y_angle = np.empty(data_length)
        self.beam.beam_size_x = np.empty(data_length)
        self.beam.beam_size_y = np.empty(data_length)
        self.beam.bunch_length = np.empty(data_length)
        self.beam.rel_energy_spread = np.empty(data_length)
        self.beam.divergence_x = np.empty(data_length)
        self.beam.divergence_y = np.empty(data_length)
        self.beam.beta_x = np.empty(data_length)
        self.beam.beta_y = np.empty(data_length)
        self.beam.emit_nx = np.empty(data_length)
        self.beam.emit_ny = np.empty(data_length)
        self.beam.num_particles = np.empty(data_length)
        self.beam.charge = np.empty(data_length)
        self.beam.plasma_density = np.empty(data_length)  # [m^-3]
        
        self.driver.location = np.empty(data_length)
        self.driver.x = np.empty(data_length)
        self.driver.y = np.empty(data_length)
        self.driver.z = np.empty(data_length)
        self.driver.energy = np.empty(data_length)
        self.driver.x_angle = np.empty(data_length)
        self.driver.y_angle = np.empty(data_length)
        self.driver.beam_size_x = np.empty(data_length)
        self.driver.beam_size_y = np.empty(data_length)
        self.driver.bunch_length = np.empty(data_length)
        self.driver.rel_energy_spread = np.empty(data_length)
        self.driver.divergence_x = np.empty(data_length)
        self.driver.divergence_y = np.empty(data_length)
        self.driver.beta_x = np.empty(data_length)
        self.driver.beta_y = np.empty(data_length)
        self.driver.emit_nx = np.empty(data_length)
        self.driver.emit_ny = np.empty(data_length)
        self.driver.num_particles = np.empty(data_length)
        self.driver.charge = np.empty(data_length)
        self.driver.plasma_density = np.empty(data_length)  # [m^-3]

    # =============================================
    def save_evolution(self, location, beam, driver, clean):
        if self.index >= len(self.beam.location):
            raise ValueError('Data recording already completed.')
            
        self.beam.location[self.index] = location
        self.beam.x[self.index] = beam.x_offset(clean=clean)
        self.beam.y[self.index] = beam.y_offset(clean=clean)
        self.beam.z[self.index] = beam.z_offset(clean=clean)
        self.beam.energy[self.index] = beam.energy(clean=clean)
        self.beam.x_angle[self.index] = beam.x_angle(clean=clean)
        self.beam.y_angle[self.index] = beam.y_angle(clean=clean)
        self.beam.beam_size_x[self.index] = beam.beam_size_x(clean=clean)
        self.beam.beam_size_y[self.index] = beam.beam_size_y(clean=clean)
        self.beam.bunch_length[self.index] = beam.bunch_length(clean=clean)
        self.beam.rel_energy_spread[self.index] = beam.rel_energy_spread(clean=clean)
        self.beam.divergence_x[self.index] = beam.divergence_x(clean=clean)
        self.beam.divergence_y[self.index] = beam.divergence_y(clean=clean)
        self.beam.beta_x[self.index] = beam.beta_x(clean=clean)
        self.beam.beta_y[self.index] = beam.beta_y(clean=clean)
        self.beam.emit_nx[self.index] = beam.norm_emittance_x(clean=clean)
        self.beam.emit_ny[self.index] = beam.norm_emittance_y(clean=clean)
        self.beam.num_particles[self.index] = len(beam)
        self.beam.charge[self.index] = beam.charge()
        #self.beam.plasma_density = plasma_density
        
        self.driver.location[self.index] = location
        self.driver.x[self.index] = driver.x_offset(clean=clean)
        self.driver.y[self.index] = driver.y_offset(clean=clean)
        self.driver.z[self.index] = driver.z_offset(clean=clean)
        self.driver.energy[self.index] = driver.energy(clean=clean)
        self.driver.x_angle[self.index] = driver.x_angle(clean=clean)
        self.driver.y_angle[self.index] = driver.y_angle(clean=clean)
        self.driver.beam_size_x[self.index] = driver.beam_size_x(clean=clean)
        self.driver.beam_size_y[self.index] = driver.beam_size_y(clean=clean)
        self.driver.bunch_length[self.index] = driver.bunch_length(clean=clean)
        self.driver.rel_energy_spread[self.index] = driver.rel_energy_spread(clean=clean)
        self.driver.divergence_x[self.index] = driver.divergence_x(clean=clean)
        self.driver.divergence_y[self.index] = driver.divergence_y(clean=clean)
        self.driver.beta_x[self.index] = driver.beta_x(clean=clean)
        self.driver.beta_y[self.index] = driver.beta_y(clean=clean)
        self.driver.emit_nx[self.index] = driver.norm_emittance_x(clean=clean)
        self.driver.emit_ny[self.index] = driver.norm_emittance_y(clean=clean)
        self.driver.num_particles[self.index] = len(driver)
        self.driver.charge[self.index] = driver.charge()

        self.index += 1



###################################################
def save_beam(main_beam, file_path, stage_num, time_step, num_time_steps):
    main_beam.save(filename=file_path + 'main_beam_' + str(stage_num).zfill(3) + '_' + str(time_step).zfill(len(str(int(num_time_steps)))) + '.h5')



###################################################
def save_time_step(arrays, file_path):
    """
    arrays: List of arrays in the order
        xs_sorted, ys_sorted, zs_sorted, pxs_sorted, pys_sorted, pzs_sorted, weights_sorted, Ez, bubble_radius, intpl_Wx_perts, intpl_Wy_perts
    """
    
    # Headers for the columns
    headers = ['x [m]', 'y [m]', 'z [m]', 'px [kg m/s]', 'py [kg m/s]', 'pz [kg m/s]', 'Weights', 'Ez [V/m]', 'Bubble radius [m]', 'Wx perturbation [V/m]', 'Wy perturbation [V/m]']

    # Stack them into columns for CSV output
    data = np.column_stack(arrays)

    # Create parent directories if they do not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Open the file for writing (creates the file if it does not exist)
    with open(file_path, mode='w', newline='') as file:
        np.savetxt(file, data, delimiter=",", header=",".join(headers))

        # File is automatically closed when exiting the 'with' block



#vvvvvvvvvvvvvvvvvvvvvv Not currently in use vvvvvvvvvvvvvvvvvvvvvv
'''
###################################################
#def wakefunc_Stupakov(xi_lead, xi_ref, a):
#    return 2/(np.pi*eps0*a**4)*np.abs(xi_lead - xi_ref)  # [V/Cm^2]



###################################################
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



###################################################
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

    
    
###################################################
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
###################################################
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



###################################################
# Single pass integration of (Stupakov's wake function)
def calc_tr_momenta(beam, skin_depth, plasma_density, time_step, zs_sorted, bubble_radius, weights_sorted, offsets, tot_offsets_sqr, tr_momenta, gammas, tr_direction, trans_wake_config):

    enable_tr_instability = trans_wake_config.enable_tr_instability
    enable_radiation_reaction = trans_wake_config.enable_radiation_reaction
    enable_ion_motion = trans_wake_config.enable_ion_motion
    
    a = bubble_radius + 0.75*skin_depth
    zzs = np.diff(zs_sorted)  # [m] z distance between neighbouring particles. zs_sorted[i+1]-zs_sorted[i]. zs_sorted increases towards larger indicies. I.e. coincide with beam head at the end of the array.
    
    # ============= Calculate the transverse intra-beam wakefield =============
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

        ion_motion_config = trans_wake_config.ion_motion_config
        drive_beam = ion_motion_config.drive_beam
        
        # Set the coordinates used to probe beam electric fields from RF-Track
        ion_motion_config.set_probing_coordinates(drive_beam, main_beam=beam)

        #ion_motion_config.update_ion_wakefield = True #######<- Override

        if ion_motion_config.update_ion_wakefield:

            # Extract drive beam RF-Track SpaceCharge_Field object
            driver_sc_fields_obj = ion_motion_config.driver_sc_fields_obj

            # Probe drive beam field component in 3D
            driver_E_comp_3d = probe_driver_beam_field_comp(ion_motion_config, driver_sc_fields_obj, field_comp=tr_direction)

            # Update the RF-Track SpaceCharge_Field object for the main beam
            sc_fields_obj = assemble_main_sc_fields_obj(ion_motion_config, main_beam=beam)

            # Probe main beam field component in 3D
            main_E_comp_3d = probe_main_beam_field(ion_motion_config, sc_fields_obj, field_comp=tr_direction)
            
            # Calculate the ion wakefield perturbation. Component given by tr_direction.
            #W_perts = ion_wakefield_perturbation_parallel(ion_motion_config, sc_fields_obj, tr_direction=tr_direction)  # [V/m], 3D array, parallel calculation.
            W_perts = ion_wakefield_perturbation(ion_motion_config, main_E_comp_3d, driver_E_comp_3d, field_comp=tr_direction)  # [V/m], 3D array
    
            # Interpolate the ion wakefield perturbation to macroparticle positions
            intpl_W_perts, _ = intplt_ion_wakefield_perturbation(beam, W_perts, ion_motion_config, intplt_beam_region_only=True)  # [V/m], 1D array

            # ion_wakefield_perturbation() ensures that tr_direction is either 'x' or 'y'.
            if tr_direction == 'x':
                trans_wake_config.ion_motion_config.Wx_perts = W_perts
                if trans_wake_config.ion_motion_config.Wx_perts is None:
                    raise ValueError('trans_wake_config.ion_motion_config.Wx_perts set to None.')
            else:  
                trans_wake_config.ion_motion_config.Wy_perts = W_perts
                if trans_wake_config.ion_motion_config.Wy_perts is None:
                    raise ValueError('trans_wake_config.ion_motion_config.Wy_perts set to None.')
        
        else:
            if tr_direction == 'x':
                W_perts = ion_motion_config.Wx_perts
                intpl_W_perts, _ = intplt_ion_wakefield_perturbation(beam, W_perts, ion_motion_config, intplt_beam_region_only=True)  # [V/m], 1D array
            
            # ion_wakefield_perturbation() / ion_wakefield_perturbation_parallel() ensures that tr_direction is either 'x' or 'y'.
            else:
                W_perts = ion_motion_config.Wy_perts
                intpl_W_perts, _ = intplt_ion_wakefield_perturbation(beam, W_perts, ion_motion_config, intplt_beam_region_only=True)  # [V/m], 1D array

        # Calculate the perturbed ion wakefield at macroparticle positions
        background_fields = plasma_density*e/(2*eps0)*offsets - intpl_W_perts  # [V/m], 1D array 
        
    else:
        background_fields = plasma_density*e/(2*eps0)*offsets
        W_perts = None
        intpl_W_perts = np.zeros_like(offsets)

    
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
        
    return tr_momenta, W_perts, intpl_W_perts



###################################################
#def calc_tr_force_comp(plasma_density, offsets, intra_beam_wakefields, ion_wakefield_perts):
#    """
#    Calculates the transverse force component acting on each beam macroparticle.
#    """
#    
#    # Calculate the perturbed ion wakefield at macroparticle positions
#    ion_wakefields = plasma_density*e/(2*eps0)*offsets - ion_wakefield_perts  # [V/m], 1D array
#    
#    # Calculate the transverse force component on macroparticles
#    tr_forces_comp = -e*(intra_beam_wakefields + ion_wakefields)  # [N], 1D array
#
#    return tr_forces_comp
'''