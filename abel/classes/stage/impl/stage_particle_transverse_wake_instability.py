"""
Stage class with the transverse wake instability model as described in thesis 
"Instability and Beam-Beam Study for Multi-TeV PWFA e+e- and gamma gamma Linear 
Colliders" (https://cds.cern.ch/record/2754022?ln=en).

Ben Chen, 6 October 2023, University of Oslo
"""

import numpy as np
from scipy.constants import c, e, m_e, epsilon_0 as eps0
from matplotlib import pyplot as plt
import time

from types import SimpleNamespace
import os, copy, warnings, uuid, shutil

from abel.physics_models.particles_transverse_wake_instability import *
from abel.utilities.plasma_physics import beta_matched, blowout_radius
from abel.utilities.other import find_closest_value_in_arr, pad_downwards, pad_upwards
from abel.apis.wake_t.wake_t_api import run_single_step_wake_t
from abel.classes.stage.stage import Stage
from abel.CONFIG import CONFIG
from abel.classes.beam import Beam



class StagePrtclTransWakeInstability(Stage):
    """
    TODO: Short description
    None of lines in the docstring text should exceed this length ..........

        
    Attributes
    ----------
    nom_energy_gain : [eV] float
        The total energy gain of the stage and its ramps.
    
    nom_accel_gradient : [eV/m] float
        The effective nominal acceleration gradient for the stage and its ramps.

    length : [m] float
        The total length of the stage and its ramps.

    nom_energy_gain_flattop : [eV] float
        The total energy gain of the stage.

    nom_accel_gradient_flattop : [eV/m] float
        The nominal acceleration gradient for the stage.

    length : [m] float
        The total length of the stage.
    
    ...

    main_source : ``Source`` object
        Main beam source.

    ...


    Methods
    -------
    __init__(...)
        Class constructor...

    track(beam_incoming, savedepth=0, runnable=None, verbose=False)
        Tracks the particles through the stage.

    ...
    """

    # ==================================================
    def __init__(self, nom_accel_gradient=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=1.0, main_source=None, drive_beam=None, time_step_mod=0.05, show_prog_bar=None, test_beam_between_ramps=False, Ez_fit_obj=None, Ez_roi=None, rb_fit_obj=None, bubble_radius_roi=None, probe_evol_period=0, save_final_step=True, make_animations=False, enable_tr_instability=True, enable_radiation_reaction=True, enable_ion_motion=False, ion_charge_num=1.0, ion_mass=None, num_z_cells_main=None, num_x_cells_rft=50, num_y_cells_rft=50, num_xy_cells_probe=41, uniform_z_grid=False, ion_wkfld_update_period=1, drive_beam_update_period=0):
        """
        TODO: Short description
        None of lines in the docstring text should exceed this length ..........

        Parameters
        ----------
        ...

        driver_source : ``Source`` object of drive beam.
        
        main_source : ``Source`` object of main beam.

        driver_beam : ``Beam`` object of drive beam.

        #main_beam : ``Beam`` object of main beam.
        
        #length : [m] float
            Length of the plasma stage.
        
        nom_energy_gain : [eV] float
            Nominal/target energy gain of the acceleration stage.
        
        plasma_density : [m^-3] float
            Plasma density.

        time_step_mod : [beta_wave_length/c] float, optional
            Determines the time step of the instability tracking in units of 
            beta_wave_length/c.
            
        #Ez_fit_obj : [V/m] interpolation object
            1D interpolation object of longitudinal E-field fitted to axial 
            E-field using a selection of zs along the main beam. Used to 
            determine the value of the longitudinal E-field for all beam zs.

        #Ez_roi : [V/m] 1D ndarray
            Longitudinal E-field in the region of interest fitted to a selection 
            of zs along the main beam (main beam head to tail).

        #rb_fit_obj : [m] interpolation object
            1D interpolation object of plasma bubble radius fitted to axial 
            bubble radius using a selection of zs along the main beam. Used to 
            determine the value of the bubble radius for all beam zs.
        
        #bubble_radius_roi : [m] 1D ndarray
            The bubble radius in the region of interest fitted to a selection of 
            zs along the main beam.

        ramp_beta_mag : float, optional
            Used for demagnifying and magnifying beams passing through entrance 
            and exit plasma ramps. Default value: 5.0.

        enable_radiation_reaction : bool, optional
            Flag for enabling radiation reactions. Defaults to ``True``.

        ...

        probe_evol_period : int, optional
            Set to larger than 0 to determine the probing period for beam 
            evolution diagnostics. This is given in units of time steps, so that 
            e.g. ``probe_evol_period=3`` will probe the beam evolution every 3rd 
            time step. Default value: 0.

        ...

        ion_wkfld_update_period : int, optional
            Determines the ion wakefield perturbation update period. This is 
            given in units of time steps, so that e.g. 
            ``ion_wkfld_update_period=3`` will update the ion wakefield 
            perturbation every 3rd time step. Default value: 1.
        
        drive_beam_update_period : int, optional
            Set to larger than 0 to activate driver evolution and determine 
            the drive beam update period. Default value: 0.

        """
        
        super().__init__(nom_accel_gradient=nom_accel_gradient, nom_energy_gain=nom_energy_gain, plasma_density=plasma_density, driver_source=driver_source, ramp_beta_mag=ramp_beta_mag)
        
        self.main_source = main_source
        self.drive_beam = drive_beam

        self.time_step_mod = time_step_mod  # Determines the time step of the instability tracking in units of beta_wave_length/c.
        self.interstage_dipole_field = None

        # Physics flags
        self.enable_tr_instability = enable_tr_instability 
        self.enable_radiation_reaction = enable_radiation_reaction
        self.enable_ion_motion = enable_ion_motion

        # Ion motion parameters
        self.ion_charge_num = ion_charge_num
        self.ion_mass = ion_mass
        
        self.num_z_cells_main = num_z_cells_main
        self.num_x_cells_rft = num_x_cells_rft
        self.num_y_cells_rft = num_y_cells_rft
        self.num_xy_cells_probe = num_xy_cells_probe
        self.uniform_z_grid = uniform_z_grid
        self.ion_wkfld_update_period = ion_wkfld_update_period
        self.drive_beam_update_period = drive_beam_update_period

        # Longitudinal electric field and plasma ion bubble radius
        self.Ez_fit_obj = Ez_fit_obj  # [V/m] 1d interpolation object of longitudinal E-field fitted to Ez_axial using a selection of zs along the main beam.
        self.Ez_roi = Ez_roi  # [V/m] longitudinal E-field in the region of interest (main beam head to tail).
        #self.Ez_axial = None  # Moved to self.initial.plasma.wakefield.onaxis.Ezs
        #self.zs_Ez_axial = None  # Moved to self.initial.plasma.wakefield.onaxis.zs
        self.rb_fit_obj = rb_fit_obj  # [m] 1d interpolation object of bubble radius fitted to bubble_radius_axial using a selection of zs along the main beam.
        self.bubble_radius_roi = bubble_radius_roi  # [m] bubble radius in the region of interest.
        self.bubble_radius_axial = None
        self.zs_bubble_radius_axial = None
        self.estm_R_blowout = None  # [m] estimated (max) blowout radius to be calculated.
        
        self.main_num_profile = None
        self.z_slices = None
        self.main_slices_edges = None
        self.driver_num_profile = None
        self.driver_z_slices = None

        # Beam evolution diagnostics
        if isinstance(probe_evol_period, int) == False:
            raise ValueError('probe_evol_period has to be an integer.')
        self.probe_evol_period = probe_evol_period
        self.evolution = None
        self.make_animations = make_animations
        self.run_path = None
        self.save_final_step = save_final_step

        # Simulation flag
        self.show_prog_bar = show_prog_bar
        self.test_beam_between_ramps = test_beam_between_ramps

        # Bookkeeping quantities
        self.driver_to_wake_efficiency = None
        self.wake_to_beam_efficiency = None
        self.driver_to_beam_efficiency = None
        
        self.reljitter = SimpleNamespace()
        self.reljitter.plasma_density = 0

        self.interstages_enabled = False  # TODO: to be removed.

        # internally sampled values (given some jitter)
        self.__n = None 
        self.driver_initial = None

    
    # ==================================================
    # Track the particles through. Note that when called as part of a Linac object, a copy of the original stage (where no changes has been made) is sent to track() every time. All changes done to self here are saved to a separate stage under the Linac object.
    def track(self, beam_incoming, savedepth=0, runnable=None, verbose=False):
        """
        Tracks the particles through the stage.
        """

        #if self.parent is not None and self.upramp is not None and self.downramp is not None:
        #    raise ValueError('Currently does not support ramps with both upramp and downramp.')

        # Set the diagnostics directory
        if runnable is not None:
            self.run_path = runnable.run_path()
            shot_path = runnable.shot_path()
        else:
            shot_path = None

        # Override enable/disable progress bar
        self.show_prog_bar = verbose

        # Extract quantities
        if self.length_flattop is None:
            raise ValueError('length_flattop is not defined.')
        plasma_density = self.plasma_density
        #gamma0 = beam_incoming.gamma()
        
        self.stage_number = beam_incoming.stage_number

        
        # ========== Get the drive beam ==========
        if self.driver_source.jitter.x == 0 and self.driver_source.jitter.y == 0 and self.drive_beam is not None:    #############################
            driver_incoming = self.drive_beam  # This guarantees zero drive beam jitter between stages, as identical drive beams are used in every stage and not re-sampled.
        else:
            driver_incoming = self.driver_source.track()  # Generate a drive beam with jitter.
            #self.drive_beam = driver_incoming                    ######################


        # ========== Rotate the coordinate system of the beams ==========
        # Perform coordinate rotations before calling on upramp tracking.
        if self.parent is None:  # Ensures that this is the main stage and not a ramp.

            # Will only rotate the beam coordinate system if the driver source of the stage has angular jitter or angular offset
            drive_beam_rotated, beam_rotated = self.rotate_beam_coordinate_systems(driver_incoming, beam_incoming)


        # ========== Prepare ramps ==========
        # If ramps exist, set ramp lengths, nominal energies, nominal energy gains
        # and flattop nominal energy if not already done.
        self._prepare_ramps()

        
        # ========== Apply plasma density up ramp (demagnify beta function) ==========
        if self.upramp is not None:  # if self has an upramp

            # Pass the drive beam and main beam to track_upramp() and get the ramped beams in return
            beam0, drive_beam_ramped = self.track_upramp(beam_rotated, drive_beam_rotated)
        
        else:  # Do the following if there are no upramp (can be either a lone stage or the first upramp)
            if self.parent is None:  # Just a lone stage
                beam0 = copy.deepcopy(beam_rotated)
                drive_beam_ramped = copy.deepcopy(drive_beam_rotated)
            else:
                beam0 = copy.deepcopy(beam_incoming)
                drive_beam_ramped = copy.deepcopy(driver_incoming)

            if self.ramp_beta_mag is not None:

                beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=drive_beam_ramped)
                drive_beam_ramped.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=drive_beam_ramped)

                ## NOTE: beam0 and drive_beam_ramped should not be changed after this line to check for continuity between ramps and stage.


        # ========== Record longitudinal number profile ==========
        # Number profile N(z). Dimensionless, same as dN/dz with each bin multiplied with the widths of the bins.
        main_num_profile, z_slices = self.longitudinal_number_distribution(beam=beam0)
        self.z_slices = z_slices  # Update the longitudinal position of the beam slices needed to fit Ez and bubble radius.
        self.main_num_profile = main_num_profile

        driver_num_profile, driver_z_slices = self.longitudinal_number_distribution(beam=drive_beam_ramped)
        self.driver_num_profile = driver_num_profile
        self.driver_z_slices = driver_z_slices
        

        # ========== Wake-T simulation and extraction ==========
        # Extract driver xy-offsets for later use
        driver_x_offset = drive_beam_ramped.x_offset()
        driver_y_offset = drive_beam_ramped.y_offset()

        # Perform a single time step Wake-T simulation
        wake_t_evolution = run_single_step_wake_t(self.plasma_density, copy.deepcopy(drive_beam_ramped), copy.deepcopy(beam0))

        # Read the Wake-T simulation data
        self.store_rb_Ez_2stage(wake_t_evolution, copy.deepcopy(drive_beam_ramped), copy.deepcopy(beam0))

        
        # ========== Instability tracking ==========
        # Filter out beam particles outside of the plasma bubble
        beam_filtered = self.bubble_filter(copy.deepcopy(beam0), sort_zs=True)
        beam_filtered.location = beam0.location

        # Make plots if all beam particles are outside the bubble.
        if len(beam_filtered) == 0:
            zs = beam0.zs()
            indices = np.argsort(zs)
            zs_sorted = zs[indices]

            fig, axs = plt.subplots(nrows=1, ncols=2, layout='constrained', figsize=(5*2, 4))
            axs[0].plot(zs_sorted*1e6, rb_fit(zs_sorted)*1e6, 'r')
            axs[0].plot(zs_sorted*1e6, -rb_fit(zs_sorted)*1e6, 'r')
            axs[0].scatter(beam0.zs()*1e6, beam0.xs()*1e6)
            axs[0].set_xlabel(r'$z$ [µm]')
            axs[0].set_ylabel(r'$x$ [µm]')
            axs[1].plot(zs_sorted*1e6, rb_fit(zs_sorted)*1e6, 'r')
            axs[1].plot(zs_sorted*1e6, -rb_fit(zs_sorted)*1e6, 'r')
            axs[1].scatter(beam0.zs()*1e6, beam0.ys()*1e6)
            axs[1].set_xlabel(r'$z$ [µm]')
            axs[1].set_ylabel(r'$y$ [µm]')
            raise ValueError('No beam particle left.')
        
        if self.num_z_cells_main is None:
            self.num_z_cells_main = round(np.sqrt( len(drive_beam_ramped)+len(beam_filtered) )/2)
        
        # Set up animations for beam evolution inside the stage
        if self.make_animations:
            # Create the temporary folder
            parent_dir = CONFIG.temp_path
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            tmpfolder = os.path.join(parent_dir, str(uuid.uuid4())) + os.sep
            os.mkdir(tmpfolder)
        else:
            tmpfolder = None

        # # Prepare fields from Wake-T for use in drive beam tracking # TODO: remove when driver evolution has been implemented.
        # if self.drive_beam_update_period > 0:  # Do drive beam evolution
        #     wake_t_fields = plasma_stage.fields[-1]  # Extract the wakefields used for driver tracking.
        #     wake_t_fields.r_fld = wake_t_fields.r_fld + np.sqrt(driver_x_offset**2 + driver_y_offset**2)  # Compensate for the drive beam offset.
        # else:
        #     wake_t_fields = None

        # Set up the configuration for the instability model
        trans_wake_config = PrtclTransWakeConfig(
            plasma_density=self.plasma_density, 
            stage_length=self.length_flattop, 
            drive_beam=drive_beam_ramped, 
            main_beam=beam_filtered, 
            time_step_mod=self.time_step_mod, 
            show_prog_bar=self.show_prog_bar,
            probe_evol_period=self.probe_evol_period, 
            make_animations=self.make_animations, 
            tmpfolder=tmpfolder, 
            shot_path=shot_path, 
            stage_num=beam_incoming.stage_number, 
            enable_tr_instability=self.enable_tr_instability, 
            enable_radiation_reaction=self.enable_radiation_reaction, 
            enable_ion_motion=self.enable_ion_motion, 
            ion_charge_num=self.ion_charge_num, 
            ion_mass=self.ion_mass, 
            num_z_cells_main=self.num_z_cells_main, 
            num_x_cells_rft=self.num_x_cells_rft, 
            num_y_cells_rft=self.num_y_cells_rft, 
            num_xy_cells_probe=self.num_xy_cells_probe, 
            uniform_z_grid=self.uniform_z_grid, 
            driver_x_jitter=self.driver_source.jitter.x, 
            driver_y_jitter=self.driver_source.jitter.y, 
            ion_wkfld_update_period=self.ion_wkfld_update_period, 
            drive_beam_update_period=self.drive_beam_update_period, 
            #wake_t_fields=wake_t_fields  # TODO: remove when driver evolution has been implemented.
        )
        
        inputs = [drive_beam_ramped, beam_filtered, trans_wake_config.plasma_density, self.Ez_fit_obj, self.rb_fit_obj, trans_wake_config.stage_length, trans_wake_config.time_step_mod]
        some_are_none = any(input is None for input in inputs)
        
        if some_are_none:
            none_indices = [i for i, x in enumerate(inputs) if x is None]
            print(none_indices)
            raise ValueError('At least one input is set to None.')

        # Add the driver offsets to the Wake-T r-coordinate
        # Changes to info_rho should only be done after the plasma ion bubble radius has been traced and extracted.
        rho = wake_t_evolution.initial.plasma.density.rho*-e
        info_rho = wake_t_evolution.initial.plasma.density.metadata
        rs_rho = info_rho.r + np.sqrt(driver_x_offset**2 + driver_y_offset**2)
        info_rho.r = rs_rho
        info_rho.imshow_extent[2] = rs_rho.min()
        info_rho.imshow_extent[3] = rs_rho.max()

        # Save the initial step with ramped beams in rotated coordinate system after upramp
        Ez_axis_wakeT = wake_t_evolution.initial.plasma.wakefield.onaxis.Ezs
        zs_Ez_wakeT = wake_t_evolution.initial.plasma.wakefield.onaxis.zs
        self.__save_initial_step(Ez0_axial=Ez_axis_wakeT, zs_Ez0=zs_Ez_wakeT, rho0=rho, metadata_rho0=info_rho, driver0=drive_beam_ramped, beam0=beam_filtered)
        
        # Perform main tracking
        beam, driver, evolution = transverse_wake_instability_particles(beam_filtered, copy.deepcopy(drive_beam_ramped), Ez_fit_obj=self.Ez_fit_obj, rb_fit_obj=self.rb_fit_obj, trans_wake_config=trans_wake_config)

        self.evolution = evolution

        ## NOTE: beam and driver cannot be changed after this line in order to test for continuity between ramps and stage.

        # Save the final step with ramped beams in rotated coordinate system before downramp
        if self.save_final_step:
            self.__save_final_step(Ez_axis_wakeT, zs_Ez_wakeT, rho, info_rho, driver, beam)
        else:
            self.final = None


        # ==========  Apply plasma density down ramp (magnify beta function) ==========
        if self.downramp is not None:
            beam_outgoing, driver_outgoing = self.track_downramp(copy.deepcopy(beam), copy.deepcopy(driver))

        else:  # Do the following if there are no downramp. 
            beam_outgoing = copy.deepcopy(beam)
            driver_outgoing = copy.deepcopy(driver)
            if self.ramp_beta_mag is not None:  # Do the following before rotating back to original frame.

                beam_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)
                driver_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)


        # ========== Rotate the coordinate system of the beams back to original ==========
        # Perform un-rotation after track_downramp(). Also adds drift to the drive beam.
        if self.parent is None:  # Ensures that the un-rotation is only performed by the main stage and not by its ramps.
            
            # Will only rotate the beam coordinate system if the driver source of the stage has angular jitter or angular offset
            driver_outgoing, beam_outgoing = self.undo_beam_coordinate_systems_rotation(driver_incoming, driver_outgoing, beam_outgoing)
        
        
        # ==========  Make animations ==========
        if self.probe_evol_period > 0 and self.make_animations:
            self.animate_sideview_x(tmpfolder)
            self.animate_sideview_y(tmpfolder)
            self.animate_phasespace_x(tmpfolder)
            self.animate_phasespace_y(tmpfolder)

            # Remove temporary files
            shutil.rmtree(tmpfolder)
        

        # ========== Bookkeeping ==========
        self.driver_to_beam_efficiency = (beam_outgoing.energy()-beam_incoming.energy())/driver_outgoing.energy() * beam_outgoing.abs_charge()/driver_outgoing.abs_charge()
        
        # Store outgoing beams for comparison between ramps and its parent. Stored inside the ramps.
        if self.test_beam_between_ramps:
            if self.parent is None:
                # Store beams for the main stage
                self.store_beams_between_ramps(driver_before_tracking=drive_beam_ramped,  # Drive beam after the upramp, before tracking
                                               beam_before_tracking=beam0,  # Main beam after the upramp, before tracking
                                               driver_outgoing=driver,  # Drive beam after tracking, before the downramp
                                               beam_outgoing=beam,  # Main beam after tracking, before the downramp
                                               driver_incoming=driver_incoming)  # The original drive beam before rotation and ramps
            else:
                # Store beams for the ramps
                self.store_beams_between_ramps(driver_before_tracking=drive_beam_ramped,  # A deepcopy of the incoming drive beam before tracking.
                                               beam_before_tracking=beam0,  # A deepcopy of the incoming main beam before tracking.
                                               driver_outgoing=driver_outgoing,  # Drive beam after tracking through the ramp (has not been un-rotated)
                                               beam_outgoing=beam_outgoing,  # Main beam after tracking through the ramp (has not been un-rotated)
                                               driver_incoming=None)  # Only the main stage needs to store the original drive beam

        # Copy meta data from input beam_outgoing (will be iterated by super)
        beam_outgoing.trackable_number = beam_incoming.trackable_number
        beam_outgoing.stage_number = beam_incoming.stage_number
        beam_outgoing.location = beam_incoming.location

        # Clear some of the attributes to reduce file size of pickled files
        self.trim_attr_reduce_pickle_size()

        # Return the beam (and optionally the driver)
        if self._return_tracked_driver:
            return super().track(beam_outgoing, savedepth, runnable, verbose), driver_outgoing
        else:
            return super().track(beam_outgoing, savedepth, runnable, verbose)
        

    # ==================================================
    def store_rb_Ez_2stage(self, wake_t_evolution, drive_beam, beam):
        """
        Traces the longitudinal electric field, bubble radius and store them 
        inside the stage.

    
        Parameters
        ----------
        wake_t_evolution : ...
            Contains the 2D plasma density and wakefields for the initial and 
            final time steps.

        drive_beam : ABEL ``Beam`` object
            Drive beam.

        beam : ABEL ``Beam`` object
            Main beam.
    
            
        Returns
        ----------
        None
        """

        # Read the Wake-T simulation data
        Ez_axis_wakeT = wake_t_evolution.initial.plasma.wakefield.onaxis.Ezs
        zs_Ez_wakeT = wake_t_evolution.initial.plasma.wakefield.onaxis.zs
        #rho = wake_t_evolution.initial.plasma.density.rho*-e
        plasma_num_density = wake_t_evolution.initial.plasma.density.rho/self.plasma_density
        info_rho = wake_t_evolution.initial.plasma.density.metadata
        zs_rho = info_rho.z
        
        # Cut out axial Ez over the region of interest
        Ez_roi, Ez_fit = self.Ez_shift_fit(Ez_axis_wakeT, zs_Ez_wakeT, beam, self.z_slices)
        
        # Extract the plasma bubble radius
        self.zs_bubble_radius_axial = zs_rho
        bubble_radius_wakeT = self.trace_bubble_radius_WakeT(plasma_num_density, info_rho.r, zs_rho, threshold=0.8)  # Extracts rb with driver coordinates shifted on axis.

        # Cut out bubble radius over the region of interest
        R_blowout = blowout_radius(self.plasma_density, drive_beam.peak_current())
        self.estm_R_blowout = R_blowout
        bubble_radius_roi, rb_fit = self.rb_shift_fit(bubble_radius_wakeT, zs_rho, beam, self.z_slices)

        if bubble_radius_wakeT.max() < 0.5 * R_blowout or bubble_radius_roi.any()==0:
            warnings.warn("The bubble radius may not have been correctly extracted.", UserWarning)

        import scipy.signal as signal
        idxs_bubble_peaks, _ = signal.find_peaks(bubble_radius_roi, height=None, width=1, prominence=0.1)
        if idxs_bubble_peaks.size > 0:
            warnings.warn("The bubble radius may not be smooth.", UserWarning)

        # Save quantities to the stage
        self.Ez_fit_obj = Ez_fit
        self.rb_fit_obj = rb_fit
        
        self.Ez_roi = Ez_roi
        #self.Ez_axial = Ez_axis_wakeT  # Moved to self.initial.plasma.wakefield.onaxis.Ezs
        #self.zs_Ez_axial = zs_Ez_wakeT  # Moved to self.initial.plasma.wakefield.onaxis.zs
        self.bubble_radius_roi = bubble_radius_roi
        self.bubble_radius_axial = bubble_radius_wakeT
        
        # Make plots for control if necessary
        #self.plot_Ez_rb_cut()


    # ==================================================
    # Save initial electric field, plasma and beam quantities
    def __save_initial_step(self, Ez0_axial, zs_Ez0, rho0, metadata_rho0, driver0, beam0):
        
        # ========== Save initial axial wakefield info ========== 
        self.initial.plasma.wakefield.onaxis.zs = zs_Ez0
        self.initial.plasma.wakefield.onaxis.Ezs = Ez0_axial

        # ========== Save plasma electron number density ========== 
        self.initial.plasma.density.rho = rho0/-SI.e
        self.initial.plasma.density.rs_rho = metadata_rho0.r
        self.initial.plasma.density.zs_rho = metadata_rho0.z
        self.initial.plasma.density.extent = metadata_rho0.imshow_extent  # array([z_min, z_max, x_min, x_max])

        # ========== Save initial drive beam and main beam ==========
        #self.initial.driver.instance = copy.deepcopy(driver0)
        #self.initial.beam.instance = copy.deepcopy(beam0)

        # ========== Calculate and save initial beam particle density ==========
        zs_beams = np.append(driver0.zs(), beam0.zs())
        xs_beams = np.append(driver0.xs(), beam0.xs())
        ys_beams = np.append(driver0.ys(), beam0.ys())
        w = np.append(driver0.weightings(), beam0.weightings())  # The weights for the macroparticles. Append in same order as zs_beams.
        nbins = int(np.sqrt(len(w)/2))

        # Create a 3D histogram
        hist, edges = np.histogramdd((zs_beams, xs_beams, ys_beams), bins=(nbins, nbins, nbins), weights=w)
        edges_z = edges[0]
        edges_x = edges[1]
        edges_y = edges[2]
        
        # Calculate volume of each bin
        bin_volumes = np.diff(edges_z) * np.diff(edges_x) * np.diff(edges_y)

        # Calculate particle density per unit volume
        particle_density = hist / bin_volumes

        # Sum along the y-axis to obtain a 2D projection onto the zx plane
        projection_zx = np.sum(particle_density, axis=2)  # TODO: More suitable to use the centre zx slice instead of projection?
        projection_zx = projection_zx.T
        extent_beams = np.array([edges_z[0], edges_z[-1], edges_x[0], edges_x[-1]])
        #TODO: projection_zy?

        self.initial.beam.density.extent = extent_beams  # array([z_min, z_max, x_min, x_max])
        self.initial.beam.density.rho = projection_zx
        
        # ========== Save initial beam currents ==========
        self.calculate_beam_current(beam0, driver0) # Saves to self.initial.beam.current.zs and self.initial.beam.current.Is.


    # ==================================================
    # Save final electric field, plasma and beam quantities
    def __save_final_step(self, Ez_axial, zs_Ez, rho, metadata_rho, driver, beam):
        
        # ========== Save final axial wakefield info ========== 
        self.final.plasma.wakefield.onaxis.zs = zs_Ez
        self.final.plasma.wakefield.onaxis.Ezs = Ez_axial

        # ========== Save plasma electron number density ========== 
        self.final.plasma.density.rho = rho/-SI.e
        self.final.plasma.density.rs_rho = metadata_rho.r
        self.final.plasma.density.zs_rho = metadata_rho.z
        self.final.plasma.density.extent = metadata_rho.imshow_extent  # array([z_min, z_max, x_min, x_max])

        # ========== Save final drive beam and main beam ==========
        #self.final.driver.instance = copy.deepcopy(driver)
        #self.final.beam.instance = copy.deepcopy(beam)

        # ========== Calculate and save final beam particle density ==========
        zs_beams = np.append(driver.zs(), beam.zs())
        xs_beams = np.append(driver.xs(), beam.xs())
        ys_beams = np.append(driver.ys(), beam.ys())
        w = np.append(driver.weightings(), beam.weightings())  # The weights for the macroparticles. Append in same order as zs_beams.
        nbins = int(np.sqrt(len(w)/2))

        # Create a 3D histogram
        hist, edges = np.histogramdd((zs_beams, xs_beams, ys_beams), bins=(nbins, nbins, nbins), weights=w)
        edges_z = edges[0]
        edges_x = edges[1]
        edges_y = edges[2]
        
        # Calculate volume of each bin
        bin_volumes = np.diff(edges_z) * np.diff(edges_x) * np.diff(edges_y)

        # Calculate particle density per unit volume
        particle_density = hist / bin_volumes

        # Sum along the y-axis to obtain a 2D projection onto the zx plane
        projection_zx = np.sum(particle_density, axis=2)
        projection_zx = projection_zx.T
        extent_beams = np.array([edges_z[0], edges_z[-1], edges_x[0], edges_x[-1]])
        #TODO: projection_zy?

        self.final.beam.density.extent = extent_beams  # array([z_min, z_max, x_min, x_max])
        self.final.beam.density.rho = projection_zx
        
        # ========== Save final beam currents ==========
        dz = 40*np.mean([driver.bunch_length(clean=True)/np.sqrt(len(driver)), beam.bunch_length(clean=True)/np.sqrt(len(beam))])
        num_sigmas = 6
        z_min = beam.z_offset() - num_sigmas * beam.bunch_length()
        z_max = driver.z_offset() + num_sigmas * driver.bunch_length()
        tbins = np.arange(z_min, z_max, dz)/SI.c
        
        Is, ts = (driver + beam).current_profile(bins=tbins)
        self.final.beam.current.zs = ts*SI.c
        self.final.beam.current.Is = Is       


    # ==================================================
    # May not be needed, as saving evolution to this stage is trivial.
    #def __extract_evolution(self, evolution):
    #    self.evolution = evolution


    # ==================================================
    def stage2ramp(self, ramp_plasma_density=None, ramp_length=None, probe_evol_period=1, make_animations=False):
        """
        Used for copying a predefined stage's settings and configurations to set
        up flat ramps. Overloads the parent class' method.
    
        Parameters
        ----------
        ramp_plasma_density : [m^-3] float, optional
            Plasma density for the ramp.

        ramp_length : [m] float, optional
            Length of the ramp.

        probe_evol_period : int, optional
            Set to larger than 0 to determine the probing period for beam 
            evolution diagnostics. This is given in units of time steps, so that
            e.g. ``probe_evol_period=3`` will probe the beam evolution every 3rd
            time step. Default value: 1.

        make_animations : bool, optional
            Flag for making animations.
    
            
        Returns
        ----------
        stage_copy : ``Stage`` object
            A modified deep copy of the original stage.
        """

        stage_copy = super().stage2ramp(ramp_plasma_density, ramp_length)

        # Additional configurations 
        stage_copy.probe_evol_period = probe_evol_period
        stage_copy.make_animations = make_animations

        return stage_copy
    

    # ==================================================
    # Filter out particles that collide into bubble
    def bubble_filter(self, beam, sort_zs=True):
        xs = beam.xs()
        ys = beam.ys()
        zs = beam.zs()
        pxs = beam.pxs()
        pys = beam.pys()
        pzs = beam.pzs()
        weights = beam.weightings()
    
        # Check if the arrays are sorted based on zs
        if sort_zs:
            # Sort the arrays based on zs.
            indices = np.argsort(zs)
            zs_sorted = zs[indices]
            xs_sorted = xs[indices]
            ys_sorted = ys[indices]
            pxs_sorted = pxs[indices]
            pys_sorted = pys[indices]
            pzs_sorted = pzs[indices]
            weights_sorted = weights[indices]
        else:
            zs_sorted = zs
            xs_sorted = xs
            ys_sorted = ys
            pxs_sorted = pxs
            pys_sorted = pys
            pzs_sorted = pzs
            weights_sorted = weights

        # Calculate rb based on interpolation of rb vs z
        rb_fit_obj = self.rb_fit_obj
        bubble_radius = rb_fit_obj(zs_sorted)

        # Apply the filter
        bool_indices = (np.sqrt(xs_sorted**2 + ys_sorted**2) - bubble_radius <= 0)
        zs_sorted = zs_sorted[bool_indices]
        xs_sorted = xs_sorted[bool_indices]
        ys_sorted = ys_sorted[bool_indices]
        pxs_sorted = pxs_sorted[bool_indices]
        pys_sorted = pys_sorted[bool_indices]
        pzs_sorted = pzs_sorted[bool_indices]
        weights_sorted = weights_sorted[bool_indices]

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
                                 particle_mass=beam.particle_mass)
        return beam_out

    
    # ==================================================
    def Ez_shift_fit(self, Ez, zs_Ez, beam, z_slices=None):
        """
        Cuts out the longitudinal axial E-field Ez over the beam region and 
        makes a fit using the z-coordinates for the region.

        Parameters
        ----------
        Ez : [V/m] 1D float ndarray
            Axial longitudinal E-field.
            
        zs_Ez : [m] 1D float ndarray
            z-coordinates for ``Ez``. Monotonically increasing from first to 
            last element.

        beam : ABEL ``Beam`` object
            
        z_slices : [m] 1D float ndarray, optional
            Co-moving coordinates of the beam slices.

            
        Returns
        ----------
        Ez_fit(z_slices) : [V/m] 1D float ndarray
            Axial Ez for the region of interest shifted to the location of the 
            beam.

        Ez_fit : [V/m] 1D interpolation object 
            Interpolated axial longitudinal Ez from beam head to tail.
        """

        from scipy.interpolate import interp1d
        
        zs = beam.zs()

        if z_slices is None:
            _, z_slices = self.longitudinal_number_distribution(beam=beam)
        
        # Start index (head of the beam) of extraction interval.
        head_idx, _ = find_closest_value_in_arr(arr=zs_Ez, val=zs.max())
        
        # End index (tail of the beam) of extraction interval.
        tail_idx, _ = find_closest_value_in_arr(arr=zs_Ez, val=zs.min())
        
        # Cut Ez and zs_Ez
        Ez_cut = Ez[tail_idx:head_idx+1]
        zs_cut = zs_Ez[tail_idx:head_idx+1]
        
        Ez_fit = interp1d(zs_cut, Ez_cut, kind='slinear', bounds_error=False, fill_value='extrapolate' )
        
        # Calculate sum of squared errors (sse)
        sse_Ez = np.sum((Ez_cut - Ez_fit(zs_cut))**2)
        
        if sse_Ez/np.mean(Ez_cut) > 0.05:
            warnings.warn('The longitudinal E-field may not have been accurately fitted.\n', UserWarning)
        
        return Ez_fit(z_slices), Ez_fit


    # ==================================================
    def trace_bubble_radius(self, plasma_num_density, plasma_tr_coord, plasma_z_coord, driver_offset, threshold=0.8):
        """
        - For extracting the plasma ion bubble radius by finding the coordinates
        in which the plasma number density goes from zero to a threshold value.
        - The symmetry axis is determined using the transverse offset of the 
        drive beam.
        - z is the propagation direction pointing to the right.
        
        Parameters
        ----------
        plasma_num_density : [n0] 2D float ndarray
            Plasma number density in units of initial number density n0. Need to
            be oriented with propagation direction pointing to the right and 
            positive offset pointing upwards.
            
        plasma_tr_coord : [m] 1D float ndarray 
            Transverse coordinate of ``plasma_num_density``. Needs to be 
            strictly growing from start to end.

        plasma_z_coord : [m] 1D float ndarray 
            Longitudinal coordinate of ``plasma_num_density``. Needs to be
            strictly growing from start to end.

        driver_offset : [m] float
            Mean transverse offset of the drive beam.
            
        threshold : float, optional
            Defines a threshold for the plasma density to determine
            ``bubble_radius``.

            
        Returns
        ----------
        bubble_radius : [m] 1D float ndarray 
            Plasma bubble radius over the simulation box, measured from the 
            drive beam axis.
        """

        import scipy.signal as signal
        from scipy.interpolate import interp1d
        
        # Check if plasma_tr_coord is strictly growing from start to end
        if not np.all(np.diff(plasma_tr_coord) > 0):
            raise ValueError('plasma_tr_coord needs to be strictly increasing from start to end.')
        
        # Find the value in plasma_tr_coord closest to driver_offset and corresponding index
        idx_middle, _ = find_closest_value_in_arr(plasma_tr_coord, driver_offset)

        rows, cols = np.shape(plasma_num_density)
        bubble_radius = np.zeros(cols)
        slopes = np.diff(plasma_num_density)

        for i in range(0,cols):  # Loop through all transverse slices.
            
            # Extract a transverse slice
            slice = plasma_num_density[:,i]
            
            idxs_peaks, _ = signal.find_peaks(slice, height=1.2, width=1.5, prominence=None)  # Find all relevant peaks in the slice.
            
            if idxs_peaks.size == 0:
                idxs_peaks, _ = signal.find_peaks(slice, height=1.0, width=1, prominence=None)  # Slightly reduce requirement if no peaks found.
                
            idxs_peaks_above = idxs_peaks[idxs_peaks > idx_middle]  # Get the slice peak indices located above middle.
            idxs_peaks_below = idxs_peaks[idxs_peaks < idx_middle]  # Get the slice peak indices located below middle.

            # Check if there are peaks on both sides
            if idxs_peaks_above.size > 0 and idxs_peaks_below.size > 0:
                
                # Get the indices for the maximum negative and positive slopes of plasma_num_density
                slopes = np.diff(slice)
                slopes = np.insert(arr=slopes, obj=0, values=0.0)
                idx_max_pos_slope = np.argmax(slopes)
                idx_max_neg_slope = np.argmin(slopes)

                _, idx_above = find_closest_value_in_arr(idxs_peaks_above, idx_max_pos_slope)  # Get the slice peak above middle closest to max_pos_slope.
                _, idx_below = find_closest_value_in_arr(idxs_peaks_below, idx_max_neg_slope)  # Get the slice peak below middle closest to max_neg_slope.
                
                # Get the plasma_num_density slice elements between the peaks. Set the rest to nan.
                valley = copy.deepcopy(slice)
                valley[:idx_below] = np.nan
                valley[idx_above+1:] = np.nan
                
                # Check that the "valley" of the slice is not too shallow. If too shallow, bubble_radius[i] = 0.
                if len(valley) > 0 and np.nanmin(valley) < threshold and np.nanmax(valley) >= threshold:
                    
                    # Find the indices of all the elements in valley >= threshold.
                    idxs_slice_above_thres = np.where(valley >= threshold)[0]
                    
                    # Get the valley indices above threshold located above middle
                    idxs_valley_above_middle = idxs_slice_above_thres[idxs_slice_above_thres > idx_middle]

                    # Find element in valley closest to threshold
                    idx_valley_above_middle_closest2thres, _ = find_closest_value_in_arr(valley[idxs_valley_above_middle], threshold)
                    idx = idxs_valley_above_middle[idx_valley_above_middle_closest2thres]
                    
                    bubble_radius[i] = np.abs(plasma_tr_coord[idx] - driver_offset)

        if self.estm_R_blowout is None:
            drive_beam = self.driver_source.track()
            R_blowout = blowout_radius(self.plasma_density, drive_beam.peak_current())
        else:
            R_blowout = self.estm_R_blowout

        # Apply corrections until there are no more elements in bubble_radius > R_blowout
        n_tries = 0
        while bubble_radius.max() > R_blowout and n_tries < 10:
        
            mask = self.mask_bubble_radius_spikes(bubble_radius, plasma_z_coord)
            num_rm_elements = np.sum(~mask)  # Number of elements to remove

            if num_rm_elements > 0.33*len(bubble_radius):
                warnings.warn('bubble_radius may contain too many abnormal peaks. Removing these many elements may result in a bad interpolation later.')
            if num_rm_elements == len(bubble_radius):
                raise ValueError('Cannot remove all elements from bubble_radius.')

            # Use interpolation to replace the deleted points
            f_interp = interp1d(plasma_z_coord[mask], bubble_radius[mask], kind='slinear', fill_value="extrapolate") 
            bubble_radius = f_interp(plasma_z_coord)  # Cleaned bubble radius

            n_tries += 1
        
        return bubble_radius


    # ==================================================
    def mask_bubble_radius_spikes(self, bubble_radius, zs_bubble_radius, make_plot=False):
        """
        For identifying large abnormal spikes in ``bubble_radius``.

        Parameters
        ----------
        bubble_radius : [m] 1D float ndarray
            Plasma ion bubble radius along the whole simulation domain.
            
        zs_bubble_radius_axial : [m] 1D float ndarray, optional
            Co-moving coordinates of the bubble radius. Same length as 
            ``bubble_radius``

            
        Returns
        ----------
        mask : [m] 1D bool ndarray
            An array of the same length as ``bubble_radius`` where the elements 
            corresponding to abnormal spikes are ``False``.
        """

        import scipy.signal as signal

        # Mask based on the first derivative of bubble_radius
        drb_dz = np.gradient(bubble_radius, zs_bubble_radius)  # First derivative

        lower_p = np.percentile(np.abs(drb_dz), 10)  # Lower percentile
        upper_p = np.percentile(np.abs(drb_dz), 90)  # Upper percentile
        ip_range = upper_p - lower_p  # Interpercentile range
        lower_bound = lower_p - 1.5 * ip_range
        upper_bound = upper_p + 1.5 * ip_range
        d1_outliers = np.where((np.abs(drb_dz) < lower_bound) | (np.abs(drb_dz) > upper_bound))[0]

        d1_mask = np.ones_like(bubble_radius, dtype=bool)
        d1_mask[d1_outliers] = False  # Set the outlier indices to False.

        # Mask based on curvature
        d2rb_dz2 = np.gradient(drb_dz, zs_bubble_radius)  # Second derivative
        curvature = np.abs(d2rb_dz2)/(1+drb_dz**2)**(3/2)

        lower_p = np.percentile(np.abs(curvature), 10)  # Lower percentile
        upper_p = np.percentile(np.abs(curvature), 90)  # Upper percentile
        ip_range = upper_p - lower_p  # Interpercentile range
        lower_bound = lower_p - 1.5 * ip_range
        upper_bound = upper_p + 1.5 * ip_range
        curvature_outliers = np.where((np.abs(curvature) < lower_bound) | (np.abs(curvature) > upper_bound))[0]

        curvature_mask = np.ones_like(bubble_radius, dtype=bool)
        curvature_mask[curvature_outliers] = False

        # Mask based on peaks
        if self.estm_R_blowout is None:
            drive_beam = self.driver_source.track()
            R_blowout = blowout_radius(self.plasma_density, drive_beam.peak_current())
        else:
            R_blowout = self.estm_R_blowout
        idxs_peaks, _ = signal.find_peaks(bubble_radius, height=0.9*R_blowout, width=1.5, prominence=None)
        peak_mask = np.ones_like(bubble_radius, dtype=bool)
        peak_mask[idxs_peaks] = False

        if make_plot:
            plt.figure()
            plt.plot(bubble_radius*1e6, '-x')
            plt.plot(idxs_peaks, bubble_radius[idxs_peaks]*1e6, 'x', label='Peaks filter')
            plt.plot(d1_outliers, bubble_radius[d1_outliers]*1e6, 'o', label='First derivative filter')
            plt.plot(curvature_outliers, bubble_radius[curvature_outliers]*1e6, 'o', label='Curvature filter')
            plt.xlabel('Index')
            plt.ylabel(r'Bubble radius [$\mathrm{\mu}$m]')
            plt.title('Bubble radius and elements to be removed')
            plt.legend()

        # Final mask
        mask = np.logical_and(np.logical_and(d1_mask, peak_mask), curvature_mask)

        return mask


    # ==================================================
    def trace_bubble_radius_WakeT(self, plasma_num_density, plasma_tr_coord, plasma_z_coord, threshold=0.8):
        """
        The plasma wake calculated by Wake-T is always centered around r = 0.0, 
        so that driver_offset=0.0 are used as inputs in trace_bubble_radius().
        """
        bubble_radius = self.trace_bubble_radius(plasma_num_density=plasma_num_density, plasma_tr_coord=plasma_tr_coord, plasma_z_coord=plasma_z_coord, driver_offset=0.0, threshold=threshold)

        return bubble_radius

    
    # ==================================================
    def rb_shift_fit(self, rb, zs_rb, beam, z_slices=None):
        """
        Cuts out the bubble radius over the beam region and makes a fit using 
        the z-coordinates for the region.

        Parameters
        ----------
        rb : [m] 1D float ndarray
            Plasma ion bubble radius along the whole simulation domain.
            
        zs_rb : [m] 1D float ndarray
            z-coordinates for ``rb``. Monotonically increasing from first to 
            last element.

        beam : ABEL ``Beam`` object
            
        z_slices : [m] 1D float ndarray, optional
            Co-moving coordinates of the beam slices.

            
        Returns
        ----------
        rb_roi : [m] 1D float ndarray
            Plasma ion bubble radius for the region of interest shifted to the 
            location of the beam.

        rb_fit : [m] 1D interpolation object 
            Interpolated plasma ion bubble radius from beam head to tail.
        """

        from scipy.interpolate import interp1d
        
        zs = beam.zs()

        if z_slices is None:
            _, z_slices = self.longitudinal_number_distribution(beam=beam)
        
        # Start index (head of the beam) of extraction interval.
        head_idx, _ = find_closest_value_in_arr(arr=zs_rb, val=zs.max())
        
        # End index (tail of the beam) of extraction interval.
        tail_idx, _ = find_closest_value_in_arr(arr=zs_rb, val=zs.min())
        
        # Cut rb and zs_rb
        rb_cut = rb[tail_idx:head_idx+1]
        zs_cut = zs_rb[tail_idx:head_idx+1]

        rb_fit = interp1d(zs_cut, rb_cut, kind='slinear', bounds_error=False, fill_value='extrapolate')
        #rb_fit = interp1d(zs_cut, rb_cut, kind='quadratic', bounds_error=False, fill_value='extrapolate')
        
        # Calculate sum of squared errors (sse)
        sse_rb = np.sum((rb_cut - rb_fit(zs_cut))**2)
        
        if sse_rb/np.mean(rb_cut) > 0.05:
            warnings.warn('The plasma ion bubble radius may not have been accurately fitted.\n', UserWarning)

        rb_roi = rb_fit(z_slices)

        # Check the smoothness of the bubble radius in the region of interest
        drb_dz = np.gradient(rb_roi, z_slices)  # Compute first order derivative
        smoothness_threshold = 2.0  # Define a threshold for smoothness
        if np.all(np.abs(np.diff(drb_dz)) > smoothness_threshold):
            
            # Do a deeper smoothness test using the sorted beam macro particle zs
            zs = beam.zs()
            indices = np.argsort(zs)
            zs_sorted = zs[indices]
            rb_prtcl = rb_fit(zs_sorted)
            drb_dz = np.gradient(rb_prtcl, zs_sorted)  # Compute first order derivative
            d2rb_dz2 = np.gradient(rb_prtcl, zs_sorted)  # Compute second order derivative

            smoothness_metric = np.max(np.abs(d2rb_dz2))
            
            if smoothness_metric > smoothness_threshold:
                warnings.warn('The plasma ion bubble radius may not have been extracted correctly.\n', UserWarning)
        
        return rb_roi, rb_fit    
        

    # ==================================================
    # Determine the number of beam slices based on the Freedman–Diaconis rule
    def FD_rule_num_slice(self, zs=None):
        if zs is None:
            zs = self.initial.beam.instance.zs()
        q3, q1 = np.percentile(zs, [75 ,25])
        iqr = q3 - q1  # Calculate the interquartile range
        beam_slice_thickness = 2*iqr*len(zs)**(-1/3)
        num_uniform_beam_slice = int(np.round((zs.max()-zs.min()) / beam_slice_thickness))
        return num_uniform_beam_slice


    # ==================================================
    # Return the longitudinal number distribution using the beam particles' z-coordinates.
    def longitudinal_number_distribution(self, beam, bin_number=None, make_plot=False):

        zs = beam.zs()
        if bin_number is None:
            bin_number = self.FD_rule_num_slice(zs)

        weights = beam.weightings()  # The weight of each macroparticle.
        num_profile, edges = np.histogram(zs, weights=weights, bins=bin_number)  # Compute the histogram of z using bin_number bins.
        
        self.main_slices_edges = edges
        
        z_ctrs = (edges[0:-1] + edges[1:])/2  # Centres of the bins (zs).
        
        if make_plot is True:
            plt.figure(figsize=(10, 5))
            plt.plot(z_ctrs*1e6, num_profile, 'g')
            for edge in edges:
                plt.axvline(x=edge*1e6, color=(0.5, 0.5, 0.5), linestyle='-', alpha=0.5)
            plt.xlabel(r'$\xi$ [$\mathrm{\mu}$m]')
            plt.ylabel(r'Number profile $N(\xi)$')

        return num_profile, z_ctrs

    
    # ==================================================
    def matched_beta_function(self, energy):
        return beta_matched(self.plasma_density, energy) * self.ramp_beta_mag
    
    
    # ==================================================
    # Calculate the normalised amplitude (Lambda).
    def calc_norm_amp(self, particle_offsets, particle_angles):

        beam_size = np.std(particle_offsets)
        beam_size_angle = np.std(particle_angles)

        return np.sum((particle_offsets/beam_size)**2 + (particle_angles/beam_size_angle)**2)

    
    # ==================================================
    def energy_efficiency(self):
        return None # TODO

    
    # ==================================================
    def energy_usage(self):
        return None # TODO
        
    
    # ==================================================
    #def __get_initial_driver(self, resample=False):
    #    if resample or self.driver_initial is None:
    #        self.driver_initial = self.driver_source.track()
    #    return self.driver_initial

    
    # ==================================================
    def __get_plasma_density(self, resample=False):
        if resample or self.__n is None:
            self.__n = self.plasma_density * np.random.normal(loc = 1, scale = self.reljitter.plasma_density)
        return self.__n
        

    # ==================================================
    def trim_attr_reduce_pickle_size(self):
        "Clear attributes to reduce space in the pickled file."
        if self.upramp is not None:
            self.upramp.drive_beam = None
            self.upramp.initial = None
            self.upramp.final = None
        if self.downramp is not None:
            self.downramp.drive_beam = None
            self.downramp.initial = None
            self.downramp.final = None

        self.drive_beam = None
        #self.upramp = None
        #self.downramp = None
        #self.final = None


    # ==================================================
    # Overloads the plot_wakefield method in the Stage class.
    def plot_wakefield(self, saveToFile=None, includeWakeRadius=True):
        
        # Get wakefield
        #Ezs = self.Ez_axial
        #zs_Ez = self.zs_Ez_axial
        Ezs = self.initial.plasma.wakefield.onaxis.Ezs
        zs_Ez = self.initial.plasma.wakefield.onaxis.zs
        zs_rho =  self.zs_bubble_radius_axial
        bubble_radius = self.bubble_radius_axial
        
        # get current profile
        has_final_step = self.final is not None
        if has_final_step:
            Is = self.final.beam.current.Is
            zs = self.final.beam.current.zs
            title = 'Final step'
        else:
            Is = self.initial.beam.current.Is
            zs = self.initial.beam.current.zs
            title = 'Initial step'
        
        # plot it
        fig, axs = plt.subplots(1, 2+int(includeWakeRadius))
        fig.set_figwidth(CONFIG.plot_fullwidth_default*(2+int(includeWakeRadius))/3)
        fig.set_figheight(4)
        col0 = "xkcd:light gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        af = 0.1
        zlims = [min(zs_Ez)*1e6, max(zs_Ez)*1e6]
        
        axs[0].plot(zs_Ez*1e6, np.zeros(zs_Ez.shape), '-', color=col0)
        if self.nom_energy_gain is not None:
            axs[0].plot(zs_Ez*1e6, -self.nom_energy_gain/self.get_length()*np.ones(zs_Ez.shape)/1e9, ':', color=col0)
        axs[0].plot(zs_Ez*1e6, Ezs/1e9, '-', color=col1)
        axs[0].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
        axs[0].set_ylabel('Longitudinal electric field [GV/m]')
        axs[0].set_xlim(zlims)
        axs[0].set_ylim(bottom=1.05*min(Ezs)/1e9, top=1.3*max(Ezs)/1e9)
        
        axs[1].fill(np.concatenate((zs, np.flip(zs)))*1e6, np.concatenate((-Is, np.zeros(Is.shape)))/1e3, color=col1, alpha=af)
        axs[1].plot(zs*1e6, -Is/1e3, '-', color=col1)
        axs[1].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
        axs[1].set_ylabel('Beam current [kA]')
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=0, top=1.2*max(-Is)/1e3)
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=1.2*min(-Is)/1e3, top=1.2*max(-Is)/1e3)
        
        if includeWakeRadius:
            axs[2].fill(np.concatenate((zs_rho, np.flip(zs_rho)))*1e6, np.concatenate((bubble_radius, np.ones(zs_rho.shape)))*1e6, color=col2, alpha=af)
            axs[2].plot(zs_rho*1e6, bubble_radius*1e6, '-', color=col2)
            axs[2].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            axs[2].set_ylabel(r'Plasma-wake radius [$\mathrm{\mu}$m]')
            axs[2].set_xlim(zlims)
            axs[2].set_ylim(bottom=0, top=max(bubble_radius*1.2)*1e6)

        fig.suptitle(title)
        
        # save to file
        if saveToFile is not None:
            plt.savefig(saveToFile, format="pdf", bbox_inches="tight")


    # ==================================================
    # Overloads the plot_wake method in the Stage class.
    def plot_wake(self, show_Ez=True, trace_rb=False, savefig=None, aspect='auto'):
        
        from matplotlib.colors import LogNorm
        
        # Extract density if not already existing
        assert hasattr(self.initial.plasma.density, 'rho'), 'No wake'
        assert hasattr(self.initial.plasma.wakefield.onaxis, 'Ezs'), 'No wakefield'
        
        # Make figures
        has_final_step = self.final is not None
        num_plots = 1 + int(has_final_step)
        fig, ax = plt.subplots(num_plots,1)
        fig.set_figwidth(CONFIG.plot_width_default*0.7)
        fig.set_figheight(CONFIG.plot_width_default*0.5*num_plots)

        # Cycle through initial and final step
        for i in range(num_plots):
            if not has_final_step:
                ax1 = ax
            else:
                ax1 = ax[i]

            # Extract initial or final
            if i==0:
                data_struct = self.initial
                title = 'Initial step'
            elif i==1:
                data_struct = self.final
                title = 'Final step'

            # Get data
            extent = data_struct.plasma.density.extent
            zs0 = data_struct.plasma.wakefield.onaxis.zs
            Ezs0 = data_struct.plasma.wakefield.onaxis.Ezs
            rho0_plasma = data_struct.plasma.density.rho
            rho0_beam = data_struct.beam.density.rho
            
            axpos = ax1.get_position()
            pad_fraction = 0.15  # Fraction of the figure width to use as padding between the ax and colorbar
            cbar_width_fraction = 0.03  # Fraction of the figure width for the colorbar width
    
            # Create colorbar axes based on the relative position and size
            cax1 = fig.add_axes([axpos.x1 + pad_fraction, axpos.y0, cbar_width_fraction, axpos.height])
            cax2 = fig.add_axes([axpos.x1 + pad_fraction + cbar_width_fraction, axpos.y0, cbar_width_fraction, axpos.height])
            clims = np.array([1e-2, 1e3])*self.plasma_density
            
            # Plot plasma electrons
            plasma_plot = ax1.imshow(rho0_plasma/1e6, extent=extent*1e6, norm=LogNorm(), origin='lower', cmap='Blues', alpha=np.array(rho0_plasma>clims.min()*2, dtype=float), aspect=aspect)
            cb = plt.colorbar(plasma_plot, cax=cax1)
            plasma_plot.set_clim(clims/1e6)
            cb.ax.tick_params(axis='y',which='both', direction='in')
            cb.set_ticklabels([])
            
            # Plot beam electrons
            charge_density_plot0 = ax1.imshow(rho0_beam/1e6, extent=data_struct.beam.density.extent*1e6, norm=LogNorm(), origin='lower', cmap=CONFIG.default_cmap, alpha=np.array(rho0_beam>clims.min()*2, dtype=float), aspect=aspect)
            cb2 = plt.colorbar(charge_density_plot0, cax = cax2)
            cb2.set_label(label=r'Electron density ' + r'[$\mathrm{cm^{-3}}$]',size=10)
            cb2.ax.tick_params(axis='y',which='both', direction='in')
            charge_density_plot0.set_clim(clims/1e6)

            # Plot traced bubble radius
            if trace_rb:
                ax1.plot(self.zs_bubble_radius_axial*1e6, self.bubble_radius_axial*1e6, color='red', alpha=0.4)

            # Plot on-axis wakefield
            if show_Ez:
                ax2 = ax1.twinx()
                ax2.plot(zs0*1e6, Ezs0/1e9, color='black')
                ax2.set_ylabel(r'$E_{z}$' ' [GV/m]')
                #zlims = [min(zs0)*1e6, max(zs0)*1e6]
                #ax2.set_xlim(zlims)
                #Ezmax = 0.8*wave_breaking_field(self.plasma_density)
                #ax2.set_ylim(bottom=-Ezmax/1e9, top=Ezmax/1e9)

            plasma_plot.set_extent(extent*1e6)  # Ensure that the extent is the same as data_struct.plasma.density.extent
    
            # Set labels
            if i==(num_plots-1):
                ax1.set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            ax1.set_ylabel(r'$x$ [$\mathrm{\mu}$m]')
            ax1.set_title(title)
            
            ax1.grid(False)
            
        # Save the figure
        if savefig is not None:
            fig.savefig(str(savefig), bbox_inches='tight', dpi=1000)
        
        return 


    # ==================================================
    def imshow_plot(self, data, axes=None, extent=None, vmin=None, vmax=None, colmap='seismic', xlab=r'$\xi$ [$\mathrm{\mu}$m]', ylab=r'$x$ [$\mathrm{\mu}$m]', clab='', gridOn=False, origin='lower', interpolation=None, aspect='auto', log_cax=False, reduce_cax_pad=False):
        
        from matplotlib.colors import LogNorm
        
        if axes is None:
            fig = plt.figure()  # an empty figure with an axes
            ax = fig.add_axes([.15, .15, .75, .75])
            cbar_ax = fig.add_axes([.85, .15, .03, .75])
        else:
            #ax = axes[0]  # TODO: adjust colourbar axes
            #cbar_ax = axes[1]            
            ax = axes
            cbar_ax = None

        if reduce_cax_pad is True:
            # Create an axis on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            from mpl_toolkits.axes_grid1 import make_axes_locatable  # For manipulating colourbars
            divider = make_axes_locatable(ax)
            cbar_ax = divider.append_axes("right", size="5%", pad=0.05)

        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()

        # Make a 2D plot
        if log_cax is True:
            p = ax.imshow(data, extent=extent, cmap=plt.get_cmap(colmap), origin=origin, aspect=aspect, interpolation=interpolation, norm=LogNorm(vmin+1, vmax))
        else:
            p = ax.imshow(data, extent=extent, vmin=vmin, vmax=vmax, cmap=plt.get_cmap(colmap), origin=origin, aspect=aspect, interpolation=interpolation)

        # Add a grid
        if gridOn == True:
            ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)

        # Add a colourbar
        cbar = plt.colorbar(p, ax=ax, cax=cbar_ax)
        cbar.set_label(clab)

        # Set the tick formatter to use power notation
        #import matplotlib.ticker as ticker
        #cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        #cbar.ax.tick_params(axis='y', which='major', pad=10)

        #import matplotlib.ticker as ticker
        #fmt = ticker.ScalarFormatter(useMathText=True)
        #fmt.set_powerlimits((-3, 19))
        #cbar.ax.yaxis.set_major_formatter(fmt)

        # Customize the colorbar tick locator and formatter
        #from matplotlib.ticker import ScalarFormatter
        #cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))  # Set the number of tick intervals
        #cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # Use scientific notation

        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)

    
    # ==================================================
    def scatter_diags(self, beam, n_th_particle=1):
        '''
        n_th_particle:  Use this to reduce the amount of plotted particles by 
        only plotting every n_th_particle particle.
        '''
        
        from matplotlib.colors import LinearSegmentedColormap  # For customising colour maps
        
        # Define the color map and boundaries
        colors = ['black', 'red', 'orange', 'yellow']
        bounds = [0, 0.2, 0.4, 0.8, 1]
        cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=256)

        # Macroparticles data
        zs = beam.zs()
        xs = beam.xs()
        xps = beam.xps()
        ys = beam.ys()
        yps = beam.yps()
        Es = beam.Es()
        weights = beam.weightings()

        # Labels for plots
        zlab = r'$z$ [$\mathrm{\mu}$m]'
        xilab = r'$\xi$ [$\mathrm{\mu}$m]'
        xlab = r'$x$ [$\mathrm{\mu}$m]'
        ylab = r'$y$ [$\mathrm{\mu}$m]'
        xps_lab = r"$x'$ [mrad]"
        yps_lab = r"$y'$ [mrad]"
        energ_lab = r'$\mathcal{E}$ [GeV]'
        
        # Set up a figure with axes
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(5*3, 4*3))
        plt.tight_layout(pad=6.0)  # Sets padding between the figure edge and the edges of subplots, as a fraction of the font size.
        fig.subplots_adjust(top=0.85)  # By setting top=..., you are specifying that the top boundary of the subplots should be at ...% of the figure’s height.
        
        # 2D z-x distribution
        ax = axs[0][0]
        p = ax.scatter(zs[::n_th_particle]*1e6, xs[::n_th_particle]*1e6, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xlab)
            
        # 2D z-x' distribution
        ax = axs[0][1]
        ax.scatter(zs[::n_th_particle]*1e6, xps[::n_th_particle]*1e3, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xps_lab)
            
        # 2D x-x' distribution
        ax = axs[0][2]
        ax.scatter(xs[::n_th_particle]*1e6, xps[::n_th_particle]*1e3, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xlab)
        ax.set_ylabel(xps_lab)
            
        # 2D z-y distribution
        ax = axs[1][0]
        ax.scatter(zs[::n_th_particle]*1e6, ys[::n_th_particle]*1e6, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xilab)
        ax.set_ylabel(ylab)
            
        # 2D z-y' distribution
        ax = axs[1][1]
        ax.scatter(zs[::n_th_particle]*1e6, yps[::n_th_particle]*1e3, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xilab)
        ax.set_ylabel(yps_lab)
            
        # 2D y-y' distribution
        ax = axs[1][2]
        ax.scatter(ys[::n_th_particle]*1e6, yps[::n_th_particle]*1e3, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(ylab)
        ax.set_ylabel(yps_lab)
            
        # 2D x-y distribution
        ax = axs[2][0]
        ax.scatter(xs[::n_th_particle]*1e6, ys[::n_th_particle]*1e6, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        
        # 2D x'-y' distribution
        ax = axs[2][1]
        ax.scatter(xps[::n_th_particle]*1e3, yps[::n_th_particle]*1e3, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xps_lab)
        ax.set_ylabel(yps_lab)

        # Energy distribution
        #ax = axs[2][1]
        #dN_dE, rel_energ = beam.rel_energy_spectrum()
        #dN_dE = dN_dE/-e
        #ax.fill_between(rel_energ*100, y1=dN_dE, y2=0, color='b', alpha=0.3)
        #ax.plot(rel_energ*100, dN_dE, color='b', alpha=0.3, label='Relative energy density')
        #ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        #ax.set_xlabel(r'$\mathcal{E}/\langle\mathcal{E}\rangle-1$ [%]')
        #ax.set_ylabel('Relative energy density')
        ## Add text to the plot
        #ax.text(0.05, 0.95, r'$\sigma_\mathcal{E}/\langle\mathcal{E}\rangle=$' f'{format(beam.rel_energy_spread()*100, ".2f")}' '%', fontsize=12, color='black', ha='left', va='top', transform=ax.transAxes)
        
        # 2D z-energy distribution
        ax = axs[2][2]
        ax.scatter(zs[::n_th_particle]*1e6, Es[::n_th_particle]/1e9, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xilab)
        ax.set_ylabel(energ_lab)
            
        # Set label and other properties for the colorbar
        fig.suptitle(r'$\Delta s=$' f'{format(beam.location, ".2f")}' ' m')
        cbar_ax = fig.add_axes([0.15, 0.91, 0.7, 0.02])   # The four values in the list correspond to the left, bottom, width, and height of the new axes, respectively.
        fig.colorbar(p, cax=cbar_ax, orientation='horizontal', label=energ_lab)


    # ==================================================
    def plot_Ez_rb_cut(self):

        z_slices = self.z_slices
        main_num_profile = self.main_num_profile
        zs_Ez = self.initial.plasma.wakefield.onaxis.zs
        Ez = self.initial.plasma.wakefield.onaxis.Ezs
        zs_rho = self.zs_bubble_radius_axial
        Ez_cut = self.Ez_roi
        bubble_radius = self.bubble_radius_axial
        bubble_radius_cut = self.bubble_radius_roi
        driver_num_profile = self.driver_num_profile
        driver_z_slices = self.driver_z_slices

        zlab=r'$z$ [$\mathrm{\mu}$m]'
        
        # Set up a figure with axes
        fig_wakeT_cut, axs_wakeT_cut = plt.subplots(nrows=1, ncols=2, layout='constrained', figsize=(10, 4))
        
        # Fill the axes with plots
        axs_wakeT_cut[0].fill_between(x=z_slices*1e6, y1=main_num_profile, y2=0, color='g', alpha=0.3)
        axs_wakeT_cut[0].plot(z_slices*1e6, main_num_profile, 'g', label='Number profile')
        axs_wakeT_cut[0].fill_between(x=driver_z_slices*1e6, y1=driver_num_profile, y2=0, color='g', alpha=0.3)
        axs_wakeT_cut[0].plot(driver_z_slices*1e6, driver_num_profile, 'g', label='Number profile')
        axs_wakeT_cut[0].grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        axs_wakeT_cut[0].set_xlabel(zlab)
        axs_wakeT_cut[0].set_ylabel('Beam number profiles $N(z)$')
        ax_Ez_cut_wakeT2 = axs_wakeT_cut[0].twinx()
        ax_Ez_cut_wakeT2.plot(zs_Ez*1e6, Ez/1e9, label='Full $E_z$')
        ax_Ez_cut_wakeT2.plot(z_slices*1e6, Ez_cut/1e9, 'r', label='Cut-out $E_z$')
        ax_Ez_cut_wakeT2.set_ylabel('$E_z$ [GV/m]')
        ax_Ez_cut_wakeT2.legend(loc='lower right')
        
        axs_wakeT_cut[1].fill_between(x=z_slices*1e6, y1=main_num_profile, y2=0, color='g', alpha=0.3)
        axs_wakeT_cut[1].plot(z_slices*1e6, main_num_profile, 'g', label='Number profile')
        axs_wakeT_cut[1].fill_between(x=driver_z_slices*1e6, y1=driver_num_profile, y2=0, color='g', alpha=0.3)
        axs_wakeT_cut[1].plot(driver_z_slices*1e6, driver_num_profile, 'g', label='Number profile')
        axs_wakeT_cut[1].grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        axs_wakeT_cut[1].set_xlabel(zlab)
        axs_wakeT_cut[1].set_ylabel('Beam number profiles $N(z)$')
        ax_rb_cut_wakeT2 = axs_wakeT_cut[1].twinx()
        ax_rb_cut_wakeT2.plot(zs_rho*1e6, bubble_radius*1e6, label=r'Full $r_\mathrm{b}$')
        ax_rb_cut_wakeT2.plot(z_slices*1e6, bubble_radius_cut*1e6, 'r', label=r'Cut-out $r_\mathrm{b}$')
        ax_rb_cut_wakeT2.set_ylabel(r'Bubble radius [$\mathrm{\mu}$m]')
        ax_rb_cut_wakeT2.legend(loc='upper right')
        fig_wakeT_cut.suptitle('Initial step')


    # ==================================================
    def plot_flattop_evolution(self, beam='beam'):
        """
        Plot the evolution of various beam parameters as a function of s.
        """

        # Select beam
        if beam == 'beam':
            evolution = self.evolution.beam
        elif beam == 'driver':
            evolution = self.evolution.driver

        # Check whether evolution data exist
        if len(evolution.location) == 0:
            print('No beam parameter evolution data found.')
            return

        energies = evolution.energy
        nom_energy = self.nom_energy_gain + energies[0]
        prop_length = evolution.location
        #num_particles = evolution.num_particles
        charges = evolution.charge
        
        x_offsets = evolution.x
        y_offsets = evolution.y
        #z_offsets = evolution.z_offset
        beam_size_xs = evolution.beam_size_x
        beam_size_ys = evolution.beam_size_y
        #bunch_lengths = self.evolution.bunch_length
        rel_energy_spreads = evolution.rel_energy_spread
        divergence_xs = evolution.divergence_x
        divergence_ys = evolution.divergence_y
        rel_energy_offsets = energies/nom_energy-1
        beta_xs = evolution.beta_x
        beta_ys = evolution.beta_y
        norm_emittance_xs = evolution.emit_nx
        norm_emittance_ys = evolution.emit_ny
        
        #long_label = '$s_\mathrm{stage}$ [m]'
        long_label = r'$\Delta s$ [m]'
        xlim_max = prop_length.max()
        xlim_min = prop_length.min()

        # line format
        col0 = "tab:gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        #af = 0.2
        
        fig, axs = plt.subplots(3,3)
        fig.set_figwidth(20)
        fig.set_figheight(12)
        plt.subplots_adjust(hspace=0.05)  # Reduce the space between subplots

        axs[0,0].plot(np.array([0.0, self.length_flattop]), np.array([energies[0], nom_energy])/1e9, ':', color=col0, label='Nominal value')
        axs[0,0].plot(prop_length, energies/1e9, color=col1)
        #axs[0,0].plot(prop_length, np.ones_like(prop_length)*nom_energy/1e9, ':', color=col0)
        #axs[0,0].set_xlabel(long_label)
        axs[0,0].set(xticklabels=[])
        axs[0,0].set_ylabel(r'Energy [GeV]')
        axs[0,0].set_xlim(xlim_min, xlim_max)

        axs[1,0].plot(prop_length, rel_energy_spreads*100, color=col1)
        #axs[1,0].set_xlabel(long_label)
        axs[1,0].set(xticklabels=[])
        axs[1,0].set_ylabel('Energy spread [%]')
        axs[1,0].set_yscale('log')
        axs[1,0].set_xlim(xlim_min, xlim_max)
        
        axs[2,0].plot(prop_length, np.zeros_like(rel_energy_offsets), ':', color=col0)
        axs[2,0].plot(prop_length, rel_energy_offsets*100, color=col1)
        axs[2,0].set_xlabel(long_label)
        axs[2,0].set_ylabel('Energy offset [%]')
        axs[2,0].set_xlim(xlim_min, xlim_max)

        axs[0,1].plot(prop_length, charges[0]*np.ones_like(charges)*1e9, ':', color=col0)
        axs[0,1].plot(prop_length, charges*1e9, color=col1)
        #axs[0,1].set_xlabel(long_label)
        axs[0,1].set(xticklabels=[])
        axs[0,1].set_ylabel('Charge [nC]')
        axs[0,1].set_xlim(xlim_min, xlim_max)

        #axs[1,1].plot(prop_length, bunch_lengths*1e6, color=col1)
        #axs[1,1].set_xlabel(long_label)
        #axs[1,1].set_ylabel(r'Bunch length [$\mathrm{\mu}$m]')

        axs[1,1].plot(prop_length, divergence_xs*1e6, color=col1, label=r"$\sigma_{x'} $")
        axs[1,1].plot(prop_length, divergence_ys*1e6, color=col2, label=r"$\sigma_{y'} $")
        #axs[1,1].set_xlabel(long_label)
        axs[1,1].set(xticklabels=[])
        axs[1,1].set_ylabel(r'Divergence [$\mathrm{\mu}$rad]')
        axs[1,1].legend()
        axs[1,1].set_xlim(xlim_min, xlim_max)
        
        #axs[2,1].plot(s_centroids, np.zeros(x_angle.shape), ':', color=col0)
        #axs[2,1].plot(s_centroids, x_angle*1e6, color=col1, marker='x', label=r'$\langle x\' \rangle$')
        #axs[2,1].plot(s_centroids, y_angle*1e6, color=col2, marker='x', label=r'$\langle y\' \rangle$')
        ##axs[2,1].plot(s, xp_offset_beam*1e6, color='red')
        ##axs[2,1].plot(s, yp_offset_beam*1e6, color='black')
        #axs[2,1].set_xlabel(long_label)
        #axs[2,1].set_ylabel(r'Angular offset [$\mathrm{\mu}$rad]')
        #axs[2,1].legend()
        
        axs[2,1].plot(prop_length, np.sqrt(energies/energies[0])*beta_xs[0]*1e3, ':', color=col0, label='Nominal value')
        axs[2,1].plot(prop_length, beta_xs*1e3, color=col1, label=r'$\beta_x$')
        axs[2,1].plot(prop_length, beta_ys*1e3, color=col2, label=r'$\beta_y$')
        axs[2,1].set_xlabel(long_label)
        axs[2,1].set_ylabel(r'Beta function [mm]')
        axs[2,1].legend()
        axs[2,1].set_xlim(xlim_min, xlim_max)

        axs[0,2].plot(prop_length, np.ones_like(norm_emittance_xs)*norm_emittance_xs[0]*1e6, ':', color=col0, label='Nominal value')
        axs[0,2].plot(prop_length, np.ones_like(norm_emittance_ys)*norm_emittance_ys[0]*1e6, ':', color=col0)
        axs[0,2].plot(prop_length, norm_emittance_xs*1e6, color=col1, label=r'$\varepsilon_{\mathrm{n}x}$')
        axs[0,2].plot(prop_length, norm_emittance_ys*1e6, color=col2, label=r'$\varepsilon_{\mathrm{n}y}$')
        #axs[0,2].set_xlabel(long_label)
        axs[0,2].set(xticklabels=[])
        axs[0,2].set_ylabel('Emittance, rms [mm mrad]')
        axs[0,2].set_yscale('log')
        axs[0,2].legend()
        axs[0,2].set_xlim(xlim_min, xlim_max)

        axs[1,2].plot(prop_length, (energies[0]/energies)**(1/4)*beam_size_xs[0]*1e6, ':', color=col0, label='Nominal value')
        axs[1,2].plot(prop_length, (energies[0]/energies)**(1/4)*beam_size_ys[0]*1e6, ':', color=col0)
        #axs[1,2].plot(prop_length, np.ones_like(beam_size_xs)*beam_size_xs[0]*1e6, ':', color=col0, label='Nominal value')
        #axs[1,2].plot(prop_length, np.ones(beam_size_ys.shape)*beam_size_ys[0]*1e6, ':', color=col0)
        axs[1,2].plot(prop_length, beam_size_xs*1e6, color=col1, label=r'$\sigma_x$')
        axs[1,2].plot(prop_length, beam_size_ys*1e6, color=col2, label=r'$\sigma_y$')
        #axs[1,2].set_xlabel(long_label)
        axs[1,2].set(xticklabels=[])
        axs[1,2].set_ylabel(r'Beam size, rms [$\mathrm{\mu}$m]')
        axs[1,2].set_yscale('log')
        axs[1,2].legend()
        axs[1,2].set_xlim(xlim_min, xlim_max)

        axs[2,2].plot(prop_length, np.zeros_like(x_offsets), ':', color=col0)
        axs[2,2].plot(prop_length, x_offsets*1e6, color=col1, label=r'$\langle x \rangle$')
        axs[2,2].plot(prop_length, y_offsets*1e6, color=col2, label=r'$\langle y \rangle$')
        axs[2,2].set_xlabel(long_label)
        axs[2,2].set_ylabel(r'Transverse offset [$\mathrm{\mu}$m]')
        #axs[2,2].set_yscale('log')
        axs[2,2].legend()
        axs[2,2].set_xlim(xlim_min, xlim_max)

        fig.suptitle('Stage ' + str(self.stage_number+1) + ', ' + beam)

    
    # ==================================================
    # Animate the horizontal sideview (top view)
    def animate_sideview_x(self, evolution_folder):
        
        from matplotlib import ticker as mticker
        from matplotlib.animation import FuncAnimation
        
        files = sorted(os.listdir(evolution_folder))
        
        if len(files) != len(self.evolution.beam.location):
            raise ValueError('The stored beam parameter evolution data does not have the same length as the number of beam files.')
        # TODO: make this function independent on self.evolution by using e.g. ss.append(beam.location) below.

        # Set default font size
        plt.rc('axes', titlesize=13)    # fontsize of the axes title
        plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=9)    # fontsize of the x tick labels
        plt.rc('ytick', labelsize=9)    # fontsize of the y tick labels
        plt.rc('legend', fontsize=9)    # legend fontsize
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)
        
        # get initial beam
        beam_init = Beam()
        beam_init = beam_init.load(evolution_folder + os.fsdecode(files[0]))
        
        max_sig_index = np.argmax(self.evolution.beam.beam_size_x)
        max_sig_beam = Beam()
        max_sig_beam = max_sig_beam.load(evolution_folder + os.fsdecode(files[max_sig_index]))
        dQdzdx0, zs0, xs0 = max_sig_beam.phase_space_density(max_sig_beam.zs, max_sig_beam.xs)
        dQdx0, _ = max_sig_beam.projected_density(max_sig_beam.xs, bins=xs0)
        Is0, _ = max_sig_beam.current_profile(bins=zs0/SI.c)
        
        # get final beam
        beam_final = Beam()
        beam_final = beam_final.load(evolution_folder + os.fsdecode(files[-1]))
        #dQdzdx_final, zs_final, xs_final = beam_final.phase_space_density(beam_final.zs, beam_final.xs)
        dQdx_final, _ = beam_final.projected_density(beam_final.xs, bins=xs0)
        Is_final, _ = beam_final.current_profile(bins=zs0/SI.c)
        
        # prepare centroid arrays
        x0s = []
        z0s = []
        Emeans = []
        sigzs = []
        sigxs = []
        ss = []
        emitns = []
        # set the colors and transparency
        col0 = "#cedeeb"
        col1 = "tab:blue"
        
        # frame function
        def frameFcn(i):
            
            # get beam for this frame
            beam = Beam()
            beam = beam.load(evolution_folder + os.fsdecode(files[i]))
            
            # plot mean energy evolution
            ss.append(self.evolution.beam.location[i])
            #ss.append(beam.location)                           ####### TODO: test this and replace ss.append(self.evolution.beam.location[i]) so that this function does not depend on self.evolution.
            Emeans.append(beam.energy())
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(Emeans)/1e9, '-', color=col0)
            axs[0,0].plot(ss[-1], Emeans[-1]/1e9, 'o', color=col1)
            axs[0,0].set_xlabel('Propagation length [m]')
            axs[0,0].set_ylabel('Mean energy [GeV]')
            axs[0,0].set_xlim(np.min(self.evolution.beam.location), np.max(self.evolution.beam.location))
            axs[0,0].set_ylim(beam_init.energy()*0.9e-9, beam_final.energy()*1.1e-9)
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')
            
            # plot emittance and bunch length evolution
            emitns.append(beam.norm_emittance_x())
            sigzs.append(beam.bunch_length())
            ylim_min = np.min([np.min(emitns)*1e6, beam_init.norm_emittance_x()*0.97e6, beam_final.norm_emittance_x()*0.97e6])
            ylim_max = np.max([np.max(emitns)*1e6, beam_init.norm_emittance_x()*1.05e6, beam_final.norm_emittance_x()*1.05e6])
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigzs)*1e6, np.array(emitns)*1e6, '-', color=col0)
            axs[0,1].plot(sigzs[-1]*1e6, emitns[-1]*1e6, 'o', color=col1)
            axs[0,1].set_ylim(ylim_min, ylim_max)
            #axs[0,1].set_ylim(np.min([np.min(emitns)*0.95e6, beam_final.norm_emittance_x()*0.8e6]), np.max([np.max(emitns)*1.05e6, emitns[0]*1.2e6]))
            #axs[0,1].set_ylim(np.min(self.evolution.beam.norm_emittance_x)*0.8e6, np.max(self.evolution.beam.norm_emittance_x)*1.2e6)
            axs[0,1].set_xlim(np.min([np.min(sigzs)*0.95e6, beam_final.bunch_length()*0.8e6]), np.max([np.max(sigzs)*1.05e6, sigzs[0]*1.2e6]))
            #axs[0,1].set_xlim(np.min(self.evolution.beam.bunch_length)*0.95e6, np.max(self.evolution.beam.bunch_length)*1.05e6)
            axs[0,1].set_xscale('log')
            axs[0,1].set_yscale('log')
            axs[0,1].set_xlabel(r'Bunch length [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel('Norm. emittance\n[mm mrad]')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            axs[0,1].xaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
            
            # plot phase space
            dQdzdx, zs, xs = beam.phase_space_density(beam.zs, beam.xs, hbins=zs0, vbins=xs0)
            axs[1,0].cla()
            cax = axs[1,0].pcolor(zs*1e6, xs*1e6, -dQdzdx, cmap=CONFIG.default_cmap, shading='auto')
            axs[1,0].set_ylabel(r"Transverse offset, $x$ [$\mathrm{\mu}$m]")
            axs[1,0].set_title('Horizontal sideview (top view)')
            
            # plot current profile
            af = 0.15
            Is, ts = beam.current_profile(bins=zs0/SI.c)
            axs[2,0].cla()
            axs[2,0].fill(np.concatenate((ts, np.flip(ts)))*SI.c*1e6, -np.concatenate((Is, np.zeros(Is.size)))/1e3, alpha=af, color=col1)
            axs[2,0].plot(ts*SI.c*1e6, -Is/1e3)
            axs[2,0].set_xlim([np.min(zs0)*1e6, np.max(zs0)*1e6])
            axs[2,0].set_ylim([0, np.max([np.max(-Is0), np.max(-Is_final)])*1.3e-3])
            axs[2,0].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel(r'$I$ [kA]')
            
            # plot position projection
            dQdx, xs2 = beam.projected_density(beam.xs, bins=xs0)
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdx, np.zeros(dQdx.size)))*1e3, np.concatenate((xs2, np.flip(xs2)))*1e6, alpha=af, color=col1)
            axs[1,1].plot(-dQdx*1e3, xs2*1e6, color=col1)
            axs[1,1].set_ylim([np.min(xs0)*1e6, np.max(xs0)*1e6])
            axs[1,1].set_xlim([0, np.max([np.max(-dQdx), np.max(-dQdx_final)])*1.1e3])
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel(r"$dQ/dx$ [nC/$\mathrm{\mu}$m]")
            axs[1,1].set_ylabel(r"$x$ [$\mathrm{\mu}$m]")
        
            # plot centroid evolution
            z0s.append(beam.z_offset())
            #sigxs.append(beam.beam_size_x())
            x0s.append(beam.x_offset())
            ylim_min = pad_downwards(np.min([np.min(x0s), np.min(self.evolution.beam.x)]), padding=0.1)*1e6
            ylim_max = pad_upwards(np.max([np.max(x0s), np.max(self.evolution.beam.x)]), padding=0.1)*1e6
            axs[2,1].cla()
            axs[2,1].plot(np.array(z0s)*1e6, np.array(x0s)*1e6, '-', color=col0)
            axs[2,1].plot(z0s[-1]*1e6, x0s[-1]*1e6, 'o', color=col1)
            axs[2,1].set_xlabel(r'$z$ offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel(r'$x$ offset [$\mathrm{\mu}$m]')
            #axs[2,1].set_xlim(np.min([np.min(z0s)-sigzs[0]/6, z0s[0]-sigzs[0]/2])*1e6, np.max([np.max(z0s)+sigzs[0]/6, (z0s[0]+sigzs[0]/2)])*1e6)
            axs[2,1].set_xlim((z0s[0]-sigzs[0]/10)*1e6, (np.max(self.evolution.beam.z)+sigzs[0]/10)*1e6)
            #axs[2,1].set_xlim(np.min(self.evolution.beam.z)*0.95e6, np.max(self.evolution.beam.z)*1.05e6)
            #axs[2,1].set_ylim(np.min(-np.max(x0s)*1.1, -0.1*sigxs[0])*1e6, np.max(np.max(x0s)*1.1, 0.1*sigxs[0])*1e6)
            #axs[2,1].set_ylim(np.min(self.evolution.beam.x)*1.1e6, np.max(self.evolution.beam.x)*1.1e6)
            axs[2,1].set_ylim(ylim_min, ylim_max)
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')
        
            return cax

        animation = FuncAnimation(fig, frameFcn, frames=range(len(files)), repeat=False, interval=100)
        
        # save the animation as a GIF
        plot_path = self.run_path + 'plots' + os.sep
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path +'sideview_x_stage_' + str(self.stage_number)+ '.gif'
        animation.save(filename, writer="pillow", fps=20)

        # hide the figure
        plt.close()

        # Reset to default matplotlib settings
        plt.rcdefaults()

        return filename


    # ==================================================
    # Animate the vertical sideview
    def animate_sideview_y(self, evolution_folder):
        
        from matplotlib import ticker as mticker
        from matplotlib.animation import FuncAnimation
        
        files = sorted(os.listdir(evolution_folder))

        if len(files) != len(self.evolution.beam.location):
            raise ValueError('The stored beam parameter evolution data does not have the same length as the number of beam files.')
        
        # Set default font size
        plt.rc('axes', titlesize=13)    # fontsize of the axes title
        plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=9)    # fontsize of the x tick labels
        plt.rc('ytick', labelsize=9)    # fontsize of the y tick labels
        plt.rc('legend', fontsize=9)    # legend fontsize
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)
        
        # get initial beam
        beam_init = Beam()
        beam_init = beam_init.load(evolution_folder + os.fsdecode(files[0]))

        # Get max beam size beam
        max_sig_index = np.argmax(self.evolution.beam.beam_size_y)
        max_sig_beam = Beam()
        max_sig_beam = max_sig_beam.load(evolution_folder + os.fsdecode(files[max_sig_index]))
        dQdzdy0, zs0, ys0 = max_sig_beam.phase_space_density(max_sig_beam.zs, max_sig_beam.ys)
        dQdy0, _ = max_sig_beam.projected_density(max_sig_beam.ys, bins=ys0)
        Is0, _ = max_sig_beam.current_profile(bins=zs0/SI.c)
        
        # get final beam
        beam_final = Beam()
        beam_final = beam_final.load(evolution_folder + os.fsdecode(files[-1]))
        #dQdzdy_final, zs_final, ys_final = beam_final.phase_space_density(beam_final.zs, beam_final.ys)
        dQdy_final, _ = beam_final.projected_density(beam_final.ys, bins=ys0)
        Is_final, _ = beam_final.current_profile(bins=zs0/SI.c)
        
        # prepare centroid arrays
        y0s = []
        z0s = []
        Emeans = []
        sigzs = []
        sigys = []
        ss = []
        emitns = []
        # set the colors and transparency
        col0 = "#f5d9c1"
        col1 = "tab:orange"
        
        # frame function
        def frameFcn(i):
            
            # get beam for this frame
            beam = Beam()
            beam = beam.load(evolution_folder + os.fsdecode(files[i]))
            
            # plot mean energy evolution
            ss.append(self.evolution.beam.location[i])
            Emeans.append(beam.energy())
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(Emeans)/1e9, '-', color=col0)
            axs[0,0].plot(ss[-1], Emeans[-1]/1e9, 'o', color=col1)
            axs[0,0].set_xlabel('Propagation length [m]')
            axs[0,0].set_ylabel('Mean energy [GeV]')
            axs[0,0].set_xlim(np.min(self.evolution.beam.location), np.max(self.evolution.beam.location))
            axs[0,0].set_ylim(beam_init.energy()*0.9e-9, beam_final.energy()*1.1e-9)
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')
            
            # plot emittance and bunch length evolution
            emitns.append(beam.norm_emittance_y())
            sigzs.append(beam.bunch_length())
            ylim_min = np.min([np.min(emitns)*1e6, beam_init.norm_emittance_y()*0.97e6, beam_final.norm_emittance_y()*0.97e6])
            ylim_max = np.max([np.max(emitns)*1e6, beam_init.norm_emittance_y()*1.05e6, beam_final.norm_emittance_y()*1.05e6])
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigzs)*1e6, np.array(emitns)*1e6, '-', color=col0)
            axs[0,1].plot(sigzs[-1]*1e6, emitns[-1]*1e6, 'o', color=col1)
            axs[0,1].set_ylim(ylim_min, ylim_max)
            axs[0,1].set_xlim(np.min([np.min(sigzs)*0.95e6, beam_final.bunch_length()*0.8e6]), np.max([np.max(sigzs)*1.05e6, sigzs[0]*1.2e6]))
            axs[0,1].set_xscale('log')
            axs[0,1].set_yscale('log')
            axs[0,1].set_xlabel(r'Bunch length [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel('Norm. emittance\n[mm mrad]')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            axs[0,1].xaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
            
            # plot phase space
            dQdzdy, zs, ys = beam.phase_space_density(beam.zs, beam.ys, hbins=zs0, vbins=ys0)
            axs[1,0].cla()
            cax = axs[1,0].pcolor(zs*1e6, ys*1e6, -dQdzdy, cmap=CONFIG.default_cmap, shading='auto')
            axs[1,0].set_ylabel(r"Transverse offset, $y$ [$\mathrm{\mu}$m]")
            axs[1,0].set_title('Vertical sideview')
            
            # plot current profile
            af = 0.15
            Is, ts = beam.current_profile(bins=zs0/SI.c)
            axs[2,0].cla()
            axs[2,0].fill(np.concatenate((ts, np.flip(ts)))*SI.c*1e6, -np.concatenate((Is, np.zeros(Is.size)))/1e3, alpha=af, color=col1)
            axs[2,0].plot(ts*SI.c*1e6, -Is/1e3, color=col1)
            axs[2,0].set_xlim([np.min(zs0)*1e6, np.max(zs0)*1e6])
            axs[2,0].set_ylim([0, np.max([np.max(-Is0), np.max(-Is_final)])*1.3e-3])
            axs[2,0].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel(r'$I$ [kA]')
            
            # plot position projection
            dQdy, ys2 = beam.projected_density(beam.ys, bins=ys0)
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdy, np.zeros(dQdy.size)))*1e3, np.concatenate((ys2, np.flip(ys2)))*1e6, alpha=af, color=col1)
            axs[1,1].plot(-dQdy*1e3, ys2*1e6, color=col1)
            axs[1,1].set_ylim([np.min(ys0)*1e6, np.max(ys0)*1e6])
            axs[1,1].set_xlim([0, np.max([np.max(-dQdy), np.max(-dQdy_final)])*1.1e3])
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel(r"$dQ/dy$ [nC/$\mathrm{\mu}$m]")
            axs[1,1].set_ylabel(r"$y$ [$\mathrm{\mu}$m]")
        
            # plot centroid evolution
            z0s.append(beam.z_offset())
            #sigys.append(beam.beam_size_y())
            y0s.append(beam.y_offset())
            ylim_min = pad_downwards(np.min([np.min(y0s), np.min(self.evolution.beam.y)]), padding=0.1)*1e6
            ylim_max = pad_upwards(np.max([np.max(y0s), np.max(self.evolution.beam.y)]), padding=0.1)*1e6
            axs[2,1].cla()
            axs[2,1].plot(np.array(z0s)*1e6, np.array(y0s)*1e6, '-', color=col0)
            axs[2,1].plot(z0s[-1]*1e6, y0s[-1]*1e6, 'o', color=col1)
            axs[2,1].set_xlabel(r'$z$ offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel(r'$y$ offset [$\mathrm{\mu}$m]')
            axs[2,1].set_xlim(np.min([np.min(z0s)-sigzs[0]/6, z0s[0]-sigzs[0]/2])*1e6, np.max([np.max(z0s)+sigzs[0]/6, (z0s[0]+sigzs[0]/2)])*1e6)
            axs[2,1].set_ylim(ylim_min, ylim_max)
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')
        
            return cax

        animation = FuncAnimation(fig, frameFcn, frames=range(len(files)), repeat=False, interval=100)
        
        # save the animation as a GIF
        plot_path = self.run_path + 'plots' + os.sep
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path +'sideview_y_stage_' + str(self.stage_number)+ '.gif'
        animation.save(filename, writer="pillow", fps=20)

        # hide the figure
        plt.close()

        # Reset to default matplotlib settings
        plt.rcdefaults()

        return filename


    # ==================================================
    # Animate the horizontal phase space
    def animate_phasespace_x(self, evolution_folder):
        
        from matplotlib import ticker as mticker
        from matplotlib.animation import FuncAnimation
        
        files = sorted(os.listdir(evolution_folder))

        if len(files) != len(self.evolution.beam.location):
            raise ValueError('The stored beam parameter evolution data does not have the same length as the number of beam files.')

        # Set default font size
        plt.rc('axes', titlesize=13)    # fontsize of the axes title
        plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=9)    # fontsize of the x tick labels
        plt.rc('ytick', labelsize=9)    # fontsize of the y tick labels
        plt.rc('legend', fontsize=9)    # legend fontsize
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)

        # get initial beam
        beam_init = Beam()
        beam_init = beam_init.load(evolution_folder + os.fsdecode(files[0]))

        # get max beam size beam
        max_sig_index = np.argmax(self.evolution.beam.beam_size_x )
        max_sig_beam = Beam()
        max_sig_beam = max_sig_beam.load(evolution_folder + os.fsdecode(files[max_sig_index]))
        dQdxdpx0, xs0, _ = max_sig_beam.phase_space_density(max_sig_beam.xs, max_sig_beam.pxs)

        # get max divergence beam
        max_sig_xp_index = np.argmax(self.evolution.beam.divergence_x)
        max_sig_xp_beam = Beam()
        max_sig_xp_beam = max_sig_xp_beam.load(evolution_folder + os.fsdecode(files[max_sig_xp_index]))
        _, _, pxs0 = max_sig_xp_beam.phase_space_density(max_sig_xp_beam.xs, max_sig_xp_beam.pxs)
        
        # get final beam
        beam_final = Beam()
        beam_final = beam_final.load(evolution_folder + os.fsdecode(files[-1]))
        _, _, pxs_final = beam_final.phase_space_density(beam_final.xs, beam_final.pxs)

        # calculate limits
        #pxlim = np.max(np.abs(pxs0))
        pxlim = np.max([np.max(np.abs(pxs0)), np.max(np.abs(pxs_final))])
        if np.max(np.abs(pxs_final)) > np.max(np.abs(pxs0)):
            pxs0 = pxs_final
        
        # calculate projections
        #dQdx0, _ = max_sig_beam.projected_density(max_sig_beam.xs, bins=xs0)
        #dQdpx0, _ = max_sig_xp_beam.projected_density(max_sig_xp_beam.pxs, bins=pxs0)
        dQdx0, _ = max_sig_xp_beam.projected_density(max_sig_xp_beam.xs, bins=xs0)
        dQdpx0, _ = max_sig_beam.projected_density(max_sig_beam.pxs, bins=pxs0)
        dQdx_final, _ = beam_final.projected_density(beam_final.xs, bins=xs0)
        dQdpx_final, _ = beam_final.projected_density(beam_final.pxs, bins=pxs0)
        
        # prepare centroid arrays
        x0s = []
        xp0s = []
        sigxs = []
        sigxps = []
        ss = []
        emitns = []
        
        # set the colors and transparency
        col0 = "#cedeeb"
        col1 = "tab:blue"
        
        # frame function
        def frameFcn(i):
            
            # get beam for this frame
            beam = Beam()
            beam = beam.load(evolution_folder + os.fsdecode(files[i]))
            
            # plot emittance evolution
            ss.append(self.evolution.beam.location[i])
            emitns.append(beam.norm_emittance_x(clean=False))
            ylim_min = np.min([np.min(emitns)*1e6, beam_init.norm_emittance_x()*0.97e6, beam_final.norm_emittance_x()*0.97e6])
            ylim_max = np.max([np.max(emitns)*1e6, beam_init.norm_emittance_x()*1.05e6, beam_final.norm_emittance_x()*1.05e6])
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(emitns)*1e6, '-', color=col0)
            axs[0,0].plot(ss[-1], emitns[-1]*1e6, 'o', color=col1)
            axs[0,0].set_xlabel('Propagation distance [m]')
            axs[0,0].set_ylabel('Norm. emittance\n[mm mrad]')
            axs[0,0].set_xlim(np.min(self.evolution.beam.location), np.max(self.evolution.beam.location))
            #axs[0,0].set_ylim(beam_init.norm_emittance_x()*0.9e6, np.max(self.evolution.norm_emittance_x)*1.2e6)
            axs[0,0].set_ylim(ylim_min, ylim_max)
            #axs[0,0].set_yscale('log')
            #axs[0,0].yaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')
            
            # plot beam size and divergence evolution
            sigxs.append(beam.beam_size_x())
            sigxps.append(beam.divergence_x())
            xlim_min = np.min([np.min(sigxs)*0.9e6,  beam_final.beam_size_x()*0.9e6])
            xlim_max = np.max([np.max(sigxs)*1.1e6, max_sig_beam.beam_size_x()*1.1e6])
            ylim_min = np.min([np.min(sigxps)*0.9e3, np.min(self.evolution.beam.divergence_x)*0.9e3])
            ylim_max = np.max([np.max(sigxps)*1.1e3, max_sig_xp_beam.divergence_x()*1.1e3])
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigxs)*1e6, np.array(sigxps)*1e3, '-', color=col0)
            axs[0,1].plot(sigxs[-1]*1e6, sigxps[-1]*1e3, 'o', color=col1)
            axs[0,1].set_xlim(xlim_min, xlim_max)
            #axs[0,1].set_xlim(beam_init.beam_size_x()*0.8e6, max_sig_beam.beam_size_x()*1.2e6)
            #axs[0,1].set_ylim(np.min([np.min(sigxps)*0.9e3, beam_final.divergence_x()*0.8e3]), np.max([np.max(sigxps)*1.1e3, sigxs[0]*1.2e3]))
            #axs[0,1].set_ylim(np.min(self.evolution.beam.divergence_x)*0.9e3, max_sig_xp_beam.divergence_x()*1.1e3)
            axs[0,1].set_ylim(ylim_min, ylim_max)
            axs[0,1].set_xscale('log')
            axs[0,1].set_yscale('log')
            axs[0,1].set_xlabel(r'Beam size [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel('Divergence [mrad]')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            axs[0,1].xaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
            
            # plot phase space
            dQdxdpx, xs, pxs = beam.phase_space_density(beam.xs, beam.pxs, hbins=xs0, vbins=pxs0)
            axs[1,0].cla()
            cax = axs[1,0].pcolor(xs*1e6, pxs*1e-6*SI.c/SI.e, -dQdxdpx, cmap=CONFIG.default_cmap, shading='auto')
            axs[1,0].set_ylabel("Momentum, $p_x$ [MeV/c]")
            axs[1,0].set_title('Horizontal phase space')
            axs[1,0].set_ylim([-pxlim*1e-6*SI.c/SI.e, pxlim*1e-6*SI.c/SI.e])
            
            # plot position projection
            af = 0.15
            dQdx, xs2 = beam.projected_density(beam.xs, bins=xs0)
            axs[2,0].cla()
            axs[2,0].fill(np.concatenate((xs2, np.flip(xs2)))*1e6, -np.concatenate((dQdx, np.zeros(dQdx.size)))*1e3, alpha=af, color=col1)
            axs[2,0].plot(xs2*1e6, -dQdx*1e3, color=col1)
            axs[2,0].set_xlim([np.min(xs0)*1e6, np.max(xs0)*1e6])
            axs[2,0].set_ylim([0, np.max([np.max(-dQdx0), np.max(-dQdx_final)])*1.2e3])
            axs[2,0].set_xlabel(r'Transverse position, $x$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel(r'$dQ/dx$ [nC/$\mathrm{\mu}$m]')
            
            # plot angular projection
            dQdpx, pxs2 = beam.projected_density(beam.pxs, bins=pxs0)
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdpx, np.zeros(dQdpx.size)))*1e9/(1e-6*SI.c/SI.e), np.concatenate((pxs2, np.flip(pxs2)))*1e-6*SI.c/SI.e, alpha=af, color=col1)
            axs[1,1].plot(-dQdpx*1e9/(1e-6*SI.c/SI.e), pxs2*1e-6*SI.c/SI.e, color=col1)
            axs[1,1].set_xlim([0, np.max([np.max(-dQdpx0), np.max(-dQdpx_final)])*1e9/(1e-6*SI.c/SI.e)*1.2])
            axs[1,1].set_ylim([-pxlim*1e-6*SI.c/SI.e, pxlim*1e-6*SI.c/SI.e])
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel(r"$dQ/dp_x$ [nC c/MeV]")
            axs[1,1].set_ylabel("Momentum, $p_x$ [MeV/c]")
            
            # plot centroid evolution
            x0s.append(beam.x_offset())
            xp0s.append(beam.x_angle())
            xlim_min = pad_downwards(np.min([np.min(x0s), np.min(self.evolution.beam.x)]), padding=0.1)*1e6
            xlim_max = pad_upwards(np.max([np.max(x0s), np.max(self.evolution.beam.x)]), padding=0.1)*1e6
            ylim_min = pad_downwards(np.min([np.min(xp0s), np.min(self.evolution.beam.x_angle)]), padding=0.1)*1e6
            ylim_max = pad_upwards(np.max([np.max(xp0s), np.max(self.evolution.beam.x_angle)]), padding=0.1)*1e6
            axs[2,1].cla()
            axs[2,1].plot(np.array(x0s)*1e6, np.array(xp0s)*1e6, '-', color=col0)
            axs[2,1].plot(x0s[-1]*1e6, xp0s[-1]*1e6, 'o', color=col1)
            axs[2,1].set_xlabel(r'Centroid offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel(r'Centroid angle [$\mathrm{\mu}$rad]')
            axs[2,1].set_xlim(xlim_min, xlim_max)
            axs[2,1].set_ylim(ylim_min, ylim_max)
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')
            
            return cax
        
        # make all frames
        animation = FuncAnimation(fig, frameFcn, frames=range(len(files)), repeat=False, interval=100)
        
        # save the animation as a GIF
        plot_path = self.run_path + 'plots/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path + 'phasespace_x_stage_' + str(self.stage_number) + '.gif'
        animation.save(filename, writer="pillow", fps=20)

        # hide the figure
        plt.close()

        return filename


    # ==================================================
    # Animate the vertical phase space
    def animate_phasespace_y(self, evolution_folder):
        
        from matplotlib import ticker as mticker
        from matplotlib.animation import FuncAnimation
        
        files = sorted(os.listdir(evolution_folder))

        if len(files) != len(self.evolution.beam.location):
            raise ValueError('The stored beam parameter evolution data does not have the same length as the number of beam files.')

        # Set default font size
        plt.rc('axes', titlesize=13)    # fontsize of the axes title
        plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=9)    # fontsize of the x tick labels
        plt.rc('ytick', labelsize=9)    # fontsize of the y tick labels
        plt.rc('legend', fontsize=9)    # legend fontsize
        
        # set up figure
        fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2, 1]})
        fig.set_figwidth(CONFIG.plot_width_default*0.8)
        fig.set_figheight(CONFIG.plot_width_default*0.8)
        
        # get initial beam
        beam_init = Beam()
        beam_init = beam_init.load(evolution_folder + os.fsdecode(files[0]))

        # get max beam size beam
        max_sig_index = np.argmax(self.evolution.beam.beam_size_y)
        max_sig_beam = Beam()
        max_sig_beam = max_sig_beam.load(evolution_folder + os.fsdecode(files[max_sig_index]))
        dQdydpy0, ys0, _ = max_sig_beam.phase_space_density(max_sig_beam.ys, max_sig_beam.pys)

        # get max divergence beam
        max_sig_yp_index = np.argmax(self.evolution.beam.divergence_y)
        max_sig_yp_beam = Beam()
        max_sig_yp_beam = max_sig_yp_beam.load(evolution_folder + os.fsdecode(files[max_sig_yp_index]))
        _, _, pys0 = max_sig_yp_beam.phase_space_density(max_sig_yp_beam.ys, max_sig_yp_beam.pys)
        
        # get final beam
        beam_final = Beam()
        beam_final = beam_final.load(evolution_folder + os.fsdecode(files[-1]))
        _, _, pys_final = beam_final.phase_space_density(beam_final.ys, beam_final.pys)

        # calculate limits
        #pylim = np.max(np.abs(pys0))
        pylim = np.max([np.max(np.abs(pys0)), np.max(np.abs(pys_final))])
        if np.max(np.abs(pys_final)) > np.max(np.abs(pys0)):
            pys0 = pys_final
        
        # calculate projections
        #dQdy0, _ = max_sig_beam.projected_density(max_sig_beam.ys, bins=ys0)
        #dQdpy0, _ = max_sig_yp_beam.projected_density(max_sig_yp_beam.pys, bins=pys0)
        dQdy0, _ = max_sig_yp_beam.projected_density(max_sig_yp_beam.ys, bins=ys0)
        dQdpy0, _ = max_sig_beam.projected_density(max_sig_beam.pys, bins=pys0)
        dQdy_final, _ = beam_final.projected_density(beam_final.ys, bins=ys0)
        dQdpy_final, _ = beam_final.projected_density(beam_final.pys, bins=pys0)
        
        # prepare centroid arrays
        y0s = []
        yp0s = []
        sigys = []
        sigyps = []
        ss = []
        emitns = []
        
        # set the colors and transparency
        col0 = "#f5d9c1"
        col1 = "tab:orange"
        
        # frame function
        def frameFcn(i):
            
            # get beam for this frame
            beam = Beam()
            beam = beam.load(evolution_folder + os.fsdecode(files[i]))
            
            # plot emittance evolution
            ss.append(self.evolution.beam.location[i])
            emitns.append(beam.norm_emittance_y(clean=False))
            ylim_min = np.min([np.min(emitns)*1e6, beam_init.norm_emittance_y()*0.97e6, beam_final.norm_emittance_y()*0.97e6])
            ylim_max = np.max([np.max(emitns)*1e6, beam_init.norm_emittance_y()*1.05e6, beam_final.norm_emittance_y()*1.05e6])
            axs[0,0].cla()
            axs[0,0].plot(np.array(ss), np.array(emitns)*1e6, '-', color=col0)
            axs[0,0].plot(ss[-1], emitns[-1]*1e6, 'o', color=col1)
            axs[0,0].set_xlabel('Propagation distance [m]')
            axs[0,0].set_ylabel('Norm. emittance\n[mm mrad]')
            axs[0,0].set_xlim(np.min(self.evolution.beam.location), np.max(self.evolution.beam.location))
            axs[0,0].set_ylim(ylim_min, ylim_max)
            #axs[0,0].set_yscale('log')
            #axs[0,0].yaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,0].xaxis.tick_top()
            axs[0,0].xaxis.set_label_position('top')
            
            # plot beam size and divergence evolution
            sigys.append(beam.beam_size_y())
            sigyps.append(beam.divergence_y())
            xlim_min = np.min([np.min(sigys)*0.9e6,  beam_final.beam_size_y()*0.9e6])
            #xlim_max = np.max([np.max(sigys)*1.1e6, max_sig_beam.beam_size_y()*1.1e6])
            xlim_max = np.max([np.max(sigys)*1.1e6, sigys[0]*1.1e6])
            ylim_min = np.min([np.min(sigyps)*0.9e3, np.min(self.evolution.beam.divergence_y)*0.9e3])
            #ylim_max = np.max([np.max(sigyps)*1.1e3, max_sig_yp_beam.divergence_y()*1.1e3])
            ylim_max = np.max([np.max(sigyps)*1.1e3, sigyps[0]*1.1e3])
            axs[0,1].cla()
            axs[0,1].plot(np.array(sigys)*1e6, np.array(sigyps)*1e3, '-', color=col0)
            axs[0,1].plot(sigys[-1]*1e6, sigyps[-1]*1e3, 'o', color=col1)
            axs[0,1].set_xlim(xlim_min, xlim_max)
            axs[0,1].set_ylim(ylim_min, ylim_max)
            axs[0,1].set_xscale('log')
            axs[0,1].set_yscale('log')
            axs[0,1].set_xlabel(r'Beam size [$\mathrm{\mu}$m]')
            axs[0,1].set_ylabel('Divergence [mrad]')
            axs[0,1].yaxis.tick_right()
            axs[0,1].yaxis.set_label_position('right')
            axs[0,1].xaxis.tick_top()
            axs[0,1].xaxis.set_label_position('top')
            axs[0,1].xaxis.set_minor_formatter(mticker.NullFormatter())
            axs[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
            
            # plot phase space
            dQdydpy, ys, pys = beam.phase_space_density(beam.ys, beam.pys, hbins=ys0, vbins=pys0)
            axs[1,0].cla()
            cax = axs[1,0].pcolor(ys*1e6, pys*1e-6*SI.c/SI.e, -dQdydpy, cmap=CONFIG.default_cmap, shading='auto')
            axs[1,0].set_ylabel("Momentum, $p_y$ [MeV/c]")
            axs[1,0].set_title('Vertical phase space')
            axs[1,0].set_ylim([-pylim*1e-6*SI.c/SI.e, pylim*1e-6*SI.c/SI.e])
            
            # plot position projection
            af = 0.15
            dQdy, ys2 = beam.projected_density(beam.ys, bins=ys0)
            axs[2,0].cla()
            axs[2,0].fill(np.concatenate((ys2, np.flip(ys2)))*1e6, -np.concatenate((dQdy, np.zeros(dQdy.size)))*1e3, alpha=af, color=col1)
            axs[2,0].plot(ys2*1e6, -dQdy*1e3, color=col1)
            axs[2,0].set_xlim([np.min(ys0)*1e6, np.max(ys0)*1e6])
            axs[2,0].set_ylim([0, np.max([np.max(-dQdy0), np.max(-dQdy_final)])*1.2e3])
            axs[2,0].set_xlabel(r'Transverse position, $y$ [$\mathrm{\mu}$m]')
            axs[2,0].set_ylabel(r'$dQ/dy$ [nC/$\mathrm{\mu}$m]')
            
            # plot angular projection
            dQdpy, pys2 = beam.projected_density(beam.pys, bins=pys0)
            axs[1,1].cla()
            axs[1,1].fill(-np.concatenate((dQdpy, np.zeros(dQdpy.size)))*1e9/(1e-6*SI.c/SI.e), np.concatenate((pys2, np.flip(pys2)))*1e-6*SI.c/SI.e, alpha=af, color=col1)
            axs[1,1].plot(-dQdpy*1e9/(1e-6*SI.c/SI.e), pys2*1e-6*SI.c/SI.e, color=col1)
            axs[1,1].set_xlim([0, np.max([np.max(-dQdpy0), np.max(-dQdpy_final)])*1e9/(1e-6*SI.c/SI.e)*1.2])
            axs[1,1].set_ylim([-pylim*1e-6*SI.c/SI.e, pylim*1e-6*SI.c/SI.e])
            axs[1,1].yaxis.tick_right()
            axs[1,1].yaxis.set_label_position('right')
            axs[1,1].xaxis.set_label_position('top')
            axs[1,1].set_xlabel(r"$dQ/dp_y$ [nC c/MeV]")
            axs[1,1].set_ylabel("Momentum, $p_y$ [MeV/c]")
            
            # plot centroid evolution
            y0s.append(beam.y_offset())
            yp0s.append(beam.y_angle())
            xlim_min = pad_downwards(np.min([np.min(y0s), np.min(self.evolution.beam.y)]), padding=0.1)*1e6
            xlim_max = pad_upwards(np.max([np.max(y0s), np.max(self.evolution.beam.y)]), padding=0.1)*1e6
            ylim_min = pad_downwards(np.min([np.min(yp0s), np.min(self.evolution.beam.y_angle)]), padding=0.1)*1e6
            ylim_max = pad_upwards(np.max([np.max(yp0s), np.max(self.evolution.beam.y_angle)]), padding=0.1)*1e6
            axs[2,1].cla()
            axs[2,1].plot(np.array(y0s)*1e6, np.array(yp0s)*1e6, '-', color=col0)
            axs[2,1].plot(y0s[-1]*1e6, yp0s[-1]*1e6, 'o', color=col1)
            axs[2,1].set_xlabel(r'Centroid offset [$\mathrm{\mu}$m]')
            axs[2,1].set_ylabel(r'Centroid angle [$\mathrm{\mu}$rad]')
            axs[2,1].set_xlim(xlim_min, xlim_max)
            axs[2,1].set_ylim(ylim_min, ylim_max)
            axs[2,1].yaxis.tick_right()
            axs[2,1].yaxis.set_label_position('right')
            
            return cax
        
        # make all frames
        animation = FuncAnimation(fig, frameFcn, frames=range(len(files)), repeat=False, interval=100)
        
        # save the animation as a GIF
        plot_path = self.run_path + 'plots/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = plot_path + 'phasespace_y_stage_' + str(self.stage_number) + '.gif'
        animation.save(filename, writer="pillow", fps=20)

        # hide the figure
        plt.close()

        return filename

    
    # ==================================================
    def print_initial_summary(self, drive_beam, main_beam):
        
        print('======================================================================')
        print(f"Time step [betatron wavelength/c]:\t\t\t {self.time_step_mod :.3f}")
        print(f"Interstages enabled:\t\t\t\t\t {str(self.interstages_enabled) :s}")

        if self.interstage_dipole_field is None:
            print(f"Interstage dipole field:\t\t\t\t {'Not registered.' :s}")
        elif callable(self.interstage_dipole_field):
            interstage_dipole_field = self.interstage_dipole_field(main_beam.energy())
            print(f"Interstage dipole field:\t\t\t\t {interstage_dipole_field :.3f}")
        else:
            interstage_dipole_field = self.interstage_dipole_field
            print(f"Interstage dipole field:\t\t\t\t {interstage_dipole_field :.3f}")
            
        print(f"Ramp beta magnification:\t\t\t\t {self.ramp_beta_mag :.3f}")

        if self.main_source is None:
            print(f"Symmetrised main beam:\t\t\t\t\t Not registered")
        elif self.main_source.symmetrize:
            print(f"Symmetrised main beam:\t\t\t\t\t x, y, xp, yp symmetrised")
        elif self.main_source.symmetrize_6d:
            print(f"Symmetrised main beam:\t\t\t\t\t 6D symmetrised")
        else:
            print(f"Symmetrised main beam:\t\t\t\t\t Not symmetrised.")

        if self.driver_source.symmetrize:
            print(f"Symmetrised drive beam:\t\t\t\t\t x, y, xp, yp symmetrised\n")
        elif self.driver_source.symmetrize_6d:
            print(f"Symmetrised drive beam:\t\t\t\t\t 6D symmetrised\n")
        else:
            print(f"Symmetrised drive beam:\t\t\t\t\t Not symmetrised.\n")
        
        print(f"Transverse wake instability enabled:\t\t\t {str(self.enable_tr_instability) :s}")
        print(f"Radiation reaction enabled:\t\t\t\t {str(self.enable_radiation_reaction) :s}")
        print(f"Ion motion enabled:\t\t\t\t\t {str(self.enable_ion_motion) :s}\n")
        
        stage_length = self.length_flattop 
        if stage_length is None:
            print(f"Stage flattop length [m]:\t\t\t\t Not set")
        else:
            print(f"Stage flattop length [m]:\t\t\t\t {stage_length :.3f}")
        print(f"Plasma density [m^-3]:\t\t\t\t\t {self.plasma_density :.3e}")
        print(f"Drive beam x jitter (std) [um]:\t\t\t\t {self.driver_source.jitter.x*1e6 :.3f}")
        print(f"Drive beam y jitter (std) [um]:\t\t\t\t {self.driver_source.jitter.y*1e6 :.3f}")
        print('----------------------------------------------------------------------\n')
        
        print('-------------------------------------------------------------------------------------')
        print('Quantity \t\t\t\t\t Drive beam \t\t Main beam')
        print('-------------------------------------------------------------------------------------')
        print(f"Number of macroparticles:\t\t\t {len(drive_beam.xs()) :d}\t\t\t {len(main_beam.xs()) :d}")
        print(f"Initial beam population:\t\t\t {(np.sum(drive_beam.weightings())) :.3e} \t\t {(np.sum(main_beam.weightings())) :.3e}\n")
        
        zs = main_beam.zs()
        indices = np.argsort(zs)
        zs_sorted = zs[indices]
        weights = main_beam.weightings()
        weights_sorted = weights[indices]
        
        print(f"Weighted main beam gradient [GV/m]:\t\t\t  \t\t {weighted_mean(self.Ez_fit_obj(zs_sorted), weights_sorted, clean=False)/1e9 :.3f}")
        #_, z_centre = find_closest_value_in_arr(arr=main_beam.zs(), val=main_beam.z_offset())  # Centre z of beam.
        #print(f"Beam centre gradient [GV/m]:\t\t\t\t  \t\t {self.Ez_fit_obj(z_centre)/1e9 :.3f}")
        
        print(f"Initial mean gamma:\t\t\t\t {drive_beam.gamma() :.3f} \t\t {main_beam.gamma() :.3f}")
        print(f"Initial mean energy [GeV]:\t\t\t {drive_beam.energy()/1e9 :.3f} \t\t {main_beam.energy()/1e9 :.3f}")
        print(f"Initial rms energy spread [%]:\t\t\t {drive_beam.rel_energy_spread()*1e2 :.3f} \t\t\t {main_beam.rel_energy_spread()*1e2 :.3f}\n")

        print(f"Initial beam x offset [um]:\t\t\t {drive_beam.x_offset()*1e6 :.3f} \t\t {main_beam.x_offset()*1e6 :.3f}")
        print(f"Initial beam y offset [um]:\t\t\t {drive_beam.y_offset()*1e6 :.3f} \t\t\t {main_beam.y_offset()*1e6 :.3f}")
        print(f"Initial beam z offset [um]:\t\t\t {drive_beam.z_offset()*1e6 :.3f} \t\t {main_beam.z_offset()*1e6 :.3f}\n")

        print(f"Initial beam x angular offset [urad]:\t\t {drive_beam.x_angle()*1e6 :.3f} \t\t\t {main_beam.x_angle()*1e6 :.3f}")
        print(f"Initial beam y angular offset [urad]:\t\t {drive_beam.y_angle()*1e6 :.3f} \t\t {main_beam.y_angle()*1e6 :.3f}\n")

        print(f"Initial normalised x emittance [mm mrad]:\t {drive_beam.norm_emittance_x()*1e6 :.3f} \t\t\t {main_beam.norm_emittance_x()*1e6 :.3f}")
        print(f"Initial normalised y emittance [mm mrad]:\t {drive_beam.norm_emittance_y()*1e6 :.3f} \t\t {main_beam.norm_emittance_y()*1e6 :.3f}")
        print(f"Initial angular momentum [mm mrad]:\t\t {drive_beam.angular_momentum()*1e6 :.3f} \t\t\t {main_beam.angular_momentum()*1e6 :.3f}\n")
        
        print(f"Initial matched beta function [mm]:\t\t {self.matched_beta_function(drive_beam.energy())*1e3 :.3f} \t\t {self.matched_beta_function(main_beam.energy())*1e3 :.3f}")
        print(f"Initial x beta function [mm]:\t\t\t {drive_beam.beta_x()*1e3 :.3f} \t\t {main_beam.beta_x()*1e3 :.3f}")
        print(f"Initial y beta function [mm]:\t\t\t {drive_beam.beta_y()*1e3 :.3f} \t\t {main_beam.beta_y()*1e3 :.3f}\n")

        print(f"Initial x beam size [um]:\t\t\t {drive_beam.beam_size_x()*1e6 :.3f} \t\t\t {main_beam.beam_size_x()*1e6 :.3f}")
        print(f"Initial y beam size [um]:\t\t\t {drive_beam.beam_size_y()*1e6 :.3f} \t\t\t {main_beam.beam_size_y()*1e6 :.3f}")
        print(f"Initial rms beam length [um]:\t\t\t {drive_beam.bunch_length()*1e6 :.3f} \t\t {main_beam.bunch_length()*1e6 :.3f}")
        print(f"Initial peak current [kA]:\t\t\t {drive_beam.peak_current()/1e3 :.3f} \t\t {main_beam.peak_current()/1e3 :.3f}")
        print(f"Bubble radius at beam head [um]:\t\t \t\t\t {self.rb_fit_obj(np.max(main_beam.zs()))*1e6 :.3f}")
        print(f"Bubble radius at beam tail [um]:\t\t \t\t\t {self.rb_fit_obj(np.min(main_beam.zs()))*1e6 :.3f}")
        print('-------------------------------------------------------------------------------------')
        
    
    # ==================================================
    def print_current_summary(self, initial_main_beam, beam_out, clean=False):

        if self.evolution is None:
            print('Beam parameter evolution has not been recorded.')
            return
        else:
            evol = self.evolution

        with open(self.run_path + 'output.txt', 'w') as f:
            print('============================================================================', file=f)
            print(f"Time step [betatron wavelength/c]:\t\t {self.time_step_mod :.3f}", file=f)
            print(f"Interstages enabled:\t\t\t\t {str(self.interstages_enabled) :s}", file=f)
            
            if self.interstage_dipole_field is None:
                print(f"Interstage dipole field:\t\t\t {'Not registered.' :s}", file=f)
            elif callable(self.interstage_dipole_field):
                interstage_dipole_field = self.interstage_dipole_field(beam_out.energy())
                print(f"Interstage dipole field:\t\t\t {interstage_dipole_field :.3f}", file=f)
            else:
                interstage_dipole_field = self.interstage_dipole_field
                print(f"Interstage dipole field:\t\t\t {interstage_dipole_field :.3f}", file=f)
            
            if self.main_source is None:
                print(f"Symmetrised main beam:\t\t\t\t Not registered", file=f)
            elif self.main_source.symmetrize:
                print(f"Symmetrised main beam:\t\t\t\t x, y, xp, yp symmetrised", file=f)
            elif self.main_source.symmetrize_6d:
                print(f"Symmetrised main beam:\t\t\t\t 6D symmetrised", file=f)
            else:
                print(f"Symmetrised main beam:\t\t\t\t Not symmetrised.", file=f)

            if self.driver_source.symmetrize:
                print(f"Symmetrised drive beam:\t\t\t\t x, y, xp, yp symmetrised\n", file=f)
            elif self.driver_source.symmetrize_6d:
                print(f"Symmetrised drive beam:\t\t\t\t 6D symmetrised\n", file=f)
            else:
                print(f"Symmetrised drive beam:\t\t\t\t Not symmetrised.\n", file=f)

            print(f"Ramp beta magnification:\t\t\t {self.ramp_beta_mag :.3f}", file=f)
            print(f"Transverse wake instability enabled:\t\t {str(self.enable_tr_instability) :s}", file=f)
            print(f"Radiation reaction enabled:\t\t\t {str(self.enable_radiation_reaction) :s}", file=f)
            print(f"Ion motion enabled:\t\t\t\t {str(self.enable_ion_motion) :s}", file=f)
            print(f"\tIon charge number:\t\t\t {self.ion_charge_num :.3f}", file=f)
            print(f"\tIon mass [u]:\t\t\t\t {self.ion_mass/SI.physical_constants['atomic mass constant'][0] :.3f}", file=f)
            print(f"\tnum_z_cells:\t\t\t\t {self.num_z_cells_main :d}", file=f)
            print(f"\tnum_x_cells_rft:\t\t\t {self.num_x_cells_rft :d}", file=f)
            print(f"\tnum_y_cells_rft:\t\t\t {self.num_y_cells_rft :d}", file=f)
            print(f"\tnum_xy_cells_probe:\t\t\t {self.num_xy_cells_probe :d}", file=f)
            print(f"\tion_wkfld_update_period\t\t\t {self.ion_wkfld_update_period :d}\n", file=f)
            
            print(f"Stage length [m]:\t\t\t\t {self.get_length() :.3f}", file=f)
            print(f"Propagation length [m]:\t\t\t\t {beam_out.location :.3f}", file=f)
            print(f"Drive beam to main beam efficiency [%]:\t\t {self.driver_to_beam_efficiency*100 :.3f}", file=f)
            print(f"Plasma density [m^-3]:\t\t\t\t {self.plasma_density :.3e}", file=f)
            print(f"Drive beam x jitter (std) [um]:\t\t\t {self.driver_source.jitter.x*1e6 :.3f}", file=f)
            print(f"Drive beam y jitter (std) [um]:\t\t\t {self.driver_source.jitter.y*1e6 :.3f}", file=f)
            print(f"Clean measured beam parameters:\t\t\t {str(clean) :s}", file=f)
            print('----------------------------------------------------------------------------\n', file=f)
            
            print('-------------------------------------------------------------------------------------', file=f)
            print('Quantity \t\t\t\t\t Drive beam \t\t Main beam', file=f)
            print('-------------------------------------------------------------------------------------', file=f)
            print(f"Initial number of macroparticles:\t\t {int(evol.driver.num_particles[0]) :d}\t\t\t {len(initial_main_beam.xs()) :d}", file=f)
            print(f"Current number of macroparticles:\t\t  \t\t\t {len(beam_out.xs()) :d}", file=f)
            print(f"Initial beam population:\t\t\t {(evol.driver.charge[0]/SI.e*np.sign(evol.driver.charge[0])) :.3e} \t\t {(np.sum(initial_main_beam.weightings())) :.3e}", file=f)
            print(f"Current beam population:\t\t\t \t \t\t {(np.sum(beam_out.weightings())) :.3e}\n", file=f)

            zs = beam_out.zs()
            indices = np.argsort(zs)
            zs_sorted = zs[indices]
            weights = beam_out.weightings()
            weights_sorted = weights[indices]
            
            print(f"Weighted main beam gradient [GV/m]:\t\t\t  \t\t {weighted_mean(self.Ez_fit_obj(zs_sorted), weights_sorted, clean=False)/1e9 :.7f}", file=f)
            #_, z_centre = find_closest_value_in_arr(arr=beam_out.zs(), val=beam_out.z_offset())  # Centre z of beam.
            #print(f"Beam centre gradient [GV/m]:\t\t\t\t  \t\t {self.Ez_fit_obj(z_centre)/1e9 :.3f}", file=f)
            print(f"Current mean gamma:\t\t\t\t \t \t\t {beam_out.gamma(clean=clean) :.3f}", file=f)
            print(f"Initial mean energy [GeV]:\t\t\t {evol.driver.energy[0]/1e9 :.3f} \t\t {initial_main_beam.energy(clean=clean)/1e9 :.3f}", file=f)
            print(f"Current mean energy [GeV]:\t\t\t \t \t\t {beam_out.energy(clean=clean)/1e9 :.3f}", file=f)
            print(f"Initial rms energy spread [%]:\t\t\t {evol.driver.rel_energy_spread[0]*1e2 :.3f} \t\t\t {initial_main_beam.rel_energy_spread(clean=clean)*1e2 :.3f}", file=f)
            print(f"Current rms energy spread [%]:\t\t\t  \t\t\t {beam_out.rel_energy_spread(clean=clean)*1e2 :.3f}", file=f)
    
            print(f"Initial beam x offset [um]:\t\t\t {evol.driver.x[0]*1e6 :.3f} \t\t {initial_main_beam.x_offset(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current beam x offset [um]:\t\t\t  \t\t\t {beam_out.x_offset(clean=clean)*1e6 :.3f}", file=f)
            print(f"Initial beam y offset [um]:\t\t\t {evol.driver.y[0]*1e6 :.3f} \t\t\t {initial_main_beam.y_offset(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current beam y offset [um]:\t\t\t  \t\t\t {beam_out.y_offset(clean=clean)*1e6 :.3f}", file=f)
            print(f"Initial beam z offset [um]:\t\t\t {evol.driver.z[0]*1e6 :.3f} \t\t {initial_main_beam.z_offset(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current beam z offset [um]:\t\t\t  \t\t\t {beam_out.z_offset(clean=clean)*1e6 :.3f}\n", file=f)

            print(f"Initial beam x angular offset [urad]:\t\t {evol.driver.x_angle[0]*1e6 :.3f} \t\t\t {initial_main_beam.x_angle(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current beam x angular offset [urad]:\t\t  \t\t\t {beam_out.x_angle(clean=clean)*1e6 :.3f}", file=f)
            print(f"Initial beam y angular offset [urad]:\t\t {evol.driver.y_angle[0]*1e6 :.3f} \t\t {initial_main_beam.y_angle(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current beam y angular offset [urad]:\t\t  \t\t\t {beam_out.y_angle(clean=clean)*1e6 :.3f}\n", file=f)

            print(f"Initial normalised x emittance [mm mrad]:\t {evol.driver.emit_nx[0]*1e6 :.3f} \t\t\t {initial_main_beam.norm_emittance_x(clean=False)*1e6 :.3f}", file=f)
            print(f"Current normalised x emittance [mm mrad]:\t  \t\t\t {beam_out.norm_emittance_x(clean=False)*1e6 :.3f}", file=f)
            print(f"Initial normalised y emittance [mm mrad]:\t {evol.driver.emit_ny[0]*1e6 :.3f} \t\t {initial_main_beam.norm_emittance_y(clean=False)*1e6 :.3f}", file=f)
            print(f"Current normalised y emittance [mm mrad]:\t \t \t\t {beam_out.norm_emittance_y(clean=False)*1e6 :.3f}\n", file=f)
            
            print(f"Initial cleaned norm. x emittance [mm mrad]:\t {evol.driver.emit_nx_clean[0]*1e6 :.3f} \t\t\t {initial_main_beam.norm_emittance_x(clean=True)*1e6 :.3f}", file=f)
            print(f"Current cleaned norm. x emittance [mm mrad]:\t  \t\t\t {beam_out.norm_emittance_x(clean=True)*1e6 :.3f}", file=f)
            print(f"Initial cleaned norm. y emittance [mm mrad]:\t {evol.driver.emit_ny_clean[0]*1e6 :.3f} \t\t {initial_main_beam.norm_emittance_y(clean=True)*1e6 :.3f}", file=f)
            print(f"Current cleaned norm. y emittance [mm mrad]:\t \t \t\t {beam_out.norm_emittance_y(clean=True)*1e6 :.3f}\n", file=f)
            
            #print(f"Initial angular momentum [mm mrad]:\t\t {drive_beam.angular_momentum()*1e6 :.3f} \t\t\t {initial_main_beam.angular_momentum()*1e6 :.3f}", file=f)
            print(f"Current angular momentum [mm mrad]:\t\t  \t\t\t {beam_out.angular_momentum()*1e6 :.3f}\n", file=f)
            
            print(f"Initial matched beta function [mm]:\t\t {self.matched_beta_function(evol.driver.energy[0])*1e3 :.3f} \t\t {self.matched_beta_function(initial_main_beam.energy(clean=clean))*1e3 :.3f}", file=f)
            print(f"Initial x beta function [mm]:\t\t\t {evol.driver.beta_x[0]*1e3 :.3f} \t\t {initial_main_beam.beta_x(clean=clean)*1e3 :.3f}", file=f)
            print(f"Current x beta function [mm]:\t\t\t \t \t\t {beam_out.beta_x(clean=clean)*1e3 :.3f}", file=f)
            print(f"Initial y beta function [mm]:\t\t\t {evol.driver.beta_y[0]*1e3 :.3f} \t\t {initial_main_beam.beta_y(clean=clean)*1e3 :.3f}", file=f)
            print(f"Current y beta function [mm]:\t\t\t \t \t\t {beam_out.beta_y(clean=clean)*1e3 :.3f}\n", file=f)
    
            print(f"Initial x beam size [um]:\t\t\t {evol.driver.beam_size_x[0]*1e6 :.3f} \t\t\t {initial_main_beam.beam_size_x(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current x beam size [um]:\t\t\t  \t\t\t {beam_out.beam_size_x(clean=clean)*1e6 :.3f}", file=f)
            print(f"Initial y beam size [um]:\t\t\t {evol.driver.beam_size_y[0]*1e6 :.3f} \t\t\t {initial_main_beam.beam_size_y(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current y beam size [um]:\t\t\t  \t\t\t {beam_out.beam_size_y(clean=clean)*1e6 :.3f}", file=f)
            print(f"Initial rms beam length [um]:\t\t\t {evol.driver.bunch_length[0]*1e6 :.3f} \t\t {initial_main_beam.bunch_length(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current rms beam length [um]:\t\t\t \t \t\t {beam_out.bunch_length(clean=clean)*1e6 :.3f}", file=f)
            #print(f"Initial peak current [kA]:\t\t\t {drive_beam.peak_current()/1e3 :.3f} \t\t {initial_main_beam.peak_current()/1e3 :.3f}", file=f)
            #print(f"Current peak current [kA]:\t\t\t  \t\t\t {beam_out.peak_current()/1e3 :.3f}", file=f)
            print(f"Bubble radius at beam head [um]:\t\t \t\t\t {self.rb_fit_obj(np.max(beam_out.zs()))*1e6 :.3f}", file=f)
            print(f"Bubble radius at beam tail [um]:\t\t \t\t\t {self.rb_fit_obj(np.min(beam_out.zs()))*1e6 :.3f}", file=f)
            print('-------------------------------------------------------------------------------------', file=f)
        f.close() # Close the file

        with open(self.run_path + 'output.txt', 'r') as f:
            print(f.read())
        f.close()


    # ==================================================
    def print_summary(self):
        
        print('Type: ', type(self))

        if self.plasma_density  is None:
            print(f"Plasma density [m^-3]:\t\t\t\t\t Not set")
        else:
            print(f"Plasma density [m^-3]:\t\t\t\t\t {self.plasma_density :.3e}")
        if self.length_flattop  is None:
            print(f"Stage flattop length [m]:\t\t\t\t Not set")
        else:
            print(f"Stage flattop length [m]:\t\t\t\t {self.length_flattop  :.3f}")
        if self.length  is None:
            print(f"Stage total length [m]:\t\t\t\t\t Not set")
        else:
            print(f"Stage total length [m]:\t\t\t\t\t {self.length  :.3f}")
        
        nom_energy_gain = self.nom_energy_gain 
        if nom_energy_gain is None:
            print(f"Nominal energy gain [GeV/m]:\t\t\t\t Not set")
        else:
            print(f"Nominal energy gain [GeV/m]:\t\t\t\t {nom_energy_gain/1e9 :.3f}")

        print(f"Time step [betatron wavelength/c]:\t\t\t {self.time_step_mod :.3f}")
        print(f"Ramp beta magnification:\t\t\t\t {self.ramp_beta_mag :.3f}")
        
        print(f"Transverse wake instability enabled:\t\t\t {str(self.enable_tr_instability) :s}")
        print(f"Radiation reaction enabled:\t\t\t\t {str(self.enable_radiation_reaction) :s}")
        print(f"Ion motion enabled:\t\t\t\t\t {str(self.enable_ion_motion) :s}")
        
        
        