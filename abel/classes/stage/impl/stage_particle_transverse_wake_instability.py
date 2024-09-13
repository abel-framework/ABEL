"""
Stage class with the transverse wake instability model as described in thesis "Instability and Beam-Beam Study for Multi-TeV PWFA e+e- and gamma gamma Linear Colliders" (https://cds.cern.ch/record/2754022?ln=en).

Ben Chen, 6 October 2023, University of Oslo
"""

import numpy as np
from scipy.constants import c, e, m_e, epsilon_0 as eps0
from scipy.interpolate import interp1d
import scipy.signal as signal

import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # For logarithmic colour scales
from matplotlib.colors import LinearSegmentedColormap  # For customising colour maps
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For manipulating colourbars

from joblib import Parallel, delayed  # Parallel tracking
from joblib_progress import joblib_progress  # TODO: remove
from types import SimpleNamespace
import os, copy, warnings

from abel.physics_models.particles_transverse_wake_instability import *
from abel.physics_models.twoD_particles_transverse_wake_instability import *  # TODO: remove
from abel.utilities.plasma_physics import k_p, beta_matched, wave_breaking_field, blowout_radius
from abel.utilities.other import find_closest_value_in_arr
from abel.classes.stage.impl.stage_wake_t import StageWakeT
from abel import Stage, CONFIG
from abel import Beam



class StagePrtclTransWakeInstability(Stage):

    # ==================================================
    def __init__(self, driver_source=None, main_source=None, drive_beam=None, main_beam=None, length=None, nom_energy_gain=None, plasma_density=None, time_step_mod=0.05, show_prog_bar=False, Ez_fit_obj=None, Ez_roi=None, rb_fit_obj=None, bubble_radius_roi=None, ramp_beta_mag=1.0, probe_data_frac=None, enable_tr_instability=True, enable_radiation_reaction=True, enable_ion_motion=False, ion_charge_num=1.0, ion_mass=None, num_z_cells_main=None, num_x_cells_rft=50, num_y_cells_rft=50, num_xy_cells_probe=41, uniform_z_grid=False, update_factor=1.0):
        """
        Parameters
        ----------
        driver_source: Source object of drive beam.
        
        main_source: Source object of main beam.

        driver_beam: Beam object of drive beam.

        main_beam: Beam object of main beam.
        
        length: [m] float
            Length of the plasma stage.
        
        nom_energy_gain: [eV] float
            Nominal/target energy gain of the acceleration stage.
        
        plasma_density: [m^-3] float
            Plasma density.

        time_step_mod: [beta_wave_length/c] float
            Determines the time step of the instability tracking in units of beta_wave_length/c.
            
        Ez_fit_obj: [V/m] interpolation object
            1D interpolation object of longitudinal E-field fitted to axial E-field using a selection of zs along the main beam. Used to determine the value of the longitudinal E-field for all beam zs.

        Ez_roi: [V/m] 1D array
            Longitudinal E-field in the region of interest fitted to a selection of zs along the main beam (main beam head to tail).

        rb_fit_obj: [m] interpolation object?
            1D interpolation object of plasma bubble radius fitted to axial bubble radius using a selection of zs along the main beam. Used to determine the value of the bubble radius for all beam zs.
        
        bubble_radius_roi: [m] 1D array
            The bubble radius in the region of interest fitted to a selection of zs along the main beam.

        ramp_beta_mag: float
            Used for demagnifying and magnifying beams passing through entrance and exit plasma ramps.

        enable_radiation_reaction: bool
            Flag for enabling radiation reactions.

        ...
        """
        
        super().__init__(length, nom_energy_gain, plasma_density)

        self.driver_source = driver_source
        self.main_source = main_source
        self.drive_beam = drive_beam

        self.time_step_mod = time_step_mod  # Determines the time step of the instability tracking in units of beta_wave_length/c.
        self.interstage_dipole_field = None
        self.ramp_beta_mag = ramp_beta_mag
        
        self.enable_tr_instability = enable_tr_instability 
        self.enable_radiation_reaction = enable_radiation_reaction

        # Ion motion parameters
        self.enable_ion_motion = enable_ion_motion
        self.ion_charge_num = ion_charge_num
        self.ion_mass = ion_mass
        
        self.num_z_cells_main = num_z_cells_main
        self.num_x_cells_rft = num_x_cells_rft
        self.num_y_cells_rft = num_y_cells_rft
        self.num_xy_cells_probe = num_xy_cells_probe
        self.uniform_z_grid = uniform_z_grid
        self.update_factor = np.max([time_step_mod, update_factor])
        
        self.Ez_fit_obj = Ez_fit_obj  # [V/m] 1d interpolation object of longitudinal E-field fitted to Ez_axial using a selection of zs along the main beam.
        self.Ez_roi = Ez_roi  # [V/m] longitudinal E-field in the region of interest (main beam head to tail).
        #self.Ez_axial = None  # Moved to self.initial.plasma.wakefield.onaxis.Ezs
        #self.zs_Ez_axial = None  # Moved to self.initial.plasma.wakefield.onaxis.zs
        self.rb_fit_obj = rb_fit_obj  # [m] 1d interpolation object of bubble radius fitted to bubble_radius_axial using a selection of zs along the main beam.
        self.bubble_radius_roi = bubble_radius_roi  # [m] bubble radius in the region of interest.
        self.bubble_radius_axial = None
        self.zs_bubble_radius_axial = None
        
        self.main_num_profile = None
        self.z_slices = None
        self.main_slices_edges = None

        self.probe_data_frac = probe_data_frac
        self.diag_path = None
        self.interstages_enabled = False
        self.show_prog_bar = show_prog_bar
        self.parallel_track_2D = False
        
        self.driver_to_wake_efficiency = None
        self.wake_to_beam_efficiency = None
        self.driver_to_beam_efficiency = None
        
        self.reljitter = SimpleNamespace()
        self.reljitter.plasma_density = 0

        # internally sampled values (given some jitter)
        self.__n = None 
        self.driver_initial = None

    
    # ==================================================
    # Track the particles through. Note that when called as part of a Linac object, a copy of the original stage (where no changes has been made) is sent to track() every time. All changes done to self here are saved to a separate stage under the Linac object.
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        # Set the diagnostics directory
        self.diag_path = runnable.run_path()

        # Set the flag for displaying progress bar
        self.show_prog_bar = verbose

        # Extract quantities from the stage
        plasma_density = self.plasma_density
        stage_length = self.length
        gamma0 = beam0.gamma()
        time_step_mod = self.time_step_mod

        particle_mass = beam0.particle_mass

        
        # ========== Get the drive beam ==========
        if self.driver_source.jitter.x == 0 and self.driver_source.jitter.y == 0 and self.drive_beam is not None:                   #############################
            drive_beam0 = self.drive_beam  # This guarantees zero drive beam jitter between stages, as identical drive beams are used in every stage and not re-sampled.
        else:
            drive_beam0 = self.driver_source.track()
            self.drive_beam = drive_beam0  # Generate a drive beam with jitter.                   ############################# 
        
        beam0_copy = copy.deepcopy(beam0)  # Make a deep copy of beam0 for use in StageWakeT, which applies magnify_beta_function() separately.
        drive_beam_ramped = copy.deepcopy(drive_beam0)  # Make a deep copy to not affect the original drive beam.

        
        # ========== Apply plasma density up ramp (demagnify beta function) ========== 
        drive_beam_ramped.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=drive_beam0)
        beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=drive_beam0)
        
        # Number profile N(z). Dimensionless, same as dN/dz with each bin multiplied with the widths of the bins.
        main_num_profile, z_slices = self.longitudinal_number_distribution(beam=beam0)
        self.z_slices = z_slices  # Update the longitudinal position of the beam slices needed to fit Ez and bubble radius.
        self.main_num_profile = main_num_profile
        

        # ========== Wake-T simulation and extraction ==========
        # Define a Wake-T stage
        stage_wakeT = StageWakeT()
        #stage_wakeT.driver_source = self.driver_source
        stage_wakeT.drive_beam = drive_beam0
        k_beta = k_p(plasma_density)/np.sqrt(2*min(gamma0, drive_beam0.gamma()/2))
        lambda_betatron = (2*np.pi/k_beta)
        stage_wakeT.length = lambda_betatron/10  # [m]
        stage_wakeT.plasma_density = plasma_density  # [m^-3]
        stage_wakeT.ramp_beta_mag = self.ramp_beta_mag
        #stage_wakeT.keep_data = False  # TODO: Does not work yet.
        
        # Run the Wake-T stage
        beam_wakeT = stage_wakeT.track(beam0_copy)
        
        # Read the Wake-T simulation data
        Ez_axis_wakeT = stage_wakeT.initial.plasma.wakefield.onaxis.Ezs
        zs_Ez_wakeT = stage_wakeT.initial.plasma.wakefield.onaxis.zs
        rho = stage_wakeT.initial.plasma.density.rho*-e
        plasma_num_density = stage_wakeT.initial.plasma.density.rho/stage_wakeT.plasma_density
        info_rho = stage_wakeT.initial.plasma.density.metadata
        zs_rho = info_rho.z
        rs_rho = info_rho.r
        
        # Cut out axial Ez over the ROI
        Ez, Ez_fit = self.Ez_shift_fit(Ez_axis_wakeT, zs_Ez_wakeT, beam0, z_slices)
        
        # Extract the plasma bubble radius
        #driver_x_offset = drive_beam_ramped.x_offset()
        #driver_y_offset = drive_beam_ramped.y_offset()
        #x_offset = beam0.x_offset()             # Important to NOT use beam0_copy.
        #bubble_radius_wakeT = self.get_bubble_radius(plasma_num_density, rs_rho, driver_x_offset, threshold=0.8)
        bubble_radius_wakeT = self.get_bubble_radius_WakeT(plasma_num_density, rs_rho, threshold=0.8)

        # Cut out bubble radius over the ROI
        bubble_radius_roi, rb_fit = self.rb_shift_fit(bubble_radius_wakeT, zs_rho, beam0, z_slices) # TODO: Actually same as Ez_shift_fit. Consider making just one function instead...

        if bubble_radius_wakeT.max() < 0.5 * blowout_radius(self.plasma_density, drive_beam_ramped.peak_current()) or bubble_radius_roi.any()==0:
            warnings.warn("The bubbel radius may not have been correctly extracted.", UserWarning)

        idxs_bubble_peaks, _ = signal.find_peaks(bubble_radius_roi, height=None, width=1, prominence=0.1)
        if idxs_bubble_peaks.size > 0:
            warnings.warn("The bubbel radius may not be smooth.", UserWarning)

        # Save quantities to the stage
        self.Ez_fit_obj = Ez_fit
        self.rb_fit_obj = rb_fit
        self.__save_initial_step(Ez0_axial=Ez_axis_wakeT, zs_Ez0=zs_Ez_wakeT, rho0=rho, metadata_rho0=info_rho, driver0=drive_beam_ramped, beam0=beam0)
        
        self.Ez_roi = Ez
        #self.Ez_axial = Ez_axis_wakeT  # Moved to self.initial.plasma.wakefield.onaxis.Ezs
        #self.zs_Ez_axial = zs_Ez_wakeT  # Moved to self.initial.plasma.wakefield.onaxis.zs
        self.bubble_radius_roi = bubble_radius_roi
        self.bubble_radius_axial = bubble_radius_wakeT
        self.zs_bubble_radius_axial = zs_rho
        
        # Make plots for control if necessary
        #self.plot_Ez_rb_cut(z_slices, main_num_profile, zs_Ez_wakeT, Ez_axis_wakeT, Ez, zs_rho, bubble_radius_wakeT, bubble_radius_roi, zlab=r'$z$ [$\mathrm{\mu}$m]')
        

        # ========== Instability tracking ==========
        beam_filtered = self.bubble_filter(copy.deepcopy(beam0), sort_zs=True)

        if self.num_z_cells_main is None:
            self.num_z_cells_main = round(np.sqrt( len(drive_beam_ramped)+len(beam_filtered) )/2)
            
        trans_wake_config = PrtclTransWakeConfig(
            plasma_density=self.plasma_density, 
            stage_length=self.length, 
            drive_beam=drive_beam_ramped, 
            main_beam=beam_filtered, 
            time_step_mod=self.time_step_mod, 
            show_prog_bar=self.show_prog_bar, 
            shot_path=runnable.shot_path(), 
            stage_num=beam0.stage_number, 
            probe_data_frac=self.probe_data_frac, 
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
            update_factor=self.update_factor
        )
        
        inputs = [drive_beam_ramped, beam_filtered, trans_wake_config.plasma_density, Ez_fit, rb_fit, trans_wake_config.stage_length, trans_wake_config.time_step_mod]
        some_are_none = any(input is None for input in inputs)
        
        if some_are_none:
            none_indices = [i for i, x in enumerate(inputs) if x is None]
            print(none_indices)
            raise ValueError('At least one input is set to None.')
        
        
        if self.parallel_track_2D is True:
            
            with joblib_progress('Tracking x and y in parallel:', 2):
                results = Parallel(n_jobs=2)([
                    delayed(twoD_transverse_wake_instability_particles)(beam_filtered, beam_filtered.xs(), beam_filtered.pxs(), plasma_density, Ez_fit, rb_fit, stage_length, time_step_mod, get_centroids=False, s_slices=None, z_slices=None),
                    delayed(twoD_transverse_wake_instability_particles)(beam_filtered, beam_filtered.ys(), beam_filtered.pys(), plasma_density, Ez_fit, rb_fit, stage_length, time_step_mod, get_centroids=False, s_slices=None, z_slices=None)
                ])
                time.sleep(0.1) # hack to allow printing progress
            
            xs_sorted, pxs_sorted, zs_sorted, pzs_sorted, weights_sorted, s_slices_table, offset_slices_table, angle_slices_table = results[0]
            ys_sorted, pys_sorted, _, _, _, _, _, _ = results[1]

            # Initialise ABEL Beam object
            beam = Beam()
            
            # Set the phase space of the ABEL beam
            beam.set_phase_space(Q=np.sum(weights_sorted)*beam0.charge_sign()*e,
                                 xs=xs_sorted,
                                 ys=ys_sorted,
                                 zs=zs_sorted, 
                                 pxs=pxs_sorted,  # Always use single particle momenta?
                                 pys=pys_sorted,
                                 pzs=pzs_sorted,
                                 weightings=weights_sorted,
                                 particle_mass=particle_mass)
            
        else:
            
            beam = transverse_wake_instability_particles(beam_filtered, drive_beam_ramped, Ez_fit_obj=Ez_fit, rb_fit_obj=rb_fit, trans_wake_config=trans_wake_config)
            
        
        # ==========  Apply plasma density down ramp (magnify beta function) ========== 
        drive_beam_ramped.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=drive_beam0)
        beam.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=drive_beam0)
        

        # ========== Bookkeeping ========== 
        self.driver_to_beam_efficiency = (beam.energy()-beam0.energy())/drive_beam_ramped.energy() * beam.abs_charge()/drive_beam_ramped.abs_charge()
        self.main_beam = copy.deepcopy(beam)  # Need to make a deepcopy, or changes to beam may affect the Beam object saved here.
        
        # Copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        return super().track(beam, savedepth, runnable, verbose)


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
        projection_zx = np.sum(particle_density, axis=2)
        projection_zx = projection_zx.T
        extent_beams = np.array([edges_z[0], edges_z[-1], edges_x[0], edges_x[-1]])
        #TODO: projection_zy?

        self.initial.beam.density.extent = extent_beams  # array([z_min, z_max, x_min, x_max])
        self.initial.beam.density.rho = projection_zx
        
        # ========== Save initial beam currents ==========
        self.calculate_beam_current(beam0, driver0)

    
    # ==================================================
    def __extract_evolution(self, tmpfolder, beam0, runnable):
       insitu_path = tmpfolder + 'diags/insitu/'
       
       if not self.driver_only:
           
           insitu_file = insitu_path + 'reduced_beam.*.txt'
               
           # extract in-situ data
           all_data = read_insitu_diagnostics.read_file(insitu_file)
           average_data = all_data['average']
           
           # store variables
           self.evolution.location = beam0.location + all_data['time']*SI.c
           self.evolution.charge = read_insitu_diagnostics.total_charge(all_data)
           self.evolution.energy = read_insitu_diagnostics.energy_mean_eV(all_data)
           self.evolution.z = average_data['[z]']
           self.evolution.x = average_data['[x]']
           self.evolution.y = average_data['[y]']
           self.evolution.xp = average_data['[ux]']/average_data['[uz]']
           self.evolution.yp = average_data['[uy]']/average_data['[uz]']
           self.evolution.energy_spread = read_insitu_diagnostics.energy_spread_eV(all_data)
           self.evolution.rel_energy_spread = self.evolution.energy_spread/self.evolution.energy
           self.evolution.beam_size_x = read_insitu_diagnostics.position_std(average_data, direction='x')
           self.evolution.beam_size_y = read_insitu_diagnostics.position_std(average_data, direction='y')
           self.evolution.bunch_length = read_insitu_diagnostics.position_std(average_data, direction='z')
           self.evolution.emit_nx = read_insitu_diagnostics.emittance_x(average_data)
           self.evolution.emit_ny = read_insitu_diagnostics.emittance_y(average_data)
           # TODO: add angular momentum and normalized amplitude
       # delete or move data
       #if not self.keep_data:
           #destination_path = runnable.shot_path() + 'stage_' + str(beam0.stage_number) + '/insitu'
           #shutil.move(insitu_path, destination_path)
           # Delete data


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
        Cuts out the longitudinal axial E-field Ez over the beam region and makes a fit using the z-coordinates for the region.

        Parameters
        ----------
        Ez: [V/m] 1D array
            Axial longitudinal E-field.
            
        zs_Ez: [m] 1D float array
            zs_Ez-coordinates for Ez. Monotonically increasing from first to last element.

        beam: ABEL Beam object
            
        z_slices: [m] 1D float array
            Co-moving coordinates of the beam slices.

            
        Returns
        ----------
        Ez_fit(z_slices): [V/m] 1D array
            Axial Ez for the region of interest shifted to the location of the beam.

        Ez_fit: [V/m] 1D interpolation object 
            Interpolated axial longitudinal Ez from beam head to tail.
        """
        
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
    def get_bubble_radius(self, plasma_num_density, plasma_tr_coord, driver_offset, threshold=0.8):
        """
        - For extracting the plasma ion bubble radius by finding the coordinates in which the plasma number density goes from zero to a threshold value.
        - The symmetry axis is determined using the transverse offset of the drive beam.
        - xi is the propagation direction pointing to the right.
        
        Parameters
        ----------
        plasma_num_density: [n0] 2D float array
            Plasma number density in units of initial number density n0. Need to be oriented with propagation direction pointing to the right and positive offset pointing upwards.
            
        plasma_tr_coord: [m] 1D float array 
            Transverse coordinate of plasma_num_density. Needs to be strictly growing from start to end.

        driver_offset: [m] float
            Mean transverse offset of the drive beam.
            
        threshold: float
            Defines a threshold for the plasma density to determine bubble_radius.

            
        Returns
        ----------
        bubble_radius: [m] 1D float array 
            Plasma bubble radius over the simulation box.
        """

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
        
        return bubble_radius


    # ==================================================
    def get_bubble_radius_WakeT(self, plasma_num_density, plasma_tr_coord, threshold=0.8):
        """
        The plasma wake calculated by Wake-T is always centered around r = 0.0, so that driver_offset=0.0 are used as inputs in get_bubble_radius().
        """
        bubble_radius = self.get_bubble_radius(plasma_num_density=plasma_num_density, plasma_tr_coord=plasma_tr_coord, driver_offset=0.0, threshold=0.8)

        return bubble_radius

    
    # ==================================================
    def rb_shift_fit(self, rb, zs_rb, beam, z_slices=None):
        """
        Cuts out the bubble radius over the beam region and makes a fit using the z-coordinates for the region.

        Parameters
        ----------
        rb: [m] 1D array
            Plasma ion bubble radius.
            
        zs_rb: [m] 1D float array
            z-coordinates for rb. Monotonically increasing from first to last element.

        beam: ABEL Beam object
            
        z_slices: [m] 1D float array
            Co-moving coordinates of the beam slices.

            
        Returns
        ----------
        rb_fit(z_slices): [m] 1D float array
            Plasma ion bubble radius for the region of interest shifted to the location of the beam.

        rb_fit: [V/m] 1D interpolation object 
            Interpolated axial longitudinal Ez from beam head to tail.
        """
        
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
        
        # Calculate sum of squared errors (sse)
        sse_rb = np.sum((rb_cut - rb_fit(zs_cut))**2)
        
        if sse_rb/np.mean(rb_cut) > 0.05:
            warnings.warn('The plasma bubble radius may not have been accurately fitted.\n', UserWarning)
        
        return rb_fit(z_slices), rb_fit    
        

    # ==================================================
    # Determine the number of beam slices based on the Freedman–Diaconis rule
    def FD_rule_num_slice(self, zs=None):
        if zs is None:
            zs = self.main_beam.zs()
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
    def get_length(self):
        return self.length

    
    # ==================================================
    def get_nom_energy_gain(self):
        return self.nom_energy_gain

    
    # ==================================================
    def set_nom_energy_gain(self, nom_energy_gain):
        self.nom_energy_gain = nom_energy_gain

    
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
    def __get_initial_driver(self, resample=False):
        if resample or self.driver_initial is None:
            self.driver_initial = self.driver_source.track()
        return self.driver_initial

    
    # ==================================================
    def __get_plasma_density(self, resample=False):
        if resample or self.__n is None:
            self.__n = self.plasma_density * np.random.normal(loc = 1, scale = self.reljitter.plasma_density)
        return self.__n
        

    # ==================================================
    # Overloads the plot_wakefield method in the Stage class.
    def plot_wakefield(self, beam=None, saveToFile=None, includeWakeRadius=True):
        
        # Get wakefield
        #Ezs = self.Ez_axial
        #zs_Ez = self.zs_Ez_axial
        Ezs = self.initial.plasma.wakefield.onaxis.Ezs
        zs_Ez = self.initial.plasma.wakefield.onaxis.zs
        zs_rho =  self.zs_bubble_radius_axial
        bubble_radius = self.bubble_radius_axial
        
        # get current profile
        Is = self.initial.beam.current.Is
        zs0 = self.initial.beam.current.zs
        
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
        axs[0].set_xlabel(r'z [$\mathrm{\mu}$m]')
        axs[0].set_ylabel('Longitudinal electric field [GV/m]')
        axs[0].set_xlim(zlims)
        axs[0].set_ylim(bottom=1.05*min(Ezs)/1e9, top=1.3*max(Ezs)/1e9)
        
        axs[1].fill(np.concatenate((zs0, np.flip(zs0)))*1e6, np.concatenate((-Is, np.zeros(Is.shape)))/1e3, color=col1, alpha=af)
        axs[1].plot(zs0*1e6, -Is/1e3, '-', color=col1)
        axs[1].set_xlabel(r'z [$\mathrm{\mu}$m]')
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
        
        # save to file
        if saveToFile is not None:
            plt.savefig(saveToFile, format="pdf", bbox_inches="tight")


    # ==================================================
    # Overloads the plot_wake method in the Stage class.
    def plot_wake(self, savefig=None):
        
        # extract density if not already existing
        assert hasattr(self.initial.plasma.density, 'rho'), 'No wake'
        assert hasattr(self.initial.plasma.wakefield.onaxis, 'Ezs'), 'No wakefield'
        
        # calculate densities and extents
        Ezmax = 0.8*wave_breaking_field(self.plasma_density)
        
        # make figures
        has_final_step = hasattr(self.final.plasma.density, 'rho')
        num_plots = 1 + int(has_final_step)
        fig, ax = plt.subplots(num_plots,1)
        fig.set_figwidth(CONFIG.plot_width_default*0.7)
        fig.set_figheight(CONFIG.plot_width_default*0.5*num_plots)

        # cycle through initial and final step
        for i in range(num_plots):
            if not has_final_step:
                ax1 = ax
            else:
                ax1 = ax[i]

            # extract initial or final
            if i==0:
                data_struct = self.initial
                title = 'Initial step'
            elif i==1:
                data_struct = self.final
                title = 'Final step'

            # get data
            extent = data_struct.plasma.density.extent
            zs0 = data_struct.plasma.wakefield.onaxis.zs
            Ezs0 = data_struct.plasma.wakefield.onaxis.Ezs
            rho0_plasma = data_struct.plasma.density.rho
            rho0_beam = data_struct.beam.density.rho

            # plot on-axis wakefield and axes
            zlims = [min(zs0)*1e6, max(zs0)*1e6]
            ax2 = ax1.twinx()
            ax2.plot(zs0*1e6, Ezs0/1e9, color='black')
            ax2.set_ylabel(r'$E_{z}$' ' [GV/m]')
            ax2.set_xlim(zlims)
            #ax2.set_ylim(bottom=-Ezmax/1e9, top=Ezmax/1e9)
            axpos = ax1.get_position()
            pad_fraction = 0.15  # Fraction of the figure width to use as padding between the ax and colorbar
            cbar_width_fraction = 0.03  # Fraction of the figure width for the colorbar width
    
            # create colorbar axes based on the relative position and size
            cax1 = fig.add_axes([axpos.x1 + pad_fraction, axpos.y0, cbar_width_fraction, axpos.height])
            cax2 = fig.add_axes([axpos.x1 + pad_fraction + cbar_width_fraction, axpos.y0, cbar_width_fraction, axpos.height])
            clims = np.array([1e-2, 1e3])*self.plasma_density
            
            # plot plasma electrons
            initial = ax1.imshow(rho0_plasma/1e6, extent=extent*1e6, norm=LogNorm(), origin='lower', cmap='Blues', alpha=np.array(rho0_plasma>clims.min()*2, dtype=float))
            cb = plt.colorbar(initial, cax=cax1)
            initial.set_clim(clims/1e6)
            cb.ax.tick_params(axis='y',which='both', direction='in')
            cb.set_ticklabels([])
            
            # plot beam electrons
            charge_density_plot0 = ax1.imshow(rho0_beam/1e6, extent=data_struct.beam.density.extent*1e6, norm=LogNorm(), origin='lower', cmap=CONFIG.default_cmap, alpha=np.array(rho0_beam>clims.min()*2, dtype=float))
            cb2 = plt.colorbar(charge_density_plot0, cax = cax2)
            cb2.set_label(label=r'Electron density ' + r'[$\mathrm{cm^{-3}}$]',size=10)
            cb2.ax.tick_params(axis='y',which='both', direction='in')
            charge_density_plot0.set_clim(clims/1e6)
    
            # set labels
            if i==(num_plots-1):
                ax1.set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            ax1.set_ylabel(r'$x$ [$\mathrm{\mu}$m]')
            ax1.set_title(title)
            ax1.grid(False)
            ax2.grid(False)
            
        # save the figure
        if savefig is not None:
            fig.savefig(str(savefig), bbox_inches='tight', dpi=1000)
        
        return 
        
    
    # ==================================================
    def imshow_plot(self, data, axes=None, extent=None, vmin=None, vmax=None, colmap='seismic', xlab=r'$\xi$ [$\mathrm{\mu}$m]', ylab=r'$x$ [$\mathrm{\mu}$m]', clab='', gridOn=False, origin='lower', interpolation=None, aspect='auto', log_cax=False, reduce_cax_pad=False):
        
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
            divider = make_axes_locatable(ax)
            cbar_ax = divider.append_axes("right", size="5%", pad=0.05)

        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()

        # Make a 2D plot
        if log_cax is True:
            p = ax.imshow(data, extent=extent, cmap=plt.get_cmap(colmap), origin=origin, aspect=aspect, interpolation=interpolation, norm=colors.LogNorm(vmin+1, vmax))
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
    def distribution_plot_2D(self, arr1, arr2, weights=None, hist_bins=None, hist_range=None, axes=None, extent=None, vmin=None, vmax=None, colmap=CONFIG.default_cmap, xlab='', ylab='', clab='', origin='lower', interpolation='nearest', reduce_cax_pad=False):

        if weights is None:
            weights = self.main_beam.weightings()
        if hist_bins is None:
            nbins = int(np.sqrt(len(arr1)/2))
            hist_bins = [ nbins, nbins ]  # list of 2 ints. Number of bins along each direction, for the histograms
        if hist_range is None:
            hist_range = [[None, None], [None, None]]
            hist_range[0] = [ arr1.min(), arr1.max() ]  # List contains 2 lists of 2 floats. Extent of the histogram along each direction
            hist_range[1] = [ arr2.min(), arr2.max() ]
        if extent is None:
            extent = hist_range[0] + hist_range[1]
        
        binned_data, zedges, xedges = np.histogram2d(arr1, arr2, hist_bins, hist_range, weights=weights)
        beam_hist2d = binned_data.T/np.diff(zedges)/np.diff(xedges)
        self.imshow_plot(beam_hist2d, axes=axes, extent=extent, vmin=vmin, vmax=vmax, colmap=colmap, 
                  xlab=xlab, ylab=ylab, clab=clab, gridOn=False, origin=origin, interpolation=interpolation, reduce_cax_pad=reduce_cax_pad)

    
    # ==================================================
    def density_map_diags(self, beam=None, plot_centroids=False, save_plots=True):
        
        #colors = ['white', 'aquamarine', 'lightgreen', 'green']
        #colors = ['white', 'forestgreen', 'limegreen', 'lawngreen', 'aquamarine', 'deepskyblue']
        #bounds = [0, 0.2, 0.4, 0.8, 1]
        #cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=256)
        
        cmap = CONFIG.default_cmap

        if beam is None:
            beam = self.main_beam

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
        xps_lab = '$x\'$ [mrad]'
        yps_lab = '$y\'$ [mrad]'
        energ_lab = r'$\mathcal{E}$ [GeV]'
        
        # Set up a figure with axes
        fig, axs = plt.subplots(nrows=3, ncols=3, layout='constrained', figsize=(5*3, 4*3))
        fig.suptitle(r'$\Delta s=$' f'{format(beam.location, ".2f")}' ' m')

        nbins = int(np.sqrt(len(weights)/2))
        hist_bins = [ nbins, nbins ]  # list of 2 ints. Number of bins along each direction, for the histograms

        # 2D z-x distribution
        hist_range = [[None, None], [None, None]]
        hist_range[0] = [ zs.min(), zs.max() ]  # [m], list contains 2 lists of 2 floats. Extent of the histogram along each direction
        hist_range[1] = [ xs.min(), xs.max() ]
        extent_zx = hist_range[0] + hist_range[1]
        extent_zx = [i*1e6 for i in extent_zx]  # [um]

        self.distribution_plot_2D(arr1=zs, arr2=xs, weights=weights, hist_bins=hist_bins, hist_range=hist_range, axes=axs[0][0], extent=extent_zx, vmin=None, vmax=None, colmap=cmap, xlab=xilab, ylab=xlab, clab=r'$\partial^2 N/\partial\xi \partial x$ [$\mathrm{m}^{-2}$]', origin='lower', interpolation='nearest')
        

        # 2D z-x' distribution
        hist_range_xps = [[None, None], [None, None]]
        hist_range_xps[0] = hist_range[0]
        hist_range_xps[1] = [ xps.min(), xps.max() ]  # [rad]
        extent_xps = hist_range_xps[0] + hist_range_xps[1]
        extent_xps[0] = extent_xps[0]*1e6  # [um]
        extent_xps[1] = extent_xps[1]*1e6  # [um]
        extent_xps[2] = extent_xps[2]*1e3  # [mrad]
        extent_xps[3] = extent_xps[3]*1e3  # [mrad]

        self.distribution_plot_2D(arr1=zs, arr2=xps, weights=weights, hist_bins=hist_bins, hist_range=hist_range_xps, axes=axs[0][1], extent=extent_xps, vmin=None, vmax=None, colmap=cmap, xlab=xilab, ylab=xps_lab, clab='$\partial^2 N/\partial z \partial x\'$ [$\mathrm{m}^{-1}$ $\mathrm{rad}^{-1}$]', origin='lower', interpolation='nearest')
        
        
        # 2D x-x' distribution
        hist_range_xxp = [[None, None], [None, None]]
        hist_range_xxp[0] = hist_range[1]
        hist_range_xxp[1] = [ xps.min(), xps.max() ]  # [rad]
        extent_xxp = hist_range_xxp[0] + hist_range_xxp[1]
        extent_xxp[0] = extent_xxp[0]*1e6  # [um]
        extent_xxp[1] = extent_xxp[1]*1e6  # [um]
        extent_xxp[2] = extent_xxp[2]*1e3  # [mrad]
        extent_xxp[3] = extent_xxp[3]*1e3  # [mrad]

        self.distribution_plot_2D(arr1=xs, arr2=xps, weights=weights, hist_bins=hist_bins, hist_range=hist_range_xxp, axes=axs[0][2], extent=extent_xxp, vmin=None, vmax=None, colmap=cmap, xlab=xlab, ylab=xps_lab, clab='$\partial^2 N/\partial x\partial x\'$ [$\mathrm{m}^{-1}$ $\mathrm{rad}^{-1}$]', origin='lower', interpolation='nearest')
        

        # 2D z-y distribution
        hist_range_zy = [[None, None], [None, None]]
        hist_range_zy[0] = hist_range[0]
        hist_range_zy[1] = [ ys.min(), ys.max() ]
        extent_zy = hist_range_zy[0] + hist_range_zy[1]
        extent_zy = [i*1e6 for i in extent_zy]  # [um]

        self.distribution_plot_2D(arr1=zs, arr2=ys, weights=weights, hist_bins=hist_bins, hist_range=hist_range_zy, axes=axs[1][0], extent=extent_zy, vmin=None, vmax=None, colmap=cmap, xlab=xilab, ylab=ylab, clab=r'$\partial^2 N/\partial\xi \partial y$ [$\mathrm{m}^{-2}$]', origin='lower', interpolation='nearest')
        

        # 2D z-y' distribution
        hist_range_yps = [[None, None], [None, None]]
        hist_range_yps[0] = hist_range[0]
        hist_range_yps[1] = [ yps.min(), yps.max() ]  # [rad]
        extent_yps = hist_range_yps[0] + hist_range_yps[1]
        extent_yps[0] = extent_yps[0]*1e6  # [um]
        extent_yps[1] = extent_yps[1]*1e6  # [um]
        extent_yps[2] = extent_yps[2]*1e3  # [mrad]
        extent_yps[3] = extent_yps[3]*1e3  # [mrad]
        
        self.distribution_plot_2D(arr1=zs, arr2=yps, weights=weights, hist_bins=hist_bins, hist_range=hist_range_yps, axes=axs[1][1], extent=extent_yps, vmin=None, vmax=None, colmap=cmap, xlab=xilab, ylab=yps_lab, clab='$\partial^2 N/\partial z \partial y\'$ [$\mathrm{m}^{-1}$ $\mathrm{rad}^{-1}$]', origin='lower', interpolation='nearest')
        

        # 2D y-y' distribution
        hist_range_yyp = [[None, None], [None, None]]
        hist_range_yyp[0] = hist_range_zy[1]
        hist_range_yyp[1] = [ yps.min(), yps.max() ]  # [rad]
        extent_yyp = hist_range_yyp[0] + hist_range_yyp[1]
        extent_yyp[0] = extent_yyp[0]*1e6  # [um]
        extent_yyp[1] = extent_yyp[1]*1e6  # [um]
        extent_yyp[2] = extent_yyp[2]*1e3  # [mrad]
        extent_yyp[3] = extent_yyp[3]*1e3  # [mrad]
        
        self.distribution_plot_2D(arr1=ys, arr2=yps, weights=weights, hist_bins=hist_bins, hist_range=hist_range_yyp, axes=axs[1][2], extent=extent_yyp, vmin=None, vmax=None, colmap=cmap, xlab=ylab, ylab=yps_lab, clab='$\partial^2 N/\partial y\partial y\'$ [$\mathrm{m}^{-1}$ $\mathrm{rad}^{-1}$]', origin='lower', interpolation='nearest')
       

        # 2D x-y distribution
        hist_range_xy = [[None, None], [None, None]]
        hist_range_xy[0] = hist_range[1]
        hist_range_xy[1] = hist_range_zy[1]
        extent_xy = hist_range_xy[0] + hist_range_xy[1]
        extent_xy = [i*1e6 for i in extent_xy]  # [um]

        self.distribution_plot_2D(arr1=xs, arr2=ys, weights=weights, hist_bins=hist_bins, hist_range=hist_range_xy, axes=axs[2][0], extent=extent_xy, vmin=None, vmax=None, colmap=cmap, xlab=xlab, ylab=ylab, clab=r'$\partial^2 N/\partial x \partial y$ [$\mathrm{m}^{-2}$]', origin='lower', interpolation='nearest')
        

        # Energy distribution
        ax = axs[2][1]
        dN_dE, rel_energ = beam.rel_energy_spectrum()
        dN_dE = dN_dE/-e
        ax.fill_between(rel_energ*100, y1=dN_dE, y2=0, color='b', alpha=0.3)
        ax.plot(rel_energ*100, dN_dE, color='b', alpha=0.3, label='Relative energy density')
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        ax.set_xlabel(r'$\mathcal{E}/\langle\mathcal{E}\rangle-1$ [%]')
        ax.set_ylabel('Relative energy density')
        # Add text to the plot
        ax.text(0.05, 0.95, r'$\sigma_\mathcal{E}/\langle\mathcal{E}\rangle=$' f'{format(beam.rel_energy_spread()*100, ".2f")}' '%', fontsize=12, color='black', ha='left', va='top', transform=ax.transAxes)

        # 2D z-energy distribution
        hist_range_energ = [[None, None], [None, None]]
        hist_range_energ[0] = hist_range[0]
        hist_range_energ[1] = [ Es.min(), Es.max() ]  # [eV]
        extent_energ = hist_range_energ[0] + hist_range_energ[1]
        extent_energ[0] = extent_energ[0]*1e6  # [um]
        extent_energ[1] = extent_energ[1]*1e6  # [um]
        extent_energ[2] = extent_energ[2]/1e9  # [GeV]
        extent_energ[3] = extent_energ[3]/1e9  # [GeV]
        self.distribution_plot_2D(arr1=zs, arr2=Es, weights=weights, hist_bins=hist_bins, hist_range=hist_range_energ, axes=axs[2][2], extent=extent_energ, vmin=None, vmax=None, colmap=cmap, xlab=xilab, ylab=energ_lab, clab=r'$\partial^2 N/\partial \xi \partial\mathcal{E}$ [$\mathrm{m}^{-1}$ $\mathrm{eV}^{-1}$]', origin='lower', interpolation='nearest')

    
    # ==================================================
    def scatter_diags(self, beam=None, n_th_particle=1):
        '''
        n_th_particle:  Use this to reduce the amount of plotted particles by only plotting every n_th_particle particle.
        '''

        # Define the color map and boundaries
        colors = ['black', 'red', 'orange', 'yellow']
        bounds = [0, 0.2, 0.4, 0.8, 1]
        cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=256)
                
        if beam is None:
            beam = self.main_beam

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
        xps_lab = '$x\'$ [mrad]'
        yps_lab = '$y\'$ [mrad]'
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
    # Add plots for diagnosing the beam evolution inside a stage.

    
    # ==================================================
    def plot_Ez_rb_cut(self, z_slices=None, main_num_profile=None, zs_Ez=None, Ez=None, Ez_cut=None, zs_rho=None, bubble_radius=None, zlab=r'$z$ [$\mathrm{\mu}$m]'):

        if z_slices is None:
            z_slices = self.z_slices
        if main_num_profile is None:
            main_num_profile = self.main_num_profile
        if zs_Ez is None:
            #zs_Ez = self.zs_Ez_axial
            zs_Ez = self.initial.plasma.wakefield.onaxis.zs
        if Ez is None:
            #Ez = self.Ez_axial
            Ez = self.initial.plasma.wakefield.onaxis.Ezs
        if Ez_cut is None:
            #Ez_cut = self.Ez_roi
            zs = self.main_beam.zs()
            indices = np.argsort(zs)
            zs_sorted = zs[indices]
            Ez_cut = self.Ez_fit_obj(zs_sorted)
            bubble_radius_cut = self.rb_fit_obj(zs_sorted)
        if zs_rho is None:
            zs_rho =  self.zs_bubble_radius_axial
        if bubble_radius is None:
            bubble_radius = self.bubble_radius_axial
        if self.drive_beam is None:
            drive_beam = self.driver_source.track()
        else:
            drive_beam = self.drive_beam
        driver_num_profile, driver_z_slices = self.longitudinal_number_distribution(beam=drive_beam)
        
        
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
        ax_Ez_cut_wakeT2.plot(zs_sorted*1e6, Ez_cut/1e9, 'r', label='Cut-out $E_z$')
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
        ax_rb_cut_wakeT2.plot(zs_sorted*1e6, bubble_radius_cut*1e6, 'r', label=r'Cut-out $r_\mathrm{b}$')
        ax_rb_cut_wakeT2.set_ylabel(r'Bubble radius [$\mathrm{\mu}$m]')
        ax_rb_cut_wakeT2.legend(loc='upper right')

    
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
            main_symmetrise = 'Not registered.'
        else:
            main_symmetrise = str(self.main_source.symmetrize)
        print(f"Symmetrised main beam:\t\t\t\t\t {str(self.main_source.symmetrize) :s}")
        print(f"Symmetrised drive beam:\t\t\t\t\t {str(self.driver_source.symmetrize) :s}\n")
        
        print(f"Transverse wake instability enabled:\t\t\t {str(self.enable_tr_instability) :s}")
        print(f"Radiation reaction enabled:\t\t\t\t {str(self.enable_radiation_reaction) :s}")
        print(f"Ion motion enabled:\t\t\t\t\t {str(self.enable_ion_motion) :s}\n")
        
        print(f"Stage length [m]:\t\t\t\t\t {self.length :.3f}")
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
        
        print(f"Initial matched beta function [mm]:\t\t\t      {self.matched_beta_function(main_beam.energy())*1e3 :.3f}")
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
    def print_current_summary(self, drive_beam, initial_main_beam, beam_out, clean=False):

        with open(self.diag_path + 'output.txt', 'w') as f:
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
                main_symmetrise = 'Not registered.'
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
            print(f"Radiation reaction enabled:\t\t\t {str(self.enable_radiation_reaction) :s}", file=f)
            print(f"Transverse wake instability enabled:\t\t {str(self.enable_tr_instability) :s}", file=f)
            print(f"Radiation reaction enabled:\t\t\t {str(self.enable_radiation_reaction) :s}", file=f)
            print(f"Ion motion enabled:\t\t\t\t {str(self.enable_ion_motion) :s}", file=f)
            print(f"\tIon charge number:\t\t\t {self.ion_charge_num :.3f}", file=f)
            print(f"\tIon mass [u]:\t\t\t\t {self.ion_mass/SI.physical_constants['atomic mass constant'][0] :.3f}", file=f)
            print(f"\tnum_z_cells:\t\t\t\t {self.num_z_cells_main :d}", file=f)
            print(f"\tnum_x_cells_rft:\t\t\t {self.num_x_cells_rft :d}", file=f)
            print(f"\tnum_y_cells_rft:\t\t\t {self.num_y_cells_rft :d}", file=f)
            print(f"\tnum_xy_cells_probe:\t\t\t {self.num_xy_cells_probe :d}", file=f)
            print(f"\tupdate_factor:\t\t\t\t {self.update_factor :.3f}\n", file=f)
            
            print(f"Stage length [m]:\t\t\t\t {self.length :.3f}", file=f)
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
            print(f"Initial number of macroparticles:\t\t {len(drive_beam.xs()) :d}\t\t\t {len(initial_main_beam.xs()) :d}", file=f)
            print(f"Current number of macroparticles:\t\t  \t\t\t {len(beam_out.xs()) :d}", file=f)
            print(f"Initial beam population:\t\t\t {(np.sum(drive_beam.weightings())) :.3e} \t\t {(np.sum(initial_main_beam.weightings())) :.3e}", file=f)
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
            print(f"Initial mean energy [GeV]:\t\t\t {drive_beam.energy(clean=clean)/1e9 :.3f} \t\t {initial_main_beam.energy(clean=clean)/1e9 :.3f}", file=f)
            print(f"Current mean energy [GeV]:\t\t\t \t \t\t {beam_out.energy(clean=clean)/1e9 :.3f}", file=f)
            print(f"Initial rms energy spread [%]:\t\t\t {drive_beam.rel_energy_spread(clean=clean)*1e2 :.3f} \t\t\t {initial_main_beam.rel_energy_spread(clean=clean)*1e2 :.3f}", file=f)
            print(f"Current rms energy spread [%]:\t\t\t  \t\t\t {beam_out.rel_energy_spread(clean=clean)*1e2 :.3f}", file=f)
    
            print(f"Initial beam x offset [um]:\t\t\t {drive_beam.x_offset(clean=clean)*1e6 :.3f} \t\t {initial_main_beam.x_offset(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current beam x offset [um]:\t\t\t  \t\t\t {beam_out.x_offset(clean=clean)*1e6 :.3f}", file=f)
            print(f"Initial beam y offset [um]:\t\t\t {drive_beam.y_offset(clean=clean)*1e6 :.3f} \t\t\t {initial_main_beam.y_offset(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current beam y offset [um]:\t\t\t  \t\t\t {beam_out.y_offset(clean=clean)*1e6 :.3f}", file=f)
            print(f"Initial beam z offset [um]:\t\t\t {drive_beam.z_offset(clean=clean)*1e6 :.3f} \t\t {initial_main_beam.z_offset(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current beam z offset [um]:\t\t\t  \t\t\t {beam_out.z_offset(clean=clean)*1e6 :.3f}\n", file=f)

            print(f"Initial beam x angular offset [urad]:\t\t {drive_beam.x_angle(clean=clean)*1e6 :.3f} \t\t\t {initial_main_beam.x_angle(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current beam x angular offset [urad]:\t\t  \t\t\t {beam_out.x_angle(clean=clean)*1e6 :.3f}", file=f)
            print(f"Initial beam y angular offset [urad]:\t\t {drive_beam.y_angle(clean=clean)*1e6 :.3f} \t\t {initial_main_beam.y_angle(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current beam y angular offset [urad]:\t\t  \t\t\t {beam_out.y_angle(clean=clean)*1e6 :.3f}\n", file=f)
    
            print(f"Initial normalised x emittance [mm mrad]:\t {drive_beam.norm_emittance_x(clean=clean)*1e6 :.3f} \t\t\t {initial_main_beam.norm_emittance_x(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current normalised x emittance [mm mrad]:\t  \t\t\t {beam_out.norm_emittance_x(clean=clean)*1e6 :.3f}", file=f)
            print(f"Initial normalised y emittance [mm mrad]:\t {drive_beam.norm_emittance_y(clean=clean)*1e6 :.3f} \t\t {initial_main_beam.norm_emittance_y(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current normalised y emittance [mm mrad]:\t \t \t\t {beam_out.norm_emittance_y(clean=clean)*1e6 :.3f}\n", file=f)
            
            print(f"Initial angular momentum [mm mrad]:\t\t {drive_beam.angular_momentum()*1e6 :.3f} \t\t\t {initial_main_beam.angular_momentum()*1e6 :.3f}", file=f)
            print(f"Current angular momentum [mm mrad]:\t\t  \t\t\t {beam_out.angular_momentum()*1e6 :.3f}\n", file=f)
            
            print(f"Initial matched beta function [mm]:\t\t\t      {self.matched_beta_function(initial_main_beam.energy(clean=clean))*1e3 :.3f}", file=f)
            print(f"Initial x beta function [mm]:\t\t\t {drive_beam.beta_x(clean=clean)*1e3 :.3f} \t\t\t {initial_main_beam.beta_x(clean=clean)*1e3 :.3f}", file=f)
            print(f"Current x beta function [mm]:\t\t\t \t \t\t {beam_out.beta_x(clean=clean)*1e3 :.3f}", file=f)
            print(f"Initial y beta function [mm]:\t\t\t {drive_beam.beta_y(clean=clean)*1e3 :.3f} \t\t\t {initial_main_beam.beta_y(clean=clean)*1e3 :.3f}", file=f)
            print(f"Current y beta function [mm]:\t\t\t \t \t\t {beam_out.beta_y(clean=clean)*1e3 :.3f}\n", file=f)
    
            print(f"Initial x beam size [um]:\t\t\t {drive_beam.beam_size_x(clean=clean)*1e6 :.3f} \t\t\t {initial_main_beam.beam_size_x(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current x beam size [um]:\t\t\t  \t\t\t {beam_out.beam_size_x(clean=clean)*1e6 :.3f}", file=f)
            print(f"Initial y beam size [um]:\t\t\t {drive_beam.beam_size_y(clean=clean)*1e6 :.3f} \t\t\t {initial_main_beam.beam_size_y(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current y beam size [um]:\t\t\t  \t\t\t {beam_out.beam_size_y(clean=clean)*1e6 :.3f}", file=f)
            print(f"Initial rms beam length [um]:\t\t\t {drive_beam.bunch_length(clean=clean)*1e6 :.3f} \t\t {initial_main_beam.bunch_length(clean=clean)*1e6 :.3f}", file=f)
            print(f"Current rms beam length [um]:\t\t\t \t \t\t {beam_out.bunch_length(clean=clean)*1e6 :.3f}", file=f)
            print(f"Initial peak current [kA]:\t\t\t {drive_beam.peak_current()/1e3 :.3f} \t\t {initial_main_beam.peak_current()/1e3 :.3f}", file=f)
            print(f"Current peak current [kA]:\t\t\t  \t\t\t {beam_out.peak_current()/1e3 :.3f}", file=f)
            print(f"Bubble radius at beam head [um]:\t\t \t\t\t {self.rb_fit_obj(np.max(beam_out.zs()))*1e6 :.3f}", file=f)
            print(f"Bubble radius at beam tail [um]:\t\t \t\t\t {self.rb_fit_obj(np.min(beam_out.zs()))*1e6 :.3f}", file=f)
            print('-------------------------------------------------------------------------------------', file=f)
        f.close() # Close the file

        with open(self.diag_path + 'output.txt', 'r') as f:
            print(f.read())
        f.close()

#vvvvvvvvvvvvvvvvvvvvvv Not currently in use vvvvvvvvvvvvvvvvvvvvvv
'''
###################################################
def growing_end_seq(arr, idxs):
    
    # Find the strictly growing sequence and their continuous indices
    growing_sequence = np.array([])
    growing_indices = np.array([], dtype=int)
    sequence_head_idx = 1
    for i in range(len(arr)-1, 1, -1):
        if arr[i-1] < arr[i] and idxs[i]==idxs[i-1]+1:
            growing_sequence = np.append(growing_sequence, arr[i-1])
            growing_indices = np.append(growing_indices, int(idxs[i-1]))
            sequence_head_idx = i
        else:
            break

    growing_sequence = np.append(growing_sequence, arr[sequence_head_idx-2])
    growing_indices = np.append(growing_indices, int(idxs[sequence_head_idx-2]))
    
    growing_sequence = np.flip(growing_sequence)
    growing_indices = np.flip(growing_indices)
    growing_sequence = np.append(growing_sequence, arr[-1])
    growing_indices = np.append(growing_indices, int(idxs[-1]))
    return growing_sequence, growing_indices
'''