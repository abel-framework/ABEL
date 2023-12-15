"""
Stage class with the transverse wake instability model as described in thesis "Instability and Beam-Beam Study for Multi-TeV PWFA e+e- and gamma gamma Linear Colliders" (https://cds.cern.ch/record/2754022?ln=en).

Ben Chen, 6 October 2023, University of Oslo
"""

import numpy as np
from scipy.constants import c, e, m_e, epsilon_0 as eps0
#from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
#from scipy.stats import linregress
import scipy.signal as signal

#from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # For logarithmic colour scales
from matplotlib.colors import LinearSegmentedColormap  # For customising colour maps
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For manipulating colourbars

from joblib import Parallel, delayed  # Parallel tracking
from joblib_progress import joblib_progress
from types import SimpleNamespace
from openpmd_viewer import OpenPMDTimeSeries
import os, shutil, uuid, copy

from abel.physics_models.particles_transverse_wake_instability import *
from abel.physics_models.twoD_particles_transverse_wake_instability import *
from abel.utilities.plasma_physics import k_p, beta_matched, wave_breaking_field
#from abel.utilities.relativity import energy2gamma
from abel.utilities.statistics import prct_clean, prct_clean2d
from abel.utilities.other import find_closest_value_in_arr
#import abel.utilities.colors as cmaps  # Standardised colour maps
from abel.classes.stage.impl.stage_wake_t import StageWakeT
from abel import Stage, CONFIG
from abel import Beam



class StagePrtclTransWakeInstability(Stage):

    # ==================================================
    def __init__(self, driver_source=None, main_source=None, drive_beam=None, main_beam=None, length=None, nom_energy_gain=None, plasma_density=None, time_step_mod=0.05, main_beam_roi=6.0, num_beam_slice=None, Ez_fit_obj=None, Ez_roi=None, rb_fit_obj=None, bubble_radius_roi=None):
        """
        Parameters
        ----------
        length: [m] float
            Length of the plasma stage.
        
        nom_energy_gain: [eV] float
            Nominal/target energy gain of the acceleration stage.
        
        plasma_density: [m^-3] float
            Plasma density.
        
        driver_source: Source object of drive beam.
        
        main_source: Source object of main beam.

        main_beam: Beam object of main beam.

        driver_beam: Beam object of drive beam.

        time_step_mod: [beta_wave_length/c] float
            Determines the time step of the instability tracking in units of beta_wave_length/c.
        
        #main_beam_roi: float
            Determines the region of interest (also effective beam length) in units of main beam beam length.
        
        #num_beam_slice: int. Number of beam slices.
            
        Ez_fit: [V/m] interpolation object
            1D interpolation object of longitudinal E-field fitted to axial E-field using a selection of zs along the main beam. Used to determine the value of the longitudinal E-field for all beam zs.

        Ez_roi: [V/m] 1D array
            Longitudinal E-field in the region of interest fitted to a selection of zs along the main beam (main beam head to tail).

        rb_fit: [m] interpolation object?
            1D interpolation object of plasma bubble radius fitted to axial bubble radius using a selection of zs along the main beam. Used to determine the value of the bubble radius for all beam zs.
        
        bubble_radius_roi: [m] 1D array
            The bubble radius in the region of interest fitted to a selection of zs along the main beam.

        ...
        """
        
        super().__init__(length, nom_energy_gain, plasma_density)

        self.driver_source = driver_source
        self.main_source = main_source
        self.drive_beam = drive_beam
        self.main_beam = main_beam
        #
        #if drive_beam is None:
        #    self.drive_beam = driver_source.track()
        #else:
        #    self.drive_beam = drive_beam
        #if main_beam is None:
        #    self.main_beam = main_source.track()
        #else:
        #    self.main_beam = main_beam

        self.time_step_mod = time_step_mod  # Determines the time step of the instability tracking in units of beta_wave_length/c.
        self.interstages_enabled = False
        self.show_prog_bar = True
        self.interstage_dipole_field = None
        
        #self.main_beam_roi = main_beam_roi
        #self.num_beam_slice = num_beam_slice
        #if num_beam_slice is None:
        #    # Use the Freedman–Diaconis rule to determine the number of beam slices.
        #    if self.main_beam is None:
        #        self.num_beam_slice = self.FD_rule_num_slice(zs=main_source.track().zs())
        #    else:
        #        self.num_beam_slice = self.FD_rule_num_slice()
        #else:
        #    self.num_beam_slice = num_beam_slice

        self.Ez_fit_obj = Ez_fit_obj  # [V/m] 1d interpolation object of longitudinal E-field fitted to Ez_axial using a selection of zs along the main beam.
        self.Ez_roi = Ez_roi  # [V/m] longitudinal E-field in the region of interest (main beam head to tail).
        self.Ez_axial = None
        self.zs_Ez_axial = None
        self.rb_fit_obj = rb_fit_obj  # [m] 1d interpolation object of bubble radius fitted to bubble_radius_axial using a selection of zs along the main beam.
        self.bubble_radius_roi = bubble_radius_roi  # [m] bubble radius in the region of interest.
        self.bubble_radius_axial = None
        self.zs_bubble_radius_axial = None
        
        #main_num_profile, z_slices = self.longitudinal_number_distribution(beam=main_source.track())
        #self.main_num_profile = main_num_profile
        #self.z_slices = z_slices
        self.main_num_profile = None
        self.z_slices = None
        self.main_slices_edges = None
        
        #self.prop_dist = prop_dist # Obsolete
        #self.driver_num_profile = None
        #self.zs_driver_cut = None
        
        #self.x_slices_main = None
        #self.xp_slices_main = None
        #self.y_slices_main = None
        #self.yp_slices_main = None
        #self.energy_slices_main = None
        #self.s_slices_table_main = None
        #self.x_slices_table_main = None
        #self.xp_slices_table_main = None
        #self.y_slices_table_main = None
        #self.yp_slices_table_main = None
        
        self.diag_path = None
        self.parallel_track_2D = False

        self.ramp_beta_mag = 1
        self.driver_to_wake_efficiency = None
        self.wake_to_beam_efficiency = None
        self.driver_to_beam_efficiency = None
        
        self.reljitter = SimpleNamespace()
        self.reljitter.plasma_density = 0

        # internally sampled values (given some jitter)
        self.__n = None 
        self.driver_initial = None

    
    # ==================================================
    # Track the particles through. Note that when called as part of a Linac object, a copy of the original stage (where no changes has been made) is sent to track() every time. All changes done to self here is saved to a separate stage under the Linac object.
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        # Set the diagnostics directory
        self.diag_path = runnable.run_path()

        # Extract quantities from the stage
        plasma_density = self.plasma_density
        stage_length = self.length
        gamma0 = beam0.gamma()
        time_step_mod = self.time_step_mod

        # ========== Shift the main beam trasversely according to drive beam offset ==========
        if self.driver_source.jitter.x == 0 and self.driver_source.jitter.y == 0:
            drive_beam = self.drive_beam  # This guarantees zero drive beam jitter between stages, as identical drive beams are used in every stage and not re-sampled.
        else:
            drive_beam = self.driver_source.track()
            self.drive_beam = drive_beam  # Generate a drive beam with jitter.
        
        # Apply plasma density up ramp (demagnify beta function) before shifting the coordinates
        drive_beam.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=drive_beam)
        beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=drive_beam)
        
        driver_x_offset = drive_beam.x_offset()
        driver_y_offset = drive_beam.y_offset()
        x_offset = beam0.x_offset()
        y_offset = beam0.y_offset()
        
        xs = beam0.xs()
        beam0.set_xs(xs - driver_x_offset)
        ys = beam0.ys()
        beam0.set_ys(ys - driver_y_offset)

        #print('Driver x/y offsets:', driver_x_offset, driver_y_offset)
        #print('Effective x-offset, beam0.x_offset:', x_offset - driver_x_offset, beam0.x_offset())
        #print('Effective y-offset, beam0.y_offset:', y_offset - driver_y_offset, beam0.y_offset())
        
        # Number profile N(z). Dimensionless, same as dN/dz with each bin multiplied with the widths of the bins.
        main_num_profile, z_slices = self.longitudinal_number_distribution(beam=beam0)
        self.z_slices = z_slices  # Update the longitudinal position of the beam slices.
        self.main_num_profile = main_num_profile
        
        # Sample initial main beam slice x, energy and x'-offsets
        #x_slices_start = self.particles2slices(beam=beam0, beam_quant=beam0.xs(), z_slices=z_slices)
        #xp_slices_start = self.particles2slices(beam=beam0, beam_quant=beam0.xps(), z_slices=z_slices)
        #y_slices_start = self.particles2slices(beam=beam0, beam_quant=beam0.ys(), z_slices=z_slices)
        #yp_slices_start = self.particles2slices(beam=beam0, beam_quant=beam0.yps(), z_slices=z_slices)
        #energy_slices_start = self.particles2slices(beam=beam0, beam_quant=beam0.Es(), z_slices=z_slices)


        # ========== Wake-T simulation and extraction ==========
        # Define a Wake-T stage
        stage_wakeT = StageWakeT()
        stage_wakeT.driver_source = self.driver_source
        k_beta = k_p(plasma_density)/np.sqrt(2*min(gamma0, drive_beam.gamma()/2))
        lambda_betatron = (2*np.pi/k_beta)
        stage_wakeT.length = lambda_betatron/10  # [m]
        stage_wakeT.plasma_density = plasma_density  # [m^-3]
        #
        #stage_wakeT.box_min_z = beam0.zs().min() - 7 * beam0.bunch_length()
        #stage_wakeT.box_max_z = np.mean(drive_beam.zs()) + 5 * drive_beam.bunch_length()
        #
        stage_wakeT.opmd_diag = True  # Set to True for saving simulation results.
        #stage_wakeT.diag_dir = self.diag_path + 'wake_t'

        # Make temp folder
        if not os.path.exists(CONFIG.temp_path):
            os.mkdir(CONFIG.temp_path)
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
        if not os.path.exists(tmpfolder):
            os.mkdir(tmpfolder)
        stage_wakeT.diag_dir = tmpfolder
        
        # Run the Wake-T stage
        beam_copy = copy.deepcopy(beam0)  # Make a deep copy of beam0 to avoid changes on beam0.
        beam_wakeT = stage_wakeT.track(beam_copy)
        
        # Read the Wake-T simulation data
        path_sep = os.sep
        path = stage_wakeT.diag_dir + path_sep + 'hdf5'
        ts = OpenPMDTimeSeries(path)
        #dump_time = ts.t[0]  # Extract first time step dump.

        # Extract longitudinal E-field
        Ez_wakeT, info_Ez = ts.get_field(field='E', coord='z', iteration=0, plot=False)
        zs_Ez_wakeT = info_Ez.z
        rs_Ez = info_Ez.r
        
        # Extract axial longitudinal E-field
        Ez_axis_wakeT = Ez_wakeT[round(len(info_Ez.r)/2),:]

        # Cut out axial Ez over the ROI
        Ez, Ez_fit = self.Ez_shift_fit(Ez_axis_wakeT, zs_Ez_wakeT, beam0, z_slices)
        
        
        # Extract plasma charge density
        rho, info_rho = ts.get_field(field='rho', iteration=0, plot=False)
        
        # Calculate the number density
        plasma_num_density = rho/stage_wakeT.plasma_density/-e
        
        # Extract coordinates
        zs_rho = info_rho.z
        rs_rho = info_rho.r
        
        # Extract the plasma bubble radius
        bubble_radius_wakeT = self.get_bubble_radius(plasma_num_density, rs_rho, driver_x_offset, x_offset, threshold=0.8)

        # Cut out bubble radius over the ROI
        bubble_radius, rb_fit = self.rb_shift_fit(bubble_radius_wakeT, zs_rho, beam0, z_slices) # Actually same as Ez_shift_fit. Consider making just one function instead... 

        # Save quantities to the stage
        self.__save_initial_wake(Ez0_axial=Ez_axis_wakeT, metadata_Ez0=info_Ez, rho0=rho, metadata_rho0=info_rho, driver0=drive_beam, beam0=beam0)
        self.Ez_fit_obj = Ez_fit
        self.rb_fit_obj = rb_fit
        
        # TODO: move these to self.initial
        self.Ez_roi = Ez
        self.Ez_axial = Ez_axis_wakeT
        self.zs_Ez_axial = zs_Ez_wakeT
        self.bubble_radius_roi = bubble_radius
        self.bubble_radius_axial = bubble_radius_wakeT
        self.zs_bubble_radius_axial = zs_rho
        

        # Remove temporary directory
        if os.path.exists(tmpfolder):
            shutil.rmtree(tmpfolder)

        # Make plots for control if necessary
        #self.plot_Ez_rb_cut(z_slices, main_num_profile, zs_Ez_wakeT, Ez_axis_wakeT, Ez, zs_rho, bubble_radius_wakeT, bubble_radius, zlab='$z$ [$\mathrm{\mu}$m]')
        

        # ========== Instability tracking ==========
        inputs = [beam0, plasma_density, Ez_fit, rb_fit, stage_length, time_step_mod, z_slices]
        some_are_none = any(input is None for input in inputs)
        
        if some_are_none:
            none_indices = [i for i, x in enumerate(inputs) if x is None]
            print(none_indices)
            raise ValueError('At least one input is set to None.')

        if self.parallel_track_2D is True:
            
            with joblib_progress('Tracking x and y in parallel:', 2):
                results = Parallel(n_jobs=2)([
                    delayed(twoD_transverse_wake_instability_particles)(beam0, beam0.xs(), beam0.pxs(), plasma_density, Ez_fit, rb_fit, stage_length, time_step_mod, get_centroids=False, s_slices=None, z_slices=None),
                    delayed(twoD_transverse_wake_instability_particles)(beam0, beam0.ys(), beam0.pys(), plasma_density, Ez_fit, rb_fit, stage_length, time_step_mod, get_centroids=False, s_slices=None, z_slices=None)
                ])
                time.sleep(0.1) # hack to allow printing progress
            
            xs_sorted, pxs_sorted, zs_sorted, pzs_sorted, weights_sorted, s_slices_table, offset_slices_table, angle_slices_table = results[0]
            ys_sorted, pys_sorted, _, _, _, _, _, _ = results[1]

            # Initialise ABEL Beam object
            beam = Beam()
            
            # Set the phase space of the ABEL beam
            beam.set_phase_space(Q=np.sum(weights_sorted)*-e,
                                 xs=xs_sorted,
                                 ys=ys_sorted,
                                 zs=zs_sorted, 
                                 pxs=pxs_sorted,
                                 pys=pys_sorted,
                                 pzs=pzs_sorted)
        else:
            beam, s_slices_table, x_slices_table, xp_slices_table, y_slices_table, yp_slices_table = transverse_wake_instability_particles(beam0, plasma_density=plasma_density, Ez_fit_obj=Ez_fit, rb_fit_obj=rb_fit, stage_length=stage_length, time_step_mod=time_step_mod, get_centroids=False, s_slices=None, z_slices=None, show_prog_bar=self.show_prog_bar)
        
        #s_slices = s_slices_table[-1,:]
        #x_slices = x_slices_table[-1,:]
        #xp_slices = xp_slices_table[-1,:]
        #y_slices = y_slices_table[-1,:] ##############
        #yp_slices = yp_slices_table[-1,:] ##############
        

        # ========== Shift the main beam back to the original coordinate system ==========
        xs = beam.xs()
        beam.set_xs(xs + driver_x_offset)
        ys = beam.ys()
        beam.set_ys(ys + driver_y_offset)

        # Apply plasma density down ramp (magnify beta function) after shifting the coordinates back to original reference
        beam.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=drive_beam)
        
        self.main_beam = copy.deepcopy(beam)  # Need to make a deepcopy, or changes to beam may affect the Beam object saved here.
        #self.x_slices_main = x_slices  # Update the transverse offsets of the beam slices.
        #self.xp_slices_main = xp_slices  # [rad] Update the x's of the beam slices.
        #self.y_slices_main = y_slices  # Update the transverse offsets of the beam slices.
        #self.yp_slices_main = yp_slices  # [rad] Update the x's of the beam slices.
        #self.energy_slices_main = energy_slices  # Update the energies of the beam slices.
        #self.s_slices_table_main = s_slices_table
        #self.x_slices_table_main = x_slices_table
        #self.xp_slices_table_main = xp_slices_table
        #self.y_slices_table_main = y_slices_table
        #self.yp_slices_table_main = yp_slices_table

        
        # Copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        return super().track(beam, savedepth, runnable, verbose)


    # ==================================================
    # 
    def __save_initial_wake(self, Ez0_axial, metadata_Ez0, rho0, metadata_rho0, driver0, beam0):
        
        # ========== Save initial axial wakefield info ========== 
        self.initial.plasma.wakefield.onaxis.zs = metadata_Ez0.z
        self.initial.plasma.wakefield.onaxis.Ezs = Ez0_axial

        # ========== Save plasma electron number density info ========== 
        self.initial.plasma.density.rho = rho0/-SI.e
        self.initial.plasma.density.extent = metadata_rho0.imshow_extent  # array([z_min, z_max, x_min, x_max])

        # ========== Initial beam particle density ==========
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

        self.initial.beam.density.extent = extent_beams  # array([z_min, z_max, x_min, x_max])
        self.initial.beam.density.rho = projection_zx
        
        # ========== Save initial beam currents ==========
        self.calculate_beam_current(beam0, driver0)

    
    # ==================================================
    def fill_nan_with_mean(self, arr):
        nan_indices = np.isnan(arr)
        valid_indices = ~nan_indices
        edge_indices = np.where(nan_indices & (valid_indices | np.roll(valid_indices, 1) | np.roll(valid_indices, -1)))
    
        for i in edge_indices[0]:
            if valid_indices[i]:
                continue  # Skip non-NaN values
    
            left_neighbor_idx = i - 1
            right_neighbor_idx = i + 1
    
            while left_neighbor_idx >= 0 and np.isnan(arr[left_neighbor_idx]):
                left_neighbor_idx -= 1
    
            while right_neighbor_idx < len(arr) and np.isnan(arr[right_neighbor_idx]):
                right_neighbor_idx += 1
    
            left_neighbor = arr[left_neighbor_idx] if left_neighbor_idx >= 0 else np.nan
            right_neighbor = arr[right_neighbor_idx] if right_neighbor_idx < len(arr) else np.nan
    
            # Calculate the mean of the valid neighbors
            neighbor_mean = np.nanmean([left_neighbor, right_neighbor])
    
            arr[i] = neighbor_mean
    
        return arr

    
    # ==================================================
    # Returns the mean of a beam quantity beam_quant (1D float array) for all particles inside the beam slices.
    def particles2slices(self, beam, beam_quant, z_slices=None, bin_number=None, cut_off=None, make_plot=False):
        """
        Returns the mean of a beam quantity beam_quant for all particles inside the beam slices.

        Parameters
        ----------
        beam: Beam object.
            
        beam_quant: 1D float arrays
            Beam quantity to be binned into bins/slices defined by z_slices. The mean is calculated for the quantity for all particles in the z-bins. Includes e.g. beam.xs(), beam.Es() etc.
            
        z_slices: [m] 1D float array
            z-coordinates of the beam slices.

        bin_number: float
            Number of beam slices.

        cut_off: float
            Determines the longitudinal coordinates inside the region of interest

        make_plot: boolean
            Flag for making plots.

            
        Returns
        ----------
        beam_quant_slices: 1D float array
            beam_quant binned into bins/slices defined by z_slices. The mean is calculated for the quantity for all particles in the z-bins. Includes e.g. beam.xs(), beam.Es() etc.
        """
        
        if bin_number is None:
            bin_number = self.num_beam_slice
        if cut_off is None:
            cut_off = self.main_beam_roi * beam.bunch_length()
        
        zs = beam.zs()
        mean_z = np.mean(zs)

        # Get all elements inside the region of interest.
        bool_indices = (zs <= mean_z + cut_off) & (zs >= mean_z - cut_off)
        zs_roi = zs[bool_indices]
        #beam_quant = beam_quant[bool_indices]

        if z_slices is None:
            _, edges = np.histogram(zs_roi, bins=bin_number)  # Get the edges of the histogram of z with bin_number bins.
            z_slices = (edges[0:-1] + edges[1:])/2  # Centres of the beam slices (z).

        # Make small bins for interpolation
        Nbins = int(np.sqrt(len(zs)/2))
        _, edges = np.histogram(zs, bins=Nbins)
        zs_bins = (edges[0:-1] + edges[1:])/2  # Centres of the bins (z).
        
        # Sort the arrays
        indices = np.argsort(zs)
        zs_sorted = zs[indices]  # Particle quantity.
        beam_quant_sorted = beam_quant[indices]  # Particle quantity.

        # Compute the mean of beam_quant of all particles inside a z-bin.
        beam_quant_bins = np.empty(len(zs_bins))
        for i in range(0,len(zs_bins)-1):
            left = np.searchsorted(zs_sorted, zs_bins[i])  # zs_sorted[left:len(zs_sorted)] >= zs_bins[i], left side of bin i.
            right = np.searchsorted(zs_sorted, zs_bins[i+1], side='right')  # zs_sorted[0:right] <= zs_bins[i+1], right (larger) side of bin i.
            beam_quant_bins[i] = np.mean(beam_quant_sorted[left:right])

        if np.any(np.isnan(beam_quant_bins)):
            beam_quant_bins = self.fill_nan_with_mean(beam_quant_bins)  # Replace the nan values with the mean.
           
        beam_quant_slices = np.interp(z_slices, zs_bins, beam_quant_bins)
        beam_quant_slices = self.fill_nan_with_mean(beam_quant_slices)  # Replace the nan values with the mean.
        if np.any(np.isnan(beam_quant_slices)):
            plt.figure()
            plt.scatter(zs*1e6, beam_quant)
            plt.plot(z_slices*1e6, beam_quant_slices, 'r')
            plt.xlabel(r'$\xi$ [$\mathrm{\mu}$m]')
            raise ValueError('Interpolated array still contains Nan.')

        if make_plot is True:
            plt.figure()
            plt.scatter(zs*1e6, beam_quant)
            plt.plot(z_slices*1e6, beam_quant_slices, 'rx-')
            plt.xlabel(r'$\xi$ [$\mathrm{\mu}$m]')
        
        return beam_quant_slices

    
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
            print('Warning: the longitudinal E-field may not have been accurately fitted.\n')
        
        return Ez_fit(z_slices), Ez_fit

    
    # ==================================================
    def get_bubble_radius(self, plasma_num_density, plasma_tr_coord, driver_offset=None, main_offset=None, threshold=0.8):
        """
        - For extracting the plasma ion bubble radius by finding the coordinates in which the plasma number density goes from zero to a threshold value.
        - The symmetry axis is determined using the transverse offset of the drive beam.
        - xi is the propagation direction pointing to the right.
        
        Parameters
        ----------
        plasma_num_density: [n0] 2D float array
            Plasma number density in units of initial number density n0. Need to be oriented with propagation direction pointing to the right and positive offset pointing upwards.
            
        plasma_tr_coord: [m] 1D float array 
            Transverse coordinate of plasma_num_density.

        driver_offset: [m] float
            Mean transverse offset of the drive beam.

        main_offset: [m] float
            Mean transverse offset of the main beam.
            
        threshold: float
            Defines a threshold for the plasma density to determine bubble_radius.

            
        Returns
        ----------
        bubble_radius: [m] 1D array 
            Plasma bubble radius over the simulation box.
        """

        # Find the offsets of the beams
        if driver_offset is None:
            drive_beam = self.drive_beam
            driver_offset = drive_beam.x_offset()
        if main_offset is None:
            main_beam = self.main_beam
            main_offset = main_beam.x_offset()
        
        # Find the value in plasma_tr_coord closest to driver_offset and corresponding index
        idx_middle, _ = find_closest_value_in_arr(plasma_tr_coord, driver_offset)
        
        # Find the value in plasma_tr_coord closest to main_offset and corresponding index
        idx_offset, _ = find_closest_value_in_arr(plasma_tr_coord, main_offset)

        rows, cols = np.shape(plasma_num_density)
        #idx_middle = np.round(rows/2)
        bubble_radius = np.zeros(cols)

        for i in range(0,cols):  # Loop through all transverse slices.
            
            # Extract a transverse slice
            slice = plasma_num_density[:,i]

            peaks, _ = signal.find_peaks(slice)  # Find all peaks in the slice.
            idxs_peaks_above = peaks[peaks > idx_middle]  # Get the slice peaks above middle.
            idxs_peaks_below = peaks[peaks < idx_middle]  # Get the slice peaks below middle.

            # Check if there are peaks on both sides
            if len(idxs_peaks_above) > 0 and len(idxs_peaks_below) > 0:
                idx_above = idxs_peaks_above[-1]  # Get the slice peak above closest to middle.
                idx_below = idxs_peaks_below[0]  # Get the slice peak below closest to middle.
                valley = slice[idx_below+1:idx_above]  # Get the elements between the peaks.

                # Check that the "valley" of the slice is not too shallow. If so, bubble radius should be > 0.
                if len(valley) and valley.min() < threshold:
                    idxs_slice_above_thres = np.where(slice > threshold)[0]  # Find the indices of all the elements in the slice > threshold.
   
                    # Find the value in idxs_slice_above_thres closest to idx_offset (equivalent to the value in plasma_num_density above threshold that is closest to the main beam offset axis, i.e. the innermost plasma electron layer).
                    _, idx = find_closest_value_in_arr(idxs_slice_above_thres, idx_offset)
                    bubble_radius[i] = np.abs(plasma_tr_coord[idx] - driver_offset)
        
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
            print('Warning: the plasma bubble radius may not have been accurately fitted.\n')
        
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
            #bin_number = self.num_beam_slice

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
    def matched_beta_function(self, energy):
        return beta_matched(self.plasma_density, energy) * self.ramp_beta_mag
        

    # ==================================================
    #def beta_wavenumber_slices(self, beam=None, clean=False):  # Drop this?
#
    #    if beam is None:
    #        beam = self.main_beam
    #    xs, xps = prct_clean2d(beam.xs(), beam.xps(), clean)
    #    zs = prct_clean(beam.zs(), clean)
    #    z_slices = self.z_slices
    #    
    #    # Sort the arrays
    #    indices = np.argsort(zs)
    #    zs_sorted = zs[indices]
    #    xs_sorted = xs[indices]
    #    xps_sorted = xps[indices]
    #    
    #    edges = self.main_slices_edges
    #    k_beta = np.zeros(len(edges)-1)
#
    #    for i in range(0, len(edges)-1):
    #        left = np.searchsorted(zs_sorted, edges[i])  # zs_sorted[left:len(zs_sorted)] >= edges[i].
    #        right = np.searchsorted(zs_sorted, edges[i+1], side='right')  # zs_sorted[0:right] <= edges[i+1].
    #        #print(i, left, right)
    #        xs_in_bin = xs_sorted[left:right]
    #        xps_in_bin = xps_sorted[left:right]
    #        covx = np.cov(xs_in_bin, xps_in_bin)
    #        k_beta[i] = 1/(covx[0,0]/np.sqrt(np.linalg.det(covx)))
#
    #    #plt.figure()
    #    #plt.plot(self.z_slices*1e6, k_beta, 'x-')
    #    #for edge in edges:
    #    #    plt.axvline(x=edge*1e6, color=(0.5, 0.5, 0.5), linestyle='-', alpha=0.3)
    #    
    #    return k_beta
    
    
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
        Ezs = self.Ez_axial
        zs_Ez = self.zs_Ez_axial
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
        axs[0].set_xlabel('z (um)')
        axs[0].set_ylabel('Longitudinal electric field (GV/m)')
        axs[0].set_xlim(zlims)
        axs[0].set_ylim(bottom=1.05*min(Ezs)/1e9, top=1.3*max(Ezs)/1e9)
        
        axs[1].fill(np.concatenate((zs0, np.flip(zs0)))*1e6, np.concatenate((-Is, np.zeros(Is.shape)))/1e3, color=col1, alpha=af)
        axs[1].plot(zs0*1e6, -Is/1e3, '-', color=col1)
        axs[1].set_xlabel('z (um)')
        axs[1].set_ylabel('Beam current (kA)')
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=0, top=1.2*max(-Is)/1e3)
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=1.2*min(-Is)/1e3, top=1.2*max(-Is)/1e3)
        
        if includeWakeRadius:
            axs[2].fill(np.concatenate((zs_rho, np.flip(zs_rho)))*1e6, np.concatenate((bubble_radius, np.ones(zs_rho.shape)))*1e6, color=col2, alpha=af)
            axs[2].plot(zs_rho*1e6, bubble_radius*1e6, '-', color=col2)
            axs[2].set_xlabel('z (um)')
            axs[2].set_ylabel('Plasma-wake radius (um)')
            axs[2].set_xlim(zlims)
            axs[2].set_ylim(bottom=0, top=max(bubble_radius*1.2)*1e6)
        
        # save to file
        if saveToFile is not None:
            plt.savefig(saveToFile, format="pdf", bbox_inches="tight")

    
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
    def distribution_plot_2D(self, arr1, arr2, weights=None, hist_bins=None, hist_range=None, axes=None, extent=None, vmin=None, vmax=None, colmap='viridis', xlab='', ylab='', clab='', origin='lower', interpolation='nearest', reduce_cax_pad=False):

        if weights is None:
            weights = self.main_beam.weightings()
        if hist_bins is None:
            nbins = int(np.sqrt(len(arr1)/2))
            hist_bins = [ nbins, nbins ]  # list of 2 ints. Number of bins along each direction, for the histograms
        if hist_range is None:
            hist_range = [[None, None], [None, None]]
            hist_range[0] = [ arr1.min(), arr1.max() ]  # List contains 2 lists of 2 floats. Extent of the histogram along each direction
            hist_range[1] = [ arr2.min(), arr2.max() ]
        
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
        #xis = zs
        xs = beam.xs()
        xps = beam.xps()
        ys = beam.ys()
        yps = beam.yps()
        Es = beam.Es()
        weights = beam.weightings()

        # Labels for plots
        zlab = '$z$ [$\mathrm{\mu}$m]'
        xilab = r'$\xi$ [$\mathrm{\mu}$m]'
        #slab = '$s$ [m]'
        xlab = '$x$ [$\mathrm{\mu}$m]'
        ylab = '$y$ [$\mathrm{\mu}$m]'
        #field_lab = '$E_z$ [GV/m]'
        #dN_dz_lab = '$\partial N/\partial z$ [$\mathrm{m}^{-1}$]'
        xps_lab = '$x\'$ [mrad]'
        yps_lab = '$y\'$ [mrad]'
        energ_lab = '$\mathcal{E}$ [GeV]'
        
        # Set up a figure with axes
        fig, axs = plt.subplots(nrows=3, ncols=3, layout='constrained', figsize=(5*3, 4*3))
        fig.suptitle('$\Delta s=$' f'{format(beam.location, ".2f")}' ' m')

        nbins = int(np.sqrt(len(weights)/2))
        hist_bins = [ nbins, nbins ]  # list of 2 ints. Number of bins along each direction, for the histograms

        # 2D z-x distribution
        hist_range = [[None, None], [None, None]]
        hist_range[0] = [ zs.min(), zs.max() ]  # [m], list contains 2 lists of 2 floats. Extent of the histogram along each direction
        hist_range[1] = [ xs.min(), xs.max() ]
        extent_zx = hist_range[0] + hist_range[1]
        extent_zx = [i*1e6 for i in extent_zx]  # [um]

        self.distribution_plot_2D(arr1=zs, arr2=xs, weights=weights, hist_bins=hist_bins, hist_range=hist_range, axes=axs[0][0], extent=extent_zx, vmin=None, vmax=None, colmap=cmap, xlab=xilab, ylab=xlab, clab=r'$\partial^2 N/\partial\xi \partial x$ [$\mathrm{m}^{-2}$]', origin='lower', interpolation='nearest')
        if plot_centroids is True:
            axs[0][0].plot(self.z_slices*1e6, self.x_slices_main*1e6, 'r', alpha=0.5)
            axs[0][0].axis(extent_zx)

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
        if plot_centroids is True:
            axs[0][1].plot(self.z_slices*1e6, self.xp_slices_main*1e3, 'r', alpha=0.5)
            axs[0][1].axis(extent_xps)
        
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
        #if plot_centroids is True:
        #    axs[0][2].plot(self.x_slices_main*1e6, self.xp_slices_main*1e3, 'rx', alpha=0.5, label='From tracking')

        # 2D z-y distribution
        hist_range_zy = [[None, None], [None, None]]
        hist_range_zy[0] = hist_range[0]
        hist_range_zy[1] = [ ys.min(), ys.max() ]
        extent_zy = hist_range_zy[0] + hist_range_zy[1]
        extent_zy = [i*1e6 for i in extent_zy]  # [um]

        self.distribution_plot_2D(arr1=zs, arr2=ys, weights=weights, hist_bins=hist_bins, hist_range=hist_range_zy, axes=axs[1][0], extent=extent_zy, vmin=None, vmax=None, colmap=cmap, xlab=xilab, ylab=ylab, clab=r'$\partial^2 N/\partial\xi \partial y$ [$\mathrm{m}^{-2}$]', origin='lower', interpolation='nearest')
        if plot_centroids is True:
            axs[1][0].plot(self.z_slices*1e6, self.y_slices_main*1e6, 'r', alpha=0.5)
            axs[1][0].axis(extent_zy)

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
        if plot_centroids is True:
            axs[1][1].plot(self.z_slices*1e6, self.yp_slices_main*1e3, 'r', alpha=0.5)
            axs[1][1].axis(extent_yps)

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
        #if plot_centroids is True:
        #    axs[1][2].plot(self.y_slices_main*1e6, self.yp_slices_main*1e3, 'rx', alpha=0.5, label='From tracking')

        # 2D x-y distribution
        hist_range_xy = [[None, None], [None, None]]
        hist_range_xy[0] = hist_range[1]
        hist_range_xy[1] = hist_range_zy[1]
        extent_xy = hist_range_xy[0] + hist_range_xy[1]
        extent_xy = [i*1e6 for i in extent_xy]  # [um]

        self.distribution_plot_2D(arr1=xs, arr2=ys, weights=weights, hist_bins=hist_bins, hist_range=hist_range_xy, axes=axs[2][0], extent=extent_xy, vmin=None, vmax=None, colmap=cmap, xlab=xlab, ylab=ylab, clab=r'$\partial^2 N/\partial x \partial y$ [$\mathrm{m}^{-2}$]', origin='lower', interpolation='nearest')

        # 2D x'-y' distribution
        #hist_range_xpyp = [[None, None], [None, None]]
        #hist_range_xpyp[0] = hist_range_xps[1]
        #hist_range_xpyp[1] = hist_range_yyp[1]
        #extent_xpyp = hist_range_xpyp[0] + hist_range_xpyp[1]
        #extent_xpyp[0] = extent_xpyp[0]*1e3  # [mrad]
        #extent_xpyp[1] = extent_xpyp[1]*1e3  # [mrad]
        #extent_xpyp[2] = extent_xpyp[2]*1e3  # [mrad]
        #extent_xpyp[3] = extent_xpyp[3]*1e3  # [mrad]
        #self.distribution_plot_2D(arr1=xps, arr2=yps, weights=weights, hist_bins=hist_bins, hist_range=hist_range_xpyp, axes=axs[2][1], extent=extent_xpyp, vmin=None, vmax=None, colmap=cmap, xlab=xps_lab, ylab=yps_lab, clab='$\partial^2 N/\partial x\' \partial y\'$ [$\mathrm{rad}^{-2}$]', origin='lower', interpolation='nearest')

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
        if plot_centroids is True:
            axs[2][2].plot(self.z_slices*1e6, self.energy_slices_main/1e9, 'r', alpha=0.3)
            axs[2][2].axis(extent_energ)

    
    # ==================================================
    def scatter_diags(self, beam=None, plot_centroids=False, n_th_particle=1, show_slice_grid=False, plot_k_beta=False):
        '''
        plot_centroids:  Plot the centroids of the beam obtained from instability tracking.
        
        n_th_particle:  Use this to reduce the amount of plotted particles by only plotting every n_th_particle particle.

        show_slice_grid: Plot a vertical grid to display the longitudinal position of the beam slices.

        plot_k_beta: Plot the betatron wavenumber along the beam.
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
        zlab = '$z$ [$\mathrm{\mu}$m]'
        xilab = r'$\xi$ [$\mathrm{\mu}$m]'
        xlab = '$x$ [$\mathrm{\mu}$m]'
        ylab = '$y$ [$\mathrm{\mu}$m]'
        xps_lab = '$x\'$ [mrad]'
        yps_lab = '$y\'$ [mrad]'
        energ_lab = '$\mathcal{E}$ [GeV]'
        
        # Set up a figure with axes
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(5*3, 4*3))
        plt.tight_layout(pad=6.0)  # Sets padding between the figure edge and the edges of subplots, as a fraction of the font size.
        fig.subplots_adjust(top=0.85)  # By setting top=..., you are specifying that the top boundary of the subplots should be at ...% of the figure’s height.
        
        # 2D z-x distribution
        ax = axs[0][0]
        p = ax.scatter(zs[::n_th_particle]*1e6, xs[::n_th_particle]*1e6, c=Es[::n_th_particle]/1e9, cmap=cmap)
        #ax.axis(extent)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xlab)
        if plot_centroids is True:
            X_slices_from_beam = self.particles2slices(beam=beam, beam_quant=xs, z_slices=self.z_slices)
            ax.plot(self.z_slices*1e6, X_slices_from_beam*1e6, 'cx', alpha=0.7, label='Slice mean $x$ from particles')
            ax.plot(self.z_slices*1e6, self.x_slices_main*1e6, 'b', alpha=0.5, label='Slice mean $x$ from tracking')
            ax.legend()
        if show_slice_grid is True:
            for z_slice in self.z_slices:
                ax.axvline(x=z_slice*1e6, color=(0.5, 0.5, 0.5), linestyle='-', alpha=0.3)
        if plot_k_beta is True:  # Drop this?
            ax_twin = ax.twinx()
            #ax_Ez_cut_wakeT2 = axs_wakeT_cut[0].twinx()
            k_beta = self.beta_wavenumber_slices(beam=beam, clean=False)
            ax_twin.plot(self.z_slices*1e6, k_beta, 'g')
            ax_twin.set_ylabel(r'$k_\beta$ [$\mathrm{m}^{-1}$]')
            
        # 2D z-x' distribution
        ax = axs[0][1]
        ax.scatter(zs[::n_th_particle]*1e6, xps[::n_th_particle]*1e3, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xps_lab)
        if plot_centroids is True:
            xp_slices_from_beam = self.particles2slices(beam=beam, beam_quant=xps, z_slices=self.z_slices)
            ax.plot(self.z_slices*1e6, xp_slices_from_beam*1e3, 'cx', alpha=0.7, label='Slice mean $x\'$ from particles')
            ax.plot(self.z_slices*1e6, self.xp_slices_main*1e3, 'b', alpha=0.5, label='Slice mean $x\'$ from tracking')
            ax.legend()
            
        # 2D x-x' distribution
        ax = axs[0][2]
        ax.scatter(xs[::n_th_particle]*1e6, xps[::n_th_particle]*1e3, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xlab)
        ax.set_ylabel(xps_lab)
        if plot_centroids is True:
            ax.plot(self.x_slices_main*1e6, self.xp_slices_main*1e3, 'cx', alpha=0.5, label='From tracking')
            ax.legend()
            
        # 2D z-y distribution
        ax = axs[1][0]
        ax.scatter(zs[::n_th_particle]*1e6, ys[::n_th_particle]*1e6, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xilab)
        ax.set_ylabel(ylab)
        if plot_centroids is True:
            y_slices_from_beam = self.particles2slices(beam=beam, beam_quant=ys, z_slices=self.z_slices)
            ax.plot(self.z_slices*1e6, y_slices_from_beam*1e6, 'cx', alpha=0.7, label='Slice mean $y$ from particles')
            ax.plot(self.z_slices*1e6, self.y_slices_main*1e6, 'b', alpha=0.5, label='Slice mean $y$ from tracking')
            ax.legend()
        if show_slice_grid is True:
            for z_slice in self.z_slices:
                ax.axvline(x=z_slice*1e6, color=(0.5, 0.5, 0.5), linestyle='-', alpha=0.5)
            
        # 2D z-y' distribution
        ax = axs[1][1]
        ax.scatter(zs[::n_th_particle]*1e6, yps[::n_th_particle]*1e3, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xilab)
        ax.set_ylabel(yps_lab)
        if plot_centroids is True:
            yp_slices_from_beam = self.particles2slices(beam=beam, beam_quant=yps, z_slices=self.z_slices)
            ax.plot(self.z_slices*1e6, yp_slices_from_beam*1e3, 'cx', alpha=0.7, label='Slice mean $y\'$ from particles')
            ax.plot(self.z_slices*1e6, self.yp_slices_main*1e3, 'b', alpha=0.5, label='Slice mean $y\'$ from tracking')
            ax.legend()
            
        # 2D y-y' distribution
        ax = axs[1][2]
        ax.scatter(ys[::n_th_particle]*1e6, yps[::n_th_particle]*1e3, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(ylab)
        ax.set_ylabel(yps_lab)
        if plot_centroids is True:
            ax.plot(self.y_slices_main*1e6, self.yp_slices_main*1e3, 'cx', alpha=0.5, label='From tracking')
            ax.legend()
            
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
        if plot_centroids is True:
            energy_slices_from_beam = self.particles2slices(beam=beam, beam_quant=beam.Es(), z_slices=self.z_slices)
            ax.plot(self.z_slices*1e6, energy_slices_from_beam/1e9, 'kx', alpha=0.7, label='Slice mean $\mathcal{E}$ from particles')
            ax.plot(self.z_slices*1e6, self.energy_slices_main/1e9, 'r', alpha=0.5, label='Slice mean $\mathcal{E}$ from tracking')
            ax.legend()
            
        # Set label and other properties for the colorbar
        fig.suptitle('$\Delta s=$' f'{format(beam.location, ".2f")}' ' m')
        cbar_ax = fig.add_axes([0.15, 0.91, 0.7, 0.02])   # The four values in the list correspond to the left, bottom, width, and height of the new axes, respectively.
        fig.colorbar(p, cax=cbar_ax, orientation='horizontal', label=energ_lab)
        

    # ==================================================
    # Add plots for diagnosing the beam evolution inside a stage.

    
    # ==================================================
    def plot_Ez_rb_cut(self, z_slices=None, main_num_profile=None, zs_Ez=None, Ez=None, Ez_cut=None, zs_rho=None, bubble_radius=None, zlab='$z$ [$\mathrm{\mu}$m]'):

        if z_slices is None:
            z_slices = self.z_slices
        if main_num_profile is None:
            main_num_profile = self.main_num_profile
        if zs_Ez is None:
            zs_Ez = self.zs_Ez_axial
        if Ez is None:
            Ez = self.Ez_axial
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
        ax_rb_cut_wakeT2.plot(zs_rho*1e6, bubble_radius*1e6, label='Full $r_\mathrm{b}$')
        ax_rb_cut_wakeT2.plot(zs_sorted*1e6, bubble_radius_cut*1e6, 'r', label='Cut-out $r_\mathrm{b}$')
        ax_rb_cut_wakeT2.set_ylabel('Bubble radius [$\mathrm{\mu}$m]')
        ax_rb_cut_wakeT2.legend(loc='upper right')


    # ==================================================
    # Plots the transverse oscillation vs propagation coordinate s for selected beam slices. TODO
    def slice_offset_s_diag(self, beam):
        z_slices = self.z_slices
        
        # Beam slices data
        s_slices_table = self.s_slices_table_main
        x_slices_table = self.x_slices_table_main
        y_slices_table = self.y_slices_table_main

        # Find the corresponding offset data for comparison
        ref_slice_sigma_z_min2 = -2.0
        ref_slice_sigma_z_min1 = -1.0
        ref_slice_sigma_z_0 = 0.0
        ref_slice_sigma_z_2 = 2.0
        
        mean_z = beam.z_offset(clean=False)
        slice_index_min2, _ = find_closest_value_in_arr(z_slices, mean_z+ref_slice_sigma_z_min2*beam.bunch_length())  # index n_sigma_z*sigma_z away from beam center towards tail. Lower index towards tail.
        slice_index_min1, _ = find_closest_value_in_arr(z_slices, mean_z+ref_slice_sigma_z_min1*beam.bunch_length())
        slice_index_0, _ = find_closest_value_in_arr(z_slices, mean_z)
        slice_index_2, _ = find_closest_value_in_arr(z_slices, mean_z+ref_slice_sigma_z_2*beam.bunch_length())
        
        s_min2 = s_slices_table[:,slice_index_min2]
        X_min2 = x_slices_table[:,slice_index_min2]*1e6  # [um]
        Y_min2 = y_slices_table[:,slice_index_min2]*1e6  # [um]
        s_min1 = s_slices_table[:,slice_index_min1]
        X_min1 = x_slices_table[:,slice_index_min1]*1e6  # [um]
        Y_min1 = y_slices_table[:,slice_index_min1]*1e6  # [um]
        s_0 = s_slices_table[:,slice_index_0]
        X_0 = x_slices_table[:,slice_index_0]*1e6  # [um]
        Y_0 = y_slices_table[:,slice_index_0]*1e6  # [um]
        s_2 = s_slices_table[:,slice_index_2]
        X_2 = x_slices_table[:,slice_index_2]*1e6  # [um]
        Y_2 = y_slices_table[:,slice_index_2]*1e6  # [um]

        # Make figure with axes
        fig, axs = plt.subplots(nrows=1, ncols=4, layout="constrained", figsize=(3.5*4, 3))
        slab = '$s$ [m]'
        
        axs[0].plot(s_min2, X_min2, label='$x$')
        axs[0].grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        axs[0].set_xlabel(slab)
        axs[0].set_ylabel('$X$ [$\mathrm{\mu}$m]')
        ax = plt.twinx(axs[0])
        ax.plot(s_min2, Y_min2, 'orange', alpha=0.5, label='$y$')
        ax.set_ylabel('$Y$ [$\mathrm{\mu}$m]')
        lns = axs[0].get_lines() + ax.get_lines()
        labs = [l.get_label() for l in lns]
        axs[0].legend(lns, labs)
        axs[0].set_title(r'Transverse offset, slice at $\langle\xi\rangle$' f"{round(ref_slice_sigma_z_min2)}"r'$\sigma_z$')
        
        axs[1].plot(s_min1, X_min1, label='x')
        axs[1].grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        axs[1].set_xlabel(slab)
        axs[1].set_ylabel('$X$ [$\mathrm{\mu}$m]')
        ax = plt.twinx(axs[1])
        ax.plot(s_min1, Y_min1, 'orange', alpha=0.5, label='$y$')
        ax.set_ylabel('$Y$ [$\mathrm{\mu}$m]')
        lns = axs[1].get_lines() + ax.get_lines()
        labs = [l.get_label() for l in lns]
        axs[1].legend(lns, labs)
        axs[1].set_title(r'Transverse offset, slice at $\langle\xi\rangle$' f"{round(ref_slice_sigma_z_min1)}"r'$\sigma_z$')
        
        axs[2].plot(s_0, X_0, label='x')
        axs[2].grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        axs[2].set_xlabel(slab)
        axs[2].set_ylabel('$X$ [$\mathrm{\mu}$m]')
        ax = plt.twinx(axs[2])
        ax.plot(s_0, Y_0, 'orange', alpha=0.5, label='$y$')
        ax.set_ylabel('$Y$ [$\mathrm{\mu}$m]')
        lns = axs[2].get_lines() + ax.get_lines()
        labs = [l.get_label() for l in lns]
        axs[2].legend(lns, labs)
        axs[2].set_title(r'Transverse offset, slice at $\langle\xi\rangle$')
        
        axs[3].plot(s_2, X_2, label='Model')
        axs[3].grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        axs[3].set_xlabel(slab)
        axs[3].set_ylabel('$X$ [$\mathrm{\mu}$m]')
        ax = plt.twinx(axs[3])
        ax.plot(s_2, Y_2, 'orange', alpha=0.5, label='$y$')
        ax.set_ylabel('$Y$ [$\mathrm{\mu}$m]')
        lns = axs[3].get_lines() + ax.get_lines()
        labs = [l.get_label() for l in lns]
        axs[3].legend(lns, labs)
        axs[3].set_title(r'Transverse offset, slice at $\langle\xi\rangle +$' f"{round(ref_slice_sigma_z_2)}"r'$\sigma_z$')

    
    # ==================================================
    # Plots snapshots of the beam slices vs xi at various propagation distances s. TODO
    def centroid_snapshot_plots(self, beam):

        x_slices_table = self.x_slices_table_main
        y_slices_table = self.y_slices_table_main
        s_slices_table = self.s_slices_table_main
        z_slices = self.z_slices
        middle_slice_index, _ = find_closest_value_in_arr(z_slices, np.mean(z_slices))
        mean_s = s_slices_table[:,middle_slice_index]

        dN_dz, zs_dNdz = beam.longitudinal_num_density()
        dN_dz = -dN_dz
        zs_dNdz = zs_dNdz*1e6  # [um]
        
        
        # Make figure with axes
        fig, axs = plt.subplots(nrows=2, ncols=4, layout="constrained", figsize=(3.5*4, 3*2))
        fig.suptitle('Beam slice transverse offsets along x')
        xilab = r'$\xi$ [$\mathrm{\mu}$m]'
        xlab = r'$x$ [$\mathrm{\mu}$m]'

        z_slices = z_slices*1e6  # [um]
        
        snap_idx = 0
        x_slices = x_slices_table[snap_idx,:]*1e6  # [um]
        ax = axs[0][0]
        ax.plot(z_slices, x_slices, '-x')
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xlab)
        ax.set_title(f"$s={round(mean_s[snap_idx],2)}$ m")
        ax_twin = ax.twinx()
        ax_twin.fill_between(x=zs_dNdz, y1=dN_dz, y2=0, color='g', alpha=0.3)
        ax_twin.plot(zs_dNdz, dN_dz, 'g', alpha=0.5)
        ax_twin.yaxis.set_ticks([])

        snap_idx = int(len(mean_s)/8*2)
        x_slices = x_slices_table[snap_idx,:]*1e6  # [um]
        ax = axs[0][1]
        ax.plot(z_slices, x_slices, '-x')
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xlab)
        ax.set_title(f"$s={round(mean_s[snap_idx],2)}$ m")
        ax_twin = ax.twinx()
        ax_twin.fill_between(x=zs_dNdz, y1=dN_dz, y2=0, color='g', alpha=0.3)
        ax_twin.plot(zs_dNdz, dN_dz, 'g', alpha=0.5)
        ax_twin.yaxis.set_ticks([])

        snap_idx = int(len(mean_s)/8*3)
        x_slices = x_slices_table[snap_idx,:]*1e6  # [um]
        ax = axs[0][2]
        ax.plot(z_slices, x_slices, '-x')
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xlab)
        ax.set_title(f"$s={round(mean_s[snap_idx],2)}$ m")
        ax_twin = ax.twinx()
        ax_twin.fill_between(x=zs_dNdz, y1=dN_dz, y2=0, color='g', alpha=0.3)
        ax_twin.plot(zs_dNdz, dN_dz, 'g', alpha=0.5)
        ax_twin.yaxis.set_ticks([])

        snap_idx = int(len(mean_s)/8*4)
        x_slices = x_slices_table[snap_idx,:]*1e6  # [um]
        ax = axs[0][3]
        ax.plot(z_slices, x_slices, '-x')
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xlab)
        ax.set_title(f"$s={round(mean_s[snap_idx],2)}$ m")
        ax_twin = ax.twinx()
        ax_twin.fill_between(x=zs_dNdz, y1=dN_dz, y2=0, color='g', alpha=0.3)
        ax_twin.plot(zs_dNdz, dN_dz, 'g', alpha=0.5)
        ax_twin.yaxis.set_ticks([])

        snap_idx = int(len(mean_s)/8*5)
        x_slices = x_slices_table[snap_idx,:]*1e6  # [um]
        ax = axs[1][0]
        ax.plot(z_slices, x_slices, '-x')
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xlab)
        ax.set_title(f"$s={round(mean_s[snap_idx],2)}$ m")
        ax_twin = ax.twinx()
        ax_twin.fill_between(x=zs_dNdz, y1=dN_dz, y2=0, color='g', alpha=0.3)
        ax_twin.plot(zs_dNdz, dN_dz, 'g', alpha=0.5)
        ax_twin.yaxis.set_ticks([])

        snap_idx = int(len(mean_s)/8*6)
        x_slices = x_slices_table[snap_idx,:]*1e6  # [um]
        ax = axs[1][1]
        ax.plot(z_slices, x_slices, '-x')
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xlab)
        ax.set_title(f"$s={round(mean_s[snap_idx],2)}$ m")
        ax_twin = ax.twinx()
        ax_twin.fill_between(x=zs_dNdz, y1=dN_dz, y2=0, color='g', alpha=0.3)
        ax_twin.plot(zs_dNdz, dN_dz, 'g', alpha=0.5)
        ax_twin.yaxis.set_ticks([])

        snap_idx = int(len(mean_s)/8*7)
        x_slices = x_slices_table[snap_idx,:]*1e6  # [um]
        ax = axs[1][2]
        ax.plot(z_slices, x_slices, '-x')
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xlab)
        ax.set_title(f"$s={round(mean_s[snap_idx],2)}$ m")
        ax_twin = ax.twinx()
        ax_twin.fill_between(x=zs_dNdz, y1=dN_dz, y2=0, color='g', alpha=0.3)
        ax_twin.plot(zs_dNdz, dN_dz, 'g', alpha=0.5)
        ax_twin.yaxis.set_ticks([])

        snap_idx = -1
        x_slices = x_slices_table[snap_idx,:]*1e6  # [um]
        ax = axs[1][3]
        ax.plot(z_slices, x_slices, '-x')
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xlab)
        ax.set_title(f"$s={round(mean_s[snap_idx],2)}$ m")
        ax_twin = ax.twinx()
        ax_twin.fill_between(x=zs_dNdz, y1=dN_dz, y2=0, color='g', alpha=0.3)
        ax_twin.plot(zs_dNdz, dN_dz, 'g', alpha=0.5)
        ax_twin.yaxis.set_ticks([])

    
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

        if self.main_source is None:
            main_symmetrise = 'Not registered.'
        else:
            main_symmetrise = str(self.main_source.symmetrize)
        print(f"Symmetrised main beam:\t\t\t\t\t {str(self.main_source.symmetrize) :s}")
        print(f"Symmetrised drive beam:\t\t\t\t\t {str(self.driver_source.symmetrize) :s}\n")
        
        print(f"Stage length [m]:\t\t\t\t\t {self.length :.3f}")
        print(f"Plasma density [m^-3]:\t\t\t\t\t {self.plasma_density :.3e}")
        print(f"Ramp beta magnification:\t\t\t\t {self.ramp_beta_mag :.3f}")
        print(f"Drive beam x jitter (std) [um]:\t\t\t\t {self.driver_source.jitter.x*1e6 :.3f}")
        print(f"Drive beam y jitter (std) [um]:\t\t\t\t {self.driver_source.jitter.y*1e6 :.3f}")
        print('----------------------------------------------------------------------\n')
        
        print('-------------------------------------------------------------------------------------')
        print('Quantity \t\t\t\t\t Drive beam \t\t Main beam')
        print('-------------------------------------------------------------------------------------')
        print(f"Number of macroparticles:\t\t\t {len(drive_beam.xs()) :d}\t\t\t {len(main_beam.xs()) :d}")
        print(f"Initial beam population:\t\t\t {(np.sum(drive_beam.weightings())) :.3e} \t\t {(np.sum(main_beam.weightings())) :.3e}\n")

        _, z_centre = find_closest_value_in_arr(arr=main_beam.zs(), val=main_beam.z_offset())  # Centre z of beam.
        print(f"Beam centre gradient [GV/m]:\t\t\t\t  \t\t {self.Ez_fit_obj(z_centre)/1e9 :.3f}")
        print(f"Initial mean gamma:\t\t\t\t {np.mean(drive_beam.gamma()) :.3f} \t\t {np.mean(main_beam.gamma()) :.3f}")
        print(f"Initial mean energy [GeV]:\t\t\t {np.mean(drive_beam.Es())/1e9 :.3f} \t\t {np.mean(main_beam.Es())/1e9 :.3f}")
        print(f"Initial rms energy spread [%]:\t\t\t {drive_beam.rel_energy_spread()*1e2 :.3f} \t\t\t {main_beam.rel_energy_spread()*1e2 :.3f}\n")

        print(f"Initial beam x offset [um]:\t\t\t {drive_beam.x_offset()*1e6 :.3f} \t\t {main_beam.x_offset()*1e6 :.3f}")
        print(f"Initial beam y offset [um]:\t\t\t {drive_beam.y_offset()*1e6 :.3f} \t\t\t {main_beam.y_offset()*1e6 :.3f}")
        print(f"Initial beam z offset [um]:\t\t\t {drive_beam.z_offset()*1e6 :.3f} \t\t {main_beam.z_offset()*1e6 :.3f}\n")

        print(f"Initial normalised x emittance [mm mrad]:\t {drive_beam.norm_emittance_x()*1e6 :.3f} \t\t\t {main_beam.norm_emittance_x()*1e6 :.3f}")
        print(f"Initial normalised y emittance [mm mrad]:\t {drive_beam.norm_emittance_y()*1e6 :.3f} \t\t {main_beam.norm_emittance_y()*1e6 :.3f}\n")

        print(f"Initial matched beta function [mm]:\t\t\t      {self.matched_beta_function(np.mean(main_beam.Es()))*1e3 :.3f}")
        print(f"Initial x beta function [mm]:\t\t\t {drive_beam.beta_x()*1e3 :.3f} \t\t {main_beam.beta_x()*1e3 :.3f}")
        print(f"Initial y beta function [mm]:\t\t\t {drive_beam.beta_y()*1e3 :.3f} \t\t {main_beam.beta_y()*1e3 :.3f}\n")

        print(f"Initial x beam size [um]:\t\t\t {drive_beam.beam_size_x()*1e6 :.3f} \t\t\t {main_beam.beam_size_x()*1e6 :.3f}")
        print(f"Initial y beam size [um]:\t\t\t {drive_beam.beam_size_y()*1e6 :.3f} \t\t\t {main_beam.beam_size_y()*1e6 :.3f}")
        print(f"Initial rms beam length [um]:\t\t\t {drive_beam.bunch_length()*1e6 :.3f} \t\t {main_beam.bunch_length()*1e6 :.3f}")
        print(f"Initial peak current [kA]:\t\t\t {np.mean(drive_beam.peak_current())/1e3 :.3f} \t\t {np.mean(main_beam.peak_current())/1e3 :.3f}")
        print(f"Bubble radius at beam head [um]:\t\t \t\t\t {self.rb_fit_obj(np.max(main_beam.zs()))*1e6 :.3f}")
        print(f"Bubble radius at beam tail [um]:\t\t \t\t\t {self.rb_fit_obj(np.min(main_beam.zs()))*1e6 :.3f}")
        print('-------------------------------------------------------------------------------------')
        
    
    # ==================================================
    def print_current_summary(self, drive_beam, initial_main_beam, beam_out):

        with open(self.diag_path + 'output.txt', 'w') as f:
            print('===================================================', file=f)
            print(f"Time step [betatron wavelength/c]:\t {self.time_step_mod :.3f}", file=f)
            print(f"Interstages enabled:\t\t\t {str(self.interstages_enabled) :s}", file=f)
            
            if self.interstage_dipole_field is None:
                print(f"Interstage dipole field:\t\t {'Not registered.' :s}", file=f)
            elif callable(self.interstage_dipole_field):
                interstage_dipole_field = self.interstage_dipole_field(beam_out.energy())
                print(f"Interstage dipole field:\t\t {interstage_dipole_field :.3f}", file=f)
            else:
                interstage_dipole_field = self.interstage_dipole_field
                print(f"Interstage dipole field:\t\t {interstage_dipole_field :.3f}", file=f)

            if self.main_source is None:
                main_symmetrise = 'Not registered.'
            else:
                main_symmetrise = str(self.main_source.symmetrize)
            print(f"Symmetrised main beam:\t\t\t {main_symmetrise :s}", file=f)
            print(f"Symmetrised drive beam:\t\t\t {str(self.driver_source.symmetrize) :s}\n", file=f)
            
            print(f"Stage length [m]:\t\t\t {self.length :.3f}", file=f)
            print(f"Propagation length [m]:\t\t\t {beam_out.location :.3f}", file=f)
            print(f"Plasma density [m^-3]:\t\t\t {self.plasma_density :.3e}", file=f)
            print(f"Ramp beta magnification:\t\t {self.ramp_beta_mag :.3f}", file=f)
            print(f"Drive beam x jitter (std) [um]:\t\t {self.driver_source.jitter.x*1e6 :.3f}", file=f)
            print(f"Drive beam y jitter (std) [um]:\t\t {self.driver_source.jitter.y*1e6 :.3f}", file=f)
            print('---------------------------------------------------\n', file=f)
            
            print('-------------------------------------------------------------------------------------', file=f)
            print('Quantity \t\t\t\t\t Drive beam \t\t Main beam', file=f)
            print('-------------------------------------------------------------------------------------', file=f)
            print(f"Initial number of macroparticles:\t\t {len(drive_beam.xs()) :d}\t\t\t {len(initial_main_beam.xs()) :d}", file=f)
            print(f"Current number of macroparticles:\t\t  \t\t\t {len(beam_out.xs()) :d}", file=f)
            print(f"Initial beam population:\t\t\t {(np.sum(drive_beam.weightings())) :.3e} \t\t {(np.sum(initial_main_beam.weightings())) :.3e}", file=f)
            print(f"Current beam population:\t\t\t \t \t\t {(np.sum(beam_out.weightings())) :.3e}\n", file=f)

            
            _, z_centre = find_closest_value_in_arr(arr=beam_out.zs(), val=beam_out.z_offset())  # Centre z of beam.
            print(f"Beam centre gradient [GV/m]:\t\t\t\t  \t\t {self.Ez_fit_obj(z_centre)/1e9 :.3f}", file=f)
            print(f"Current mean gamma:\t\t\t\t \t \t\t {np.mean(beam_out.gamma()) :.3f}", file=f)
            print(f"Initial mean energy [GeV]:\t\t\t {np.mean(drive_beam.Es())/1e9 :.3f} \t\t {np.mean(initial_main_beam.Es())/1e9 :.3f}", file=f)
            print(f"Current mean energy [GeV]:\t\t\t \t \t\t {np.mean(beam_out.Es())/1e9 :.3f}", file=f)
            print(f"Initial rms energy spread [%]:\t\t\t {drive_beam.rel_energy_spread()*1e2 :.3f} \t\t\t {initial_main_beam.rel_energy_spread()*1e2 :.3f}", file=f)
            print(f"Current rms energy spread [%]:\t\t\t  \t\t\t {beam_out.rel_energy_spread()*1e2 :.3f}\n", file=f)
    
            print(f"Initial beam x offset [um]:\t\t\t {drive_beam.x_offset()*1e6 :.3f} \t\t {initial_main_beam.x_offset()*1e6 :.3f}", file=f)
            print(f"Initial beam y offset [um]:\t\t\t {drive_beam.y_offset()*1e6 :.3f} \t\t\t {initial_main_beam.y_offset()*1e6 :.3f}", file=f)
            print(f"Initial beam z offset [um]:\t\t\t {drive_beam.z_offset()*1e6 :.3f} \t\t {initial_main_beam.z_offset()*1e6 :.3f}\n", file=f)
    
            print(f"Initial normalised x emittance [mm mrad]:\t {drive_beam.norm_emittance_x()*1e6 :.3f} \t\t\t {initial_main_beam.norm_emittance_x()*1e6 :.3f}", file=f)
            print(f"Current normalised x emittance [mm mrad]:\t  \t\t\t {beam_out.norm_emittance_x()*1e6 :.3f}", file=f)
            print(f"Initial normalised y emittance [mm mrad]:\t {drive_beam.norm_emittance_y()*1e6 :.3f} \t\t {initial_main_beam.norm_emittance_y()*1e6 :.3f}", file=f)
            print(f"Current normalised y emittance [mm mrad]:\t \t \t\t {beam_out.norm_emittance_y()*1e6 :.3f}\n", file=f)
            
            print(f"Initial matched beta function [mm]:\t\t\t      {self.matched_beta_function(np.mean(initial_main_beam.Es()))*1e3 :.3f}", file=f)
            print(f"Initial x beta function [mm]:\t\t\t {drive_beam.beta_x()*1e3 :.3f} \t\t {initial_main_beam.beta_x()*1e3 :.3f}", file=f)
            print(f"Current x beta function [mm]:\t\t\t \t \t\t {beam_out.beta_x()*1e3 :.3f}", file=f)
            print(f"Initial y beta function [mm]:\t\t\t {drive_beam.beta_y()*1e3 :.3f} \t\t {initial_main_beam.beta_y()*1e3 :.3f}", file=f)
            print(f"Current y beta function [mm]:\t\t\t \t \t\t {beam_out.beta_y()*1e3 :.3f}\n", file=f)
    
            print(f"Initial x beam size [um]:\t\t\t {drive_beam.beam_size_x()*1e6 :.3f} \t\t\t {initial_main_beam.beam_size_x()*1e6 :.3f}", file=f)
            print(f"Current x beam size [um]:\t\t\t  \t\t\t {beam_out.beam_size_x()*1e6 :.3f}", file=f)
            print(f"Initial y beam size [um]:\t\t\t {drive_beam.beam_size_y()*1e6 :.3f} \t\t\t {initial_main_beam.beam_size_y()*1e6 :.3f}", file=f)
            print(f"Current y beam size [um]:\t\t\t  \t\t\t {beam_out.beam_size_y()*1e6 :.3f}", file=f)
            print(f"Initial rms beam length [um]:\t\t\t {drive_beam.bunch_length()*1e6 :.3f} \t\t {initial_main_beam.bunch_length()*1e6 :.3f}", file=f)
            print(f"Current rms beam length [um]:\t\t\t \t \t\t {beam_out.bunch_length()*1e6 :.3f}", file=f)
            print(f"Initial peak current [kA]:\t\t\t {np.mean(drive_beam.peak_current())/1e3 :.3f} \t\t {np.mean(initial_main_beam.peak_current())/1e3 :.3f}", file=f)
            print(f"Current peak current [kA]:\t\t\t  \t\t\t {np.mean(beam_out.peak_current())/1e3 :.3f}", file=f)
            print(f"Bubble radius at beam head [um]:\t\t \t\t\t {self.rb_fit_obj(np.max(beam_out.zs()))*1e6 :.3f}", file=f)
            print(f"Bubble radius at beam tail [um]:\t\t \t\t\t {self.rb_fit_obj(np.min(beam_out.zs()))*1e6 :.3f}", file=f)
            print('-------------------------------------------------------------------------------------', file=f)
        f.close() # Close the file

        with open(self.diag_path + 'output.txt', 'r') as f:
            print(f.read())
        f.close()
