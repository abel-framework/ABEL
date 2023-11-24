"""
Transverse wake instability model as described in thesis "Instability and Beam-Beam Study for Multi-TeV PWFA e+e- and gamma gamma Linear Colliders" (https://cds.cern.ch/record/2754022?ln=en).

Inputs
    length: [m] float. Length of the plasma stage.
    nom_energy_gain: [eV] float. Nominal/target energy gain of the acceleration stage.
    plasma_density: [m^-3] float. Plasma density.

    driver: Source object of drive beam.
    main: Source object of main beam.

    main_beam_roi: float. Determines the region of interest (also effective beam length) in units of main beam beam length.
    beam_length_roi: [m] float. Effective total beam length.
    num_beam_slice: int. Number of beam slices.
    
    #ref_slice_sigma_z: Specifies the slice to evaluate in units of sigma_z behind/in front of the beam center. Negative values for slices behind the center and vice versa. Obsolete
    
    Ez: [V/m] 1D array that contains the longitudinal electric field in the region of interest.
    bubble_radius: [m] 1D array that contains the bubble radius in the region of interest.
    num_profile: [1/m] 1D array that contains the longitudinal number profile of the beam.
    initial_offsets: [m] 1D array that contains the center x position of each slice.
    

Outputs
    ...

Ben Chen, 19 July 2023, University of Oslo
"""

import numpy as np
from scipy.constants import c, e, m_e, epsilon_0 as eps0
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # For logarithmic colour scales
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For manipulating colourbars
from matplotlib.colors import LinearSegmentedColormap  # For customising colour maps

from abel.physics_models.slices_transverse_wake_instability import *
from abel.utilities.plasma_physics import k_p, beta_matched, wave_breaking_field
from abel.utilities.relativity import energy2gamma
from abel.utilities.statistics import prct_clean, prct_clean2d
from abel.utilities.other import find_closest_value_in_arr
import abel.utilities.colors as cmaps
from abel.classes.stage.impl.stage_wake_t import StageWakeT
from openpmd_viewer import OpenPMDTimeSeries
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
#from scipy.stats import linregress
import scipy.signal as signal
import concurrent.futures  # Parallel execution

from types import SimpleNamespace
from abel import Stage, CONFIG
from abel import Beam
import warnings, copy
import os
import pickle


class StageSlicesTransWakeInstability(Stage):

    # ==================================================
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, drive_source=None, main_source=None, drive_beam=None, main_beam=None, main_beam_roi=3.0, beam_length_roi=None, num_beam_slice=None, Ez_roi=None, bubble_radius_roi=None, main_num_profile=None, zs_main_cut=None):
        
        super().__init__(length, nom_energy_gain, plasma_density)

        self.driver_source = drive_source
        self.main_source = main_source
        self.drive_beam = drive_beam
        self.main_beam = main_beam

        self.main_beam_roi = main_beam_roi
        self.beam_length_roi = beam_length_roi
        self.num_uniform_beam_slice = None  # Number of beam slices if the beam was sliced uniformly.
        self.num_beam_slice = None  # Number of beam slices.
        #self.ref_slice_sigma_z = ref_slice_sigma_z  # Obsolete

        #self.s_init = None  # [m] slice propagation coordinates. Beam head initially at s=0. s[0] corresponds to beam tail. s[-1] corresponds to beam head.

        self.Ez_roi = Ez_roi  # [V/m] longitudinal E-field in the region of interest.
        self.bubble_radius_roi = bubble_radius_roi  # [m]  bubble radius in the region of interest.
        self.Ez_axial = None
        self.zs_Ez_axial = None
        self.bubble_radius_axial = None
        self.zs_bubble_radius_axial = None
        self.main_num_profile = main_num_profile
        self.main_slices_edges = None
        self.zs_main_cut = zs_main_cut
        #self.prop_dist = prop_dist # Obsolete
        self.driver_num_profile = None
        self.zs_driver_cut = None
        self.x_slices_main = None  # [m], 1D array. x-coordinates of all beam slices.
        self.xp_slices_main = None
        self.y_slices_main = None
        self.yp_slices_main = None
        self.energy_slices_main = None
        
        self.s_slices_table_main = None  # Contains arrays of s for all the time steps in the stage.
        self.x_slices_table_main = None
        self.xp_slices_table_main = None
        self.y_slices_table_main = None
        self.yp_slices_table_main = None
        #self.norm_amp_x_start = None
        #self.norm_amp_x = None # Obsolete
        #self.mean_energy_s = None # Obsolete
        #self.energy_spread_s = None  # Obsolete

        self.diag_path = None

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
        
        plasma_density = self.plasma_density
        stage_length = self.length
        gamma0 = beam0.gamma()
        #norm_emitt = beam0.norm_emittance_x()
        #norm_emitt = beam0.geom_emittance_x()*gamma0

        # Number profile N(z). Dimensionless, same as dN/dz with each bin multiplied with the widths of the bins.
        main_num_profile, xi_slices = self.longitudinal_number_distribution(beam=beam0,  uniform_bins=False)
        s_slices_start = xi_slices + beam0.location
        self.zs_main_cut = xi_slices  # Update the longitudinal position of the beam slices.
        self.main_num_profile = main_num_profile
        
        # Sample initial main beam slice x, energy and x'-offsets
        x_slices_start = self.particles2slices(beam=beam0, beam_quant=beam0.xs(), z_slices=xi_slices)
        xp_slices_start = self.particles2slices(beam=beam0, beam_quant=beam0.xps(), z_slices=xi_slices)
        y_slices_start = self.particles2slices(beam=beam0, beam_quant=beam0.ys(), z_slices=xi_slices)
        yp_slices_start = self.particles2slices(beam=beam0, beam_quant=beam0.yps(), z_slices=xi_slices)
        energy_slices_start = self.particles2slices(beam=beam0, beam_quant=beam0.Es(), z_slices=xi_slices)
                

        #plt.figure()
        #plt.scatter(beam0.zs()*1e6, beam0.xs()*1e6)
        #plt.plot(xi_slices*1e6, x_slices_start*1e6, 'r')
        ##plt.scatter(beam0.zs()+ beam0.location, beam0.xs()*1e6)
        ##plt.plot(s_slices_start, x_slices_start*1e6, 'r')
        #plt.ylabel('x [um]')
        #
        #plt.figure()
        #plt.scatter(beam0.zs()*1e6, beam0.xps()*1e3)
        #plt.plot(xi_slices*1e6, xp_slices_start*1e3, 'r')
        ##plt.scatter(beam0.zs()+ beam0.location, beam0.xps()*1e3)
        ##plt.plot(s_slices_start, xp_slices_start*1e3, 'r')
        #plt.ylabel('x\' [mrad]')
        #
        #plt.figure()
        #plt.scatter(beam0.zs()*1e6, beam0.ys()*1e6)
        #plt.plot(xi_slices*1e6, y_slices_start*1e6, 'r')
        #plt.ylabel('y [um]')
        #
        #plt.figure()
        #plt.scatter(beam0.zs()*1e6, beam0.yps()*1e3)
        #plt.plot(xi_slices*1e6, yp_slices_start*1e3, 'r')
        #plt.ylabel('y\' [mrad]')
        #
        #plt.figure()
        #plt.scatter(beam0.zs()*1e6, beam0.Es()/1e9)
        #plt.plot(xi_slices*1e6, energy_slices_start/1e9, 'r')
        ##plt.scatter(beam0.zs()+ beam0.location, beam0.Es()/1e9)
        ##plt.plot(s_slices_start, energy_slices_start/1e9, 'r')
        #plt.ylabel('E [GeV]')

        #print(len(beam0.zs()))
        #print(len(s_slices_start))
        #print(s_slices_start)
        #print('np.mean(beam0.zs()): ', np.mean(beam0.zs()))
        #print('np.mean(xi_slices): ', np.mean(xi_slices))


        # ========== Wake-T simulation and extraction ==========
        # Define a Wake-T stage
        stage_wakeT = StageWakeT()
        stage_wakeT.driver_source = self.driver_source
        k_beta = k_p(plasma_density)/np.sqrt(2*min(gamma0,self.drive_beam.gamma()/2))
        lambda_betatron = (2*np.pi/k_beta)
        stage_wakeT.length = lambda_betatron/10  # [m]
        stage_wakeT.plasma_density = plasma_density  # [m^-3]
        stage_wakeT.box_min_z = beam0.zs().min() - 7 * beam0.bunch_length()
        stage_wakeT.box_max_z = np.mean(self.drive_beam.zs()) + 5 * self.drive_beam.bunch_length()
        stage_wakeT.opmd_diag = True  # Set to True for saving simulation results.
        stage_wakeT.diag_dir = self.diag_path + 'wake_t'
        
        # Run the Wake-T stage
        beam_copy = copy.deepcopy(beam0)  # Make a deep copy of beam0 to avoid changes on beam0.
        beam_wakeT = stage_wakeT.track(beam_copy)
        
        # Read the Wake-T simulation data
        path_sep = os.sep
        path = stage_wakeT.diag_dir + path_sep + 'hdf5'
        ts = OpenPMDTimeSeries(path)
        time = ts.t[0]  # Extract first time step dump.

        # Extract longitudinal E-field
        Ez_wakeT, info_Ez = ts.get_field(field='E', coord='z', iteration=0, plot=False)
        zs_Ez_wakeT = info_Ez.z
        rs_Ez = info_Ez.r
        # Extract axial longitudinal E-field
        Ez_axis_wakeT = Ez_wakeT[round(len(info_Ez.r)/2),:]

        # Cut out axial Ez over the ROI
        Ez, _ = self.Ez_shift_fit(Ez_axis_wakeT, zs_Ez_wakeT, xi_slices, beam0)
        self.Ez_roi = Ez
        self.Ez_axial = Ez_axis_wakeT
        self.zs_Ez_axial = zs_Ez_wakeT
        
        # Extract plasma charge density
        rho, info_rho = ts.get_field(field='rho', iteration=0, plot=False)
        
        # Calculate the number density
        plasma_num_density = rho/stage_wakeT.plasma_density/-e
        
        # Extract coordinates
        zs_rho = info_rho.z
        rs_rho = info_rho.r
        
        # Extract the plasma bubble radius
        bubble_radius_wakeT = self.get_bubble_radius(plasma_num_density, rs_rho, threshold=0.8)

        # Cut out bubble radius over the ROI
        bubble_radius, _ = self.rb_shift_fit(bubble_radius_wakeT, zs_rho, xi_slices, beam0) # Actually same as Ez_shift_fit. Consider making just one function instead... 
        self.bubble_radius_roi = bubble_radius
        self.bubble_radius_axial = bubble_radius_wakeT
        self.zs_bubble_radius_axial = zs_rho

        # Make plots for control if necessary
        #self.plot_Ez_rb_cut(xi_slices, main_num_profile, zs_Ez_wakeT, Ez_axis_wakeT, Ez, zs_rho, bubble_radius_wakeT, bubble_radius, zlab='$z$ [$\mathrm{\mu}$m]')
        

        # ========== Instability tracking ==========
        inputs = [plasma_density, Ez, bubble_radius, main_num_profile, x_slices_start, xp_slices_start, y_slices_start, yp_slices_start, energy_slices_start, stage_length, s_slices_start, xi_slices]
        some_are_none = any(input is None for input in inputs)
        
        if some_are_none:
            none_indices = [i for i, x in enumerate(inputs) if x is None]
            print(none_indices)
            raise ValueError('At least one input is set to None.')

        #s_slices_table, x_slices_table, xp_slices_table, energy_slices = transverse_wake_instability(plasma_density, Ez, bubble_radius, main_num_profile, x_slices_start, xp_slices_start, energy_slices_start, stage_length, s_slices_start, xi_slices)
        #_, y_slices_table, yp_slices_table, _ = transverse_wake_instability(plasma_density, Ez, bubble_radius, main_num_profile, y_slices_start, yp_slices_start, energy_slices_start, stage_length, s_slices_start, xi_slices)

        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(transverse_wake_instability_slices, plasma_density, Ez, bubble_radius, main_num_profile, x_slices_start, xp_slices_start, energy_slices_start, stage_length, s_slices_start, xi_slices)
            future2 = executor.submit(transverse_wake_instability_slices, plasma_density, Ez, bubble_radius, main_num_profile, y_slices_start, yp_slices_start, energy_slices_start, stage_length, s_slices_start, xi_slices)
        s_slices_table, x_slices_table, xp_slices_table, energy_slices = future1.result()
        _, y_slices_table, yp_slices_table, _  = future2.result()
        
        s_slices = s_slices_table[-1,:]
        x_slices = x_slices_table[-1,:]
        xp_slices = xp_slices_table[-1,:]
        y_slices = y_slices_table[-1,:] ##############
        yp_slices = yp_slices_table[-1,:] ##############
        #y_slices = y_slices_start #################
        #yp_slices = yp_slices_start ################


        # ========== Manipulate particle beam according to slice quantities ==========
        energy_gains = self.beam_energy_gains_interpolate(xi_slices, energy_slices_start, energy_slices, beam0)  # [eV] energy gains of each particle.
        gammas_after = energy2gamma(beam0.Es() + energy_gains)
        #beam = self.betatron_damping(gamma0, gammas_after, xi_slices, x_slices, xp_slices, y_slices, yp_slices, beam0)  # Apply betatron damping to beam particles around the slice centroids.
        
        ux_slices_before = self.particles2slices(beam=beam0, beam_quant=beam0.uxs(), z_slices=xi_slices)  # Proper x velocity before accleration.
        uy_slices_before = self.particles2slices(beam=beam0, beam_quant=beam0.uys(), z_slices=xi_slices)  # Proper y velocity before accleration.
        uz_slices_before = self.particles2slices(beam=beam0, beam_quant=beam0.uzs(), z_slices=xi_slices)  # Proper z velocity before accleration.
        
        beam0.accelerate(energy_gains)
        
        ux_slices_after = self.particles2slices(beam=beam0, beam_quant=beam0.uxs(), z_slices=xi_slices)  # Proper x velocity after accleration.
        uy_slices_after = self.particles2slices(beam=beam0, beam_quant=beam0.uys(), z_slices=xi_slices)  # Proper y velocity after accleration.
        uz_slices_after = self.particles2slices(beam=beam0, beam_quant=beam0.uzs(), z_slices=xi_slices)  # Proper z velocity after accleration.

        # Scale slice starting angles using proper velocities
        xp_slices_start = ux_slices_after*uz_slices_before/ux_slices_before/uz_slices_after * xp_slices_start ################
        yp_slices_start = uy_slices_after*uz_slices_before/uy_slices_before/uz_slices_after * yp_slices_start ################
        if np.any(np.isnan(xp_slices_start)):
            xp_slices_start = self.fill_nan_with_mean(xp_slices_start)  # Replace the nan values with the mean.
        if np.any(np.isnan(yp_slices_start)):
            yp_slices_start = self.fill_nan_with_mean(yp_slices_start)  # Replace the nan values with the mean.
        
        xp_slices = ux_slices_after*uz_slices_before/ux_slices_before/uz_slices_after * xp_slices
        yp_slices = uy_slices_after*uz_slices_before/uy_slices_before/uz_slices_after * yp_slices
        if np.any(np.isnan(xp_slices)):
            xp_slices = self.fill_nan_with_mean(xp_slices)  # Replace the nan values with the mean.
        if np.any(np.isnan(yp_slices)):
            yp_slices = self.fill_nan_with_mean(yp_slices)  # Replace the nan values with the mean.
        
        beam = self.slices2beam_interpolate(xi_slices, x_slices_start, xp_slices_start, y_slices_start, yp_slices_start, energy_gains, s_slices, x_slices, xp_slices, y_slices, yp_slices, beam0)  # Kick the beam particles using the centroids as guidelines.
        
        beam = self.betatron_damping(gamma0, gammas_after, xi_slices, x_slices, xp_slices, y_slices, yp_slices, beam)  # Apply betatron damping to beam particles around the slice centroids.
        
        #x_slices_squeezed = self.particles2slices(beam=beam, beam_quant=beam.xs(), z_slices=xi_slices)
        #xp_slices_squeezed = self.particles2slices(beam=beam, beam_quant=beam.xps(), z_slices=xi_slices)
        #ux_slices_squeezed = self.particles2slices(beam=beam, beam_quant=beam.uxs(), z_slices=xi_slices)
        
        #plt.figure()
        #plt.scatter(beam.zs()*1e6, beam.xs()*1e6)
        #plt.plot(xi_slices*1e6, x_slices*1e6, 'r', label='Before betatron damping')
        #plt.plot(xi_slices*1e6, x_slices_squeezed*1e6, 'kx', label='After betatron damping')
        #plt.xlabel(r'$\xi$ $\mathrm{\mu}$m]')
        #plt.ylabel('$x$ [um]')
        #plt.legend()
        #
        #plt.figure()
        #plt.scatter(beam.zs()*1e6, beam.xps()*1e3)
        #plt.plot(xi_slices*1e6, xp_slices*1e3, 'r', label='Before betatron damping')
        #plt.plot(xi_slices*1e6, xp_slices_squeezed*1e3, 'kx', label='After betatron damping')
        #plt.xlabel(r'$\xi$ $\mathrm{\mu}$m]')
        #plt.ylabel('$x\'$ [mrad]')
        #plt.legend()
        #
        #plt.figure()
        #plt.scatter(beam.zs()*1e6, beam.uxs())
        #plt.plot(xi_slices*1e6, xp_slices*self.particles2slices(beam=beam, beam_quant=beam.uzs(), z_slices=xi_slices), 'r', label='Before betatron damping')
        #plt.plot(xi_slices*1e6, ux_slices_squeezed, 'kx', label='After betatron damping')
        #plt.xlabel(r'$\xi$ $\mathrm{\mu}$m]')
        #plt.ylabel('$u_x$ [?]')
        #plt.legend()

        
        #beam.accelerate(energy_gains)
        #beam = self.slices2beam_interpolate(xi_slices, x_slices_start, xp_slices_start, y_slices_start, yp_slices_start, energy_gains, s_slices, x_slices, xp_slices, y_slices, yp_slices, beam)

        
        self.main_beam = copy.deepcopy(beam)  # Need to make a deepcopy, or changes to beam may affect the Beam object saved here.
        self.x_slices_main = x_slices  # Update the transverse offsets of the beam slices.
        self.xp_slices_main = xp_slices  # [rad] Update the x's of the beam slices.
        self.y_slices_main = y_slices  # Update the transverse offsets of the beam slices.
        self.yp_slices_main = yp_slices  # [rad] Update the x's of the beam slices.
        self.energy_slices_main = energy_slices  # Update the energies of the beam slices.
        self.s_slices_table_main = s_slices_table
        self.x_slices_table_main = x_slices_table
        self.xp_slices_table_main = xp_slices_table
        self.y_slices_table_main = y_slices_table
        self.yp_slices_table_main = yp_slices_table
        
        
        # Copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        #print(np.mean(beam.zs()))
        return super().track(beam, savedepth, runnable, verbose)

    
    # ==================================================
    def slices2beam_interpolate(self, s_slices_start, x_slices_start, xp_slices_start, y_slices_start, yp_slices_start, energy_gains, s_slices_new, x_slices_new, xp_slices_new, y_slices_new, yp_slices_new, beam0):
        """
        Converts beam slices from transverse_wake_instability() to an ABEL beam object by applying the beam offsets in each slice to all particles in the slices.
        ...
        """
        
        zs = beam0.zs()
        xs = beam0.xs()
        xps = beam0.xps()
        ys = beam0.ys()
        yps = beam0.yps()
        #Es = beam0.Es()  # [eV] obsolete

        #print(np.mean(zs))
        #print('np.mean(s_slices_start): ', np.mean(s_slices_start))

        # Interpolate
        xs_start = self.centroid_interpolate(s_slices_start, x_slices_start, zs)
        xs_final = self.centroid_interpolate(s_slices_start, x_slices_new, zs)
        ys_start = self.centroid_interpolate(s_slices_start, y_slices_start, zs)
        ys_final = self.centroid_interpolate(s_slices_start, y_slices_new, zs)
        
        xps_start = self.centroid_interpolate(s_slices_start, xp_slices_start, zs)
        xps_final = self.centroid_interpolate(s_slices_start, xp_slices_new, zs)
        yps_start = self.centroid_interpolate(s_slices_start, yp_slices_start, zs)
        yps_final = self.centroid_interpolate(s_slices_start, yp_slices_new, zs)
                
        xs = xs + xs_final - xs_start
        xps = xps + xps_final - xps_start
        ys = ys + ys_final - ys_start
        yps = yps + yps_final - yps_start
        #Es = Es + energy_gains  # obsolete, so is energy_gains.
        
        beam0.set_xs(xs)
        beam0.set_ys(ys)
        
        #from abel.utilities.relativity import energy2proper_velocity
        #beam0.set_uzs(energy2proper_velocity(Es))

        beam0.set_xps(xps)
        beam0.set_yps(yps)
        
        return beam0
    

    # ==================================================
    def centroid_interpolate(self, s_slices, slices_arr, z_particles):
        fitted_particle_arr = np.interp(z_particles, s_slices, slices_arr)
        if np.any(np.isnan(fitted_particle_arr)):
            fitted_particle_arr = self.fill_nan_with_mean(fitted_particle_arr)  # Replace the nan values with the mean.
        
        #plt.figure()
        #plt.scatter(z_particles, fitted_particle_arr, label='particles', alpha=0.5)
        #plt.plot(s_slices, slices_arr, 'k', label='Slices', alpha=0.5)
        #plt.legend()
        
        return fitted_particle_arr

    
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
            
        beam_quant: 1D float array
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
        beam_quant_slices= self.fill_nan_with_mean(beam_quant_slices)  # Replace the nan values with the mean.
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
    def betatron_damping(self, gammas_before, gammas_after, s_slices, x_slices, xp_slices, y_slices, yp_slices, beam):
        
        mag = (gammas_before/gammas_after)**(1/4)
        xs_centroids = self.centroid_interpolate(s_slices, x_slices, beam.zs())
        xps_centroids = self.centroid_interpolate(s_slices, xp_slices, beam.zs())
        uxs_centroids = xps_centroids * beam.uzs()
        ys_centroids = self.centroid_interpolate(s_slices, y_slices, beam.zs())
        yps_centroids = self.centroid_interpolate(s_slices, yp_slices, beam.zs())
        uys_centroids = yps_centroids * beam.uzs()
        
        beam.set_xs(xs_centroids + (beam.xs()-xs_centroids) * mag)
        beam.set_ys(ys_centroids + (beam.ys()-ys_centroids) * mag)
        
        beam.set_uxs(uxs_centroids + (beam.uxs()-uxs_centroids) / mag)
        beam.set_uys(uys_centroids + (beam.uys()-uys_centroids) / mag)

        return beam
        
    
    # ==================================================
    def beam_energy_gains_interpolate(self, s_slices, energy_slices_start, energy_slices, beam0):
        
        #energy_slices_start = self.energy_slices_main  # The current energies of the beam slices.
        zs = beam0.zs()
        Es = beam0.Es()  # [eV]

        #energy_slices_start_fit = interp1d(zs_num_profile, energy_slices_start, kind='cubic', bounds_error=False, fill_value='extrapolate')  # Set bounds_error=False to allow extrapolation.
        energy_slices_start_fit = interp1d(s_slices, energy_slices_start, kind='slinear', bounds_error=False, fill_value='extrapolate')  # Set bounds_error=False to allow extrapolation.
        energy_slices_fit = interp1d(s_slices, energy_slices, kind='slinear', bounds_error=False, fill_value="extrapolate")

        # Push all z-coordinates to the new position
        Es_boosted = Es + energy_slices_fit(zs) - energy_slices_start_fit(zs)

        #plt.figure()
        #plt.scatter(zs, Es_boosted - Es)
        #plt.plot(s_slices, energy_slices_fit(s_slices)-energy_slices_start_fit(s_slices), 'r')
        #plt.plot(s_slices, energy_slices_fit(s_slices)-energy_slices_start_fit(zs_num_profile))
        
        return Es_boosted - Es

    
    # ==================================================
    def Ez_shift_fit(self, Ez, zs_Ez, z_slices, beam):
        """
        Shift longitudinal E-field Ez longitudinally and make a fit to slice co-moving coordinates z_slices.

        Parameters
        ----------
        Ez: [V/m] 1D array
            Axial longitudinal E-field.
            
        zs_Ez: [m] 1D float array
            zs_Ez-coordinates for Ez. Monotonically increasing from first to last element.
            
        z_slices: [m] 1D float array
            Co-moving coordinates of the beam slices.

            
        Returns
        ----------
        Ez_fit: [V/m] 1D array
            Axial Ez for the region of interest shifted to the location of the beam.

        sse_Ez: [V^2/m^2] float 
            Sum of squared errors (sse) of Ez_fit vs. the corresponding part of Ez.
        """
        
        sigma_z = self.main_source.bunch_length  # [m]
        z_offset_main = np.mean(beam.zs())
        width = self.main_beam_roi  # Determines the width of the extraction in units of bunch lengths.
        
        # Start index (head of the beam) of extraction interval.
        head_idx, closest_val = find_closest_value_in_arr(arr=zs_Ez, val=z_offset_main+width*sigma_z)
        # End index (tail of the beam) of extraction interval.
        tail_idx, closest_val = find_closest_value_in_arr(arr=zs_Ez, val=z_offset_main-width*sigma_z)
        
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
    def get_bubble_radius(self, plasma_num_density, x, threshold=0.8):
        """
        - For extracting the plasma ion bubble radius by finding the coordinates in which the plasma number density goes from zero to a threshold value.
        - The symmetry axis is determined using the x-offset of the drive beam.
        - xi is the propagation direction pointing to the right.
        
        Parameters
        ----------
        plasma_num_density: [n0] 2D float array
            Plasma number density in units of initial number density n0. Need to be oriented with propagation direction pointing to the right and positive x pointing upwards.
            
        x: [m] 1D float array 
            Transverse coordinate of plasma_num_density.
            
        threshold: float
            Defines a threshold for the plasma density to determine bubble_radius.

            
        Returns
        ----------
        bubble_radius: [m] 1D array 
            Plasma bubble radius over the simulation box.
        """

        # Find the offsets of the beams
        drive_beam = self.drive_beam
        x_offset_driver = np.mean(drive_beam.xs())
        main_beam = self.main_beam
        x_offset_main = np.mean(main_beam.xs())
        
        # Find the value in x closest to x_offset_driver and corresponding index
        idx_middle, _ = find_closest_value_in_arr(x, x_offset_driver)
        
        # Find the value in x closest to x_offset_main and corresponding index
        idx_offset, _ = find_closest_value_in_arr(x, x_offset_main)

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
   
                    # Find the value in idxs_slice_above_thres closest to idx_offset (equivalent to the value in plasma_num_density above threshold that is closest to the main beam offset axis)
                    _, idx = find_closest_value_in_arr(idxs_slice_above_thres, idx_offset)
                    bubble_radius[i] = np.abs(x[idx] - x_offset_driver)
        
        return bubble_radius

    
    # ==================================================
    def rb_shift_fit(self, rb, zs_rb, z_slices, beam):
        """
        Shift longitudinal bubble radius longitudinally and make a fit to slice co-moving coordinates z_slices.

        Parameters
        ----------
        rb: [m] 1D array
            Plasma ion bubble radius.
            
        zs_rb: [m] 1D float array
            z-coordinates for rb. Monotonically increasing from first to last element.
            
        z_slices: [m] 1D float array
            Co-moving coordinates of the beam slices.

            
        Returns
        ----------
        rb_fit: [m] 1D float array
            Plasma ion bubble radius for the region of interest shifted to the location of the beam.

        sse_rb: [m^2] float 
            Sum of squared errors (sse) of rb_fit vs. the corresponding part of rb.
        """
        
        sigma_z = self.main_source.bunch_length  # [m]
        z_offset_main = np.mean(beam.zs())
        width = self.main_beam_roi  # Determines the width of the extraction in units of bunch lengths.
        
        # Start index (head of the beam) of extraction interval.
        head_idx, closest_val = find_closest_value_in_arr(arr=zs_rb, val=z_offset_main+width*sigma_z)
        # End index (tail of the beam) of extraction interval.
        tail_idx, closest_val = find_closest_value_in_arr(arr=zs_rb, val=z_offset_main-width*sigma_z)
        
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
        cut_off=self.main_beam_roi*self.main_beam.bunch_length()
        bool_indices = (zs <= np.mean(zs) + cut_off) & (zs >= np.mean(zs) - cut_off)
        zs_filtered = zs[bool_indices]
        q3, q1 = np.percentile(zs_filtered, [75 ,25])
        iqr = q3 - q1  # Calculate the interquartile range
        beam_slice_thickness = 2*iqr*len(zs_filtered)**(-1/3)
        num_uniform_beam_slice = int(np.round(self.beam_length_roi / beam_slice_thickness))
        return num_uniform_beam_slice


    # ==================================================
    # Return the longitudinal number distribution using the beam particles' z-coordinates.
    def longitudinal_number_distribution(self, beam, bin_number=None, cut_off=None, uniform_bins=True, make_plot=False):

        if bin_number is None:
            bin_number = self.num_uniform_beam_slice
        if cut_off is None:
            cut_off = self.main_beam_roi * beam.bunch_length()
            
        zs = beam.zs()

        # Filter out beam particles outside the region of interest
        mean_z = np.mean(zs)
        bool_indices = (zs <= mean_z + cut_off) & (zs >= mean_z - cut_off)
        zs = zs[bool_indices]
        weights = beam.weightings()  # The weight of each macroparticle.
        weights = weights[bool_indices]

        if uniform_bins is False:
            sigma_z = beam.bunch_length()

            _, thin_bins = np.histogram(zs, weights=weights, bins=bin_number)  # Compute the histogram of zs using bin_number bins.
            bool_indices = thin_bins <= mean_z + 1.0*sigma_z
            thin_bins = thin_bins[bool_indices]  # Only keep thin slices from the tail to mean_z + 1*sigma_z.
    
            # Set up bins for the beam head that gets wider towards the beam head
            arr_end = np.log(1.01*zs.max()/thin_bins[-1])
            if arr_end < 0:  # For the case when the beam is entirely placed at negative zs.
                const = -1.1*zs.max()
                arr = np.linspace(np.log(thin_bins[-1]-0.99*zs.max()+const), np.log(const), int((bin_number-len(thin_bins))/1.5))
                thick_bins = np.exp(arr) + 0.99*zs.max() - const  # Make thick slices for the front part of the beam.
            else:
                arr = np.linspace(0, arr_end, int((bin_number-len(thin_bins))/1.5))
                thick_bins = thin_bins[-1] * np.exp(arr)  # Make thick slices for the front part of the beam.
            
            #print(arr)
            #print(thick_bins)
            #print(np.log(thin_bins[-1]/zs.max()))
            #print('thick_bins[-1]',thick_bins[-1])
            #print('zs.max()',zs.max())
            #print('thin_bins[-1]',thin_bins[-1])
            #print('thick_bins[0]',thick_bins[0])
    
            #plt.figure()
            #plt.plot(arr, thick_bins)
            
            thick_bins = np.delete(thick_bins, 0)
            num_profile, edges = np.histogram(zs, weights=weights, bins=np.concatenate((thin_bins, thick_bins)))  # Compute the histogram of zs using the defined bins.
        else:
            num_profile, edges = np.histogram(zs, weights=weights, bins=bin_number)  # Compute the histogram of z using bin_number bins.
        
        self.main_slices_edges = edges
        
        z_ctrs = (edges[0:-1] + edges[1:])/2  # Centres of the bins (zs).

        #print('len(z_ctrs):', len(z_ctrs))        

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
        return beta_matched(self.plasma_density, energy)
        

    # ==================================================
    def beta_wavenumber_slices(self, beam=None, clean=False):

        if beam is None:
            beam = self.main_beam
        xs, xps = prct_clean2d(beam.xs(), beam.xps(), clean)
        zs = prct_clean(beam.zs(), clean)
        z_slices = self.zs_main_cut
        
        # Sort the arrays
        indices = np.argsort(zs)
        zs_sorted = zs[indices]
        xs_sorted = xs[indices]
        xps_sorted = xps[indices]
        
        edges = self.main_slices_edges
        k_beta = np.zeros(len(edges)-1)

        for i in range(0, len(edges)-1):
            left = np.searchsorted(zs_sorted, edges[i])  # zs_sorted[left:len(zs_sorted)] >= edges[i].
            right = np.searchsorted(zs_sorted, edges[i+1], side='right')  # zs_sorted[0:right] <= edges[i+1].
            #print(i, left, right)
            xs_in_bin = xs_sorted[left:right]
            xps_in_bin = xps_sorted[left:right]
            covx = np.cov(xs_in_bin, xps_in_bin)
            k_beta[i] = 1/(covx[0,0]/np.sqrt(np.linalg.det(covx)))

        #plt.figure()
        #plt.plot(self.zs_main_cut*1e6, k_beta, 'x-')
        #for edge in edges:
        #    plt.axvline(x=edge*1e6, color=(0.5, 0.5, 0.5), linestyle='-', alpha=0.3)
        
        return k_beta
        
        ##covx = np.cov(xs, xps)
        ##return covx[0,0]/np.sqrt(np.linalg.det(covx))
        #energy_slices = self.energy_slices_main
        #beta_slices = beta_matched(self.plasma_density, energy_slices)
        #return 1/beta_slices
    
    
    # ==================================================
    # Calculate the normalised amplitude (Lambda).
    def calc_norm_amp(self, particle_offsets, particle_angles):

        beam_size = np.std(particle_offsets)
        beam_size_angle = np.std(particle_angles)

        return np.sum((particle_offsets/beam_size)**2 + (particle_angles/beam_size_angle)**2)

    
    # ==================================================
    # Save the object to a file, obsolete
    def save(self, save_dir, filename='stage.pkl'):
        path = save_dir + filename
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    
    # ==================================================
    # Load an object from a file, obsolete
    def load(self, diag_dir, filename='stage.pkl'):
        path = diag_dir + filename
        with open(path, 'rb') as f:
            object = pickle.load(f)
        return object

    
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
    def plot_wakefield(self, beam=None, saveToFile=None, includeWakeRadius=True):
        
        # Get wakefield
        Ezs = self.Ez_axial
        zs_Ez = self.zs_Ez_axial
        zs_rho =  self.zs_bubble_radius_axial
        bubble_radius = self.bubble_radius_axial
        
        
        # get current profile
        driver = copy.deepcopy(self.__get_initial_driver())
        driver += beam  # Add beam to drive beam.
        Is, ts = driver.current_profile(bins=np.linspace(min(zs_Ez/c), max(zs_Ez/c), int(np.sqrt(len(driver))/2)))
        zs0 = ts*c
        
        # plot it
        fig, axs = plt.subplots(1, 2+int(includeWakeRadius))
        fig.set_figwidth(CONFIG.plot_fullwidth_default*(2+int(includeWakeRadius))/3)
        fig.set_figheight(4)
        col0 = "xkcd:light gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        af = 0.1
        zlims = [min(zs_Ez)*1e6*1.9, max(zs_Ez)*1e6*1.1]
        
        axs[0].plot(zs_Ez*1e6, np.zeros(zs_Ez.shape), '-', color=col0)
        if self.nom_energy_gain is not None:
            axs[0].plot(zs_Ez*1e6, -self.nom_energy_gain/self.get_length()*np.ones(zs_Ez.shape)/1e9, ':', color=col0)
        axs[0].plot(zs_Ez*1e6, Ezs/1e9, '-', color=col1)
        axs[0].set_xlabel('z (um)')
        axs[0].set_ylabel('Longitudinal electric field (GV/m)')
        axs[0].set_xlim(zlims)
        axs[0].set_ylim(bottom=-wave_breaking_field(self.plasma_density)/1e9, top=1.3*max(Ezs)/1e9)
        
        axs[1].fill(np.concatenate((zs0, np.flip(zs0)))*1e6, np.concatenate((-Is, np.zeros(Is.shape)))/1e3, color=col1, alpha=af)
        axs[1].plot(zs0*1e6, -Is/1e3, '-', color=col1)
        axs[1].set_xlabel('z (um)')
        axs[1].set_ylabel('Beam current (kA)')
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=0, top=1.2*max(-Is)/1e3)
        
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
    def distribution_plot_2D(self, arr1, arr2, weights=None, hist_bins=None, hist_range=None, axes=None, extent=None, vmin=None, vmax=None, colmap='viridis', xlab='', ylab='', clab='', origin='lower', interpolation='nearest'):

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
                  xlab=xlab, ylab=ylab, clab=clab, gridOn=False, origin=origin, interpolation=interpolation)

    
    # ==================================================
    def density_map_diags(self, beam=None, plot_centroids=False):

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

        cmap = cmaps.FLASHForward
        
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
            axs[0][0].plot(self.zs_main_cut*1e6, self.x_slices_main*1e6, 'r', alpha=0.5)
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
            axs[0][1].plot(self.zs_main_cut*1e6, self.xp_slices_main*1e3, 'r', alpha=0.5)
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
            axs[1][0].plot(self.zs_main_cut*1e6, self.y_slices_main*1e6, 'r', alpha=0.5)
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
            axs[1][1].plot(self.zs_main_cut*1e6, self.yp_slices_main*1e3, 'r', alpha=0.5)
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
        hist_range_xpyp = [[None, None], [None, None]]
        hist_range_xpyp[0] = hist_range_xps[1]
        hist_range_xpyp[1] = hist_range_yyp[1]
        extent_xpyp = hist_range_xpyp[0] + hist_range_xpyp[1]
        extent_xpyp[0] = extent_xpyp[0]*1e3  # [mrad]
        extent_xpyp[1] = extent_xpyp[1]*1e3  # [mrad]
        extent_xpyp[2] = extent_xpyp[2]*1e3  # [mrad]
        extent_xpyp[3] = extent_xpyp[3]*1e3  # [mrad]
        self.distribution_plot_2D(arr1=xps, arr2=yps, weights=weights, hist_bins=hist_bins, hist_range=hist_range_xpyp, axes=axs[2][1], extent=extent_xpyp, vmin=None, vmax=None, colmap=cmap, xlab=xps_lab, ylab=yps_lab, clab='$\partial^2 N/\partial x\' \partial y\'$ [$\mathrm{rad}^{-2}$]', origin='lower', interpolation='nearest')

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
            axs[2][2].plot(self.zs_main_cut*1e6, self.energy_slices_main/1e9, 'r', alpha=0.3)
            axs[2][2].axis(extent_energ)

    
    # ==================================================
    def scatter_diags(self, beam=None, plot_centroids=True, n_th_particle=1, show_slice_grid=False, plot_k_beta=False):
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
        plt.tight_layout(pad=4.0)  # Sets padding between the figure edge and the edges of subplots, as a fraction of the font size.
        fig.subplots_adjust(top=0.85)  # By setting top=..., you are specifying that the top boundary of the subplots should be at ...% of the figure’s height.
        
        # 2D z-x distribution
        ax = axs[0][0]
        p = ax.scatter(zs[::n_th_particle]*1e6, xs[::n_th_particle]*1e6, c=Es[::n_th_particle]/1e9, cmap=cmap)
        #ax.axis(extent)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xlab)
        if plot_centroids is True:
            X_slices_from_beam = self.particles2slices(beam=beam, beam_quant=xs, z_slices=self.zs_main_cut)
            ax.plot(self.zs_main_cut*1e6, X_slices_from_beam*1e6, 'cx', alpha=0.7, label='Slice mean $x$ from particles')
            ax.plot(self.zs_main_cut*1e6, self.x_slices_main*1e6, 'b', alpha=0.5, label='Slice mean $x$ from tracking')
            ax.legend()
        if show_slice_grid is True:
            for z_slice in self.zs_main_cut:
                ax.axvline(x=z_slice*1e6, color=(0.5, 0.5, 0.5), linestyle='-', alpha=0.3)
        if plot_k_beta is True:
            ax_twin = ax.twinx()
            #ax_Ez_cut_wakeT2 = axs_wakeT_cut[0].twinx()
            k_beta = self.beta_wavenumber_slices(beam=beam, clean=False)
            ax_twin.plot(self.zs_main_cut*1e6, k_beta, 'g')
            ax_twin.set_ylabel(r'$k_\beta$ [$\mathrm{m}^{-1}$]')
            
        # 2D z-x' distribution
        ax = axs[0][1]
        ax.scatter(zs[::n_th_particle]*1e6, xps[::n_th_particle]*1e3, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xilab)
        ax.set_ylabel(xps_lab)
        if plot_centroids is True:
            xp_slices_from_beam = self.particles2slices(beam=beam, beam_quant=xps, z_slices=self.zs_main_cut)
            ax.plot(self.zs_main_cut*1e6, xp_slices_from_beam*1e3, 'cx', alpha=0.7, label='Slice mean $x\'$ from particles')
            ax.plot(self.zs_main_cut*1e6, self.xp_slices_main*1e3, 'b', alpha=0.5, label='Slice mean $x\'$ from tracking')
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
            y_slices_from_beam = self.particles2slices(beam=beam, beam_quant=ys, z_slices=self.zs_main_cut)
            ax.plot(self.zs_main_cut*1e6, y_slices_from_beam*1e6, 'cx', alpha=0.7, label='Slice mean $y$ from particles')
            ax.plot(self.zs_main_cut*1e6, self.y_slices_main*1e6, 'b', alpha=0.5, label='Slice mean $y$ from tracking')
            ax.legend()
        if show_slice_grid is True:
            for z_slice in self.zs_main_cut:
                ax.axvline(x=z_slice*1e6, color=(0.5, 0.5, 0.5), linestyle='-', alpha=0.5)
            
        # 2D z-y' distribution
        ax = axs[1][1]
        ax.scatter(zs[::n_th_particle]*1e6, yps[::n_th_particle]*1e3, c=Es[::n_th_particle]/1e9, cmap=cmap)
        ax.set_xlabel(xilab)
        ax.set_ylabel(yps_lab)
        if plot_centroids is True:
            yp_slices_from_beam = self.particles2slices(beam=beam, beam_quant=yps, z_slices=self.zs_main_cut)
            ax.plot(self.zs_main_cut*1e6, yp_slices_from_beam*1e3, 'cx', alpha=0.7, label='Slice mean $y\'$ from particles')
            ax.plot(self.zs_main_cut*1e6, self.yp_slices_main*1e3, 'b', alpha=0.5, label='Slice mean $y\'$ from tracking')
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
        
        # 2D z-energy distribution
        ax = axs[2][2]
        ax.scatter(zs[::n_th_particle]*1e6, Es[::n_th_particle]/1e9)
        ax.set_xlabel(xilab)
        ax.set_ylabel(energ_lab)
        if plot_centroids is True:
            energy_slices_from_beam = self.particles2slices(beam=beam, beam_quant=beam.Es(), z_slices=self.zs_main_cut)
            ax.plot(self.zs_main_cut*1e6, energy_slices_from_beam/1e9, 'kx', alpha=0.7, label='Slice mean $\mathcal{E}$ from particles')
            ax.plot(self.zs_main_cut*1e6, self.energy_slices_main/1e9, 'r', alpha=0.5, label='Slice mean $\mathcal{E}$ from tracking')
            ax.legend()
            
        # Set label and other properties for the colorbar
        fig.suptitle('$\Delta s=$' f'{format(beam.location, ".2f")}' ' m')
        cbar_ax = fig.add_axes([0.15, 0.91, 0.7, 0.02])   # The four values in the list correspond to the left, bottom, width, and height of the new axes, respectively.
        fig.colorbar(p, cax=cbar_ax, orientation='horizontal', label=energ_lab)
        

    # ==================================================
    # Add plots with plasma charge density overlayed with beams, beam profiles and Ez

    # ==================================================
    # Add plots for diagnosing the beam evolution inside a stage. Also interpolate the beam particles to the spine and compare end results against interpolation outside the stage.

    
    # ==================================================
    def plot_Ez_rb_cut(self, z_slices=None, main_num_profile=None, zs_Ez=None, Ez=None, Ez_cut=None, zs_rho=None, bubble_radius=None, bubble_radius_cut=None, zlab='$z$ [$\mathrm{\mu}$m]'):

        if z_slices is None:
            z_slices = self.zs_main_cut
        if main_num_profile is None:
            main_num_profile = self.main_num_profile
        if zs_Ez is None:
            zs_Ez = self.zs_Ez_axial
        if Ez is None:
            Ez = self.Ez_axial
        if Ez_cut is None:
            Ez_cut = self.Ez_roi
        if zs_rho is None:
            zs_rho =  self.zs_bubble_radius_axial
        if bubble_radius is None:
            bubble_radius = self.bubble_radius_axial
        if bubble_radius_cut is None:
            bubble_radius_cut = self.bubble_radius_roi
        
        # Set up a figure with axes
        fig_wakeT_cut, axs_wakeT_cut = plt.subplots(nrows=1, ncols=2, layout='constrained', figsize=(10, 4))
        
        # Fill the axes with plots
        axs_wakeT_cut[0].fill_between(x=z_slices*1e6, y1=main_num_profile, y2=0, color='g', alpha=0.3)
        axs_wakeT_cut[0].plot(z_slices*1e6, main_num_profile, 'g', label='Number profile')
        axs_wakeT_cut[0].grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        axs_wakeT_cut[0].set_xlabel(zlab)
        axs_wakeT_cut[0].set_ylabel('Main beam number profile $N(z)$')
        ax_Ez_cut_wakeT2 = axs_wakeT_cut[0].twinx()
        ax_Ez_cut_wakeT2.plot(zs_Ez*1e6, Ez/1e9, label='Wake-T $E_z$')
        ax_Ez_cut_wakeT2.plot(z_slices*1e6, Ez_cut/1e9, 'r', label='Cut-out Wake-T $E_z$')
        ax_Ez_cut_wakeT2.set_ylabel('$E_z$ [GV/m]')
        ax_Ez_cut_wakeT2.legend(loc='lower right')
        
        axs_wakeT_cut[1].fill_between(x=z_slices*1e6, y1=main_num_profile, y2=0, color='g', alpha=0.3)
        axs_wakeT_cut[1].plot(z_slices*1e6, main_num_profile, 'g', label='Number profile')
        axs_wakeT_cut[1].grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        axs_wakeT_cut[1].set_xlabel(zlab)
        axs_wakeT_cut[1].set_ylabel('Main beam number profile $N(z)$')
        ax_rb_cut_wakeT2 = axs_wakeT_cut[1].twinx()
        ax_rb_cut_wakeT2.plot(zs_rho*1e6, bubble_radius*1e6, label='Wake-T $r_\mathrm{b}$')
        ax_rb_cut_wakeT2.plot(z_slices*1e6, bubble_radius_cut*1e6, 'r', label='Cut-out Wake-T $r_\mathrm{b}$')
        ax_rb_cut_wakeT2.set_ylabel('Bubble radius [$\mathrm{\mu}$m]')
        ax_rb_cut_wakeT2.legend(loc='upper right')


    # ==================================================
    # Plots the transverse oscillation vs propagation coordinate s for selected beam slices.
    def slice_offset_s_diag(self, beam):
        z_slices = self.zs_main_cut
        
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
        
        axs[1].plot(s_min1, X_min1, label='$x$')
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
        
        axs[2].plot(s_0, X_0, label='$x$')
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
        
        axs[3].plot(s_2, X_2, label='$x$')
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
    # Plots snapshots of the beam slices vs xi at various propagation distances s.
    def centroid_snapshot_plots(self, beam):

        x_slices_table = self.x_slices_table_main
        y_slices_table = self.y_slices_table_main
        s_slices_table = self.s_slices_table_main
        z_slices = self.zs_main_cut
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
    def instability_plots(self, s_ref_slice, x_ref_slice, xp_ref_slice, s_slices=None, x_slices=None, norm_amp_fac_s=None, mean_energy_s=None, energy_spread_s=None):

        #TODO: change out norm_amp_fac_s
        
        num_beam_slice = self.num_beam_slice
        beam_length_roi = self.beam_length_roi
        sigma_z = self.main_source.bunch_length
        
        if s_slices is None:
            s_slices = self.zs_main_cut
        if x_slices is None:
            x_slices = self.x_slices_main
        if norm_amp_fac_s is None:
            norm_amp_fac_s = self.norm_amp_fac_s
        if mean_energy_s is None:
            mean_energy_s = self.mean_energy_s
        if energy_spread_s is None:
            energy_spread_s = self.energy_spread_s
        
        #s_init = np.linspace(-beam_length_roi, 0, num_beam_slice)  # [m] slice propagation coordinates. Beam head initially at s=0. s[0] corresponds to beam tail. s[-1] corresponds to beam head.
        ref_slice_sigma_z = self.ref_slice_sigma_z
        ref_slice_index, _ = find_closest_value_in_arr(s_slices, np.mean(s_slices)+ref_slice_sigma_z*sigma_z)  # index n_sigma_z*sigma_z away from beam center towards tail. Lower index towards tail.
        middle_index, _ = find_closest_value_in_arr(s_slices, np.mean(s_slices))
        
        left = .093
        right = .61
        top = .68
        vertical_distance = .33

        slab = '$s$ [m]'
        xilab = r'$\xi$ [$\mathrm{\mu}$m]'
        xlab = r'$x$ [$\mathrm{\mu}$m]'
        energylab = '$\mathcal{E}$ [GeV]'
        
        z_beam = self.zs_main_cut*1e6  # [um]
        #prop_dist = self.prop_dist*1e6  # [um]
        xi_ref_slice = z_beam[ref_slice_index]  # [um]
        main_beam = self.main_beam
        dN_dz, xi_dNdz = main_beam.longitudinal_num_density()
        dN_dz = -dN_dz
        xi_dNdz = xi_dNdz*1e6 # [um]
        
        fig = plt.figure(figsize=(6.5, 9.1)) # The figsize parameter takes a tuple of two values that represent the width and height of the figure in inches

        ax1 = fig.add_axes([left,top,.335,.335])  # [left, bottom, width, height]
        ax1.set_box_aspect(1)
        ax1.plot(z_beam, x_slices*1e6)
        ax1.axvline(x=xi_ref_slice, color='r', linestyle='--', label='Reference slice')
        ax1.legend()
        ax1.set_xlabel(xilab)
        ax1.set_ylabel(xlab)
        ax1.set_title(f"Transverse offset at $s={round(s_ref_slice[-1],2)}$ m")
        ax1_twin = ax1.twinx()
        ax1_twin.fill_between(x=xi_dNdz, y1=dN_dz, y2=0, color='g', alpha=0.3)
        ax1_twin.plot(xi_dNdz, dN_dz, 'g', alpha=0.5)
        ax1_twin.yaxis.set_ticks([])
        
        
        ax2 = fig.add_axes([right,top,.335,.335])
        ax2.set_box_aspect(1)
        ax2.plot(s_ref_slice, norm_amp_fac_s)
        ax2.set_xlabel(slab)
        ax2.set_ylabel(r'$\Lambda/\Lambda_0$')
        ax2.set_title('Normalised amplification')
        
        ax3 = fig.add_axes([left,top-vertical_distance,.335,.335])
        ax3.set_box_aspect(1)
        ax3.plot(s_ref_slice, x_ref_slice*1e6)
        ax3.set_xlabel(slab)
        ax3.set_ylabel('$X$ [$\mathrm{\mu}$m]')
        ax3.set_title(r'Transverse offset, slice at $\xi=$' f"{round(xi_ref_slice,1)} "r'$\mathrm{\mu}$m')
        
        ax4 = fig.add_axes([right,top-vertical_distance,.335,.335])
        ax4.set_box_aspect(1)
        ax4.plot(s_ref_slice, xp_ref_slice*1e3)
        ax4.set_xlabel(slab)
        ax4.set_ylabel('$X\'$ [mrad]')
        ax4.set_title(r'Angular offset, slice at $\xi=$' f"{round(xi_ref_slice,1)} "r'$\mathrm{\mu}$m')
        
        ax5 = fig.add_axes([left,top-vertical_distance*2,.335,.335])
        ax5.set_box_aspect(1)
        ax5.plot(s_ref_slice, mean_energy_s/1e9)
        ax5.set_xlabel(slab)
        ax5.set_ylabel(energylab)
        ax5.set_title('Mean energy')
        
        ax6 = fig.add_axes([right,top-vertical_distance*2,.335,.335])
        ax6.set_box_aspect(1)
        ax6.plot(s_ref_slice, energy_spread_s*100)
        ax6.set_xlabel(slab)
        ax6.set_ylabel('$\sigma_\mathcal{E}/\mathcal{E}$ [%]')
        ax6.set_title('Energy spread');

    
    # ==================================================
    def print_initial_summary(self, drive_beam, main_beam):
        print('======================================================================')
        print(f"Plasma density [m^-3]:\t\t\t\t\t {self.plasma_density :.3e}")
        print(f"Mean initial gamma:\t\t\t\t\t {np.mean(main_beam.gamma()) :.3f}")
        print(f"Mean initial energy [GeV]:\t\t\t\t {np.mean(main_beam.Es())/1e9 :.3f}")
        print(f"Initial rms energy spread [%]:\t\t\t\t {main_beam.rel_energy_spread()*1e2 :.3f}\n")

        print(f"rms beam length [um]:\t\t\t\t\t {main_beam.bunch_length()*1e6 :.3f}")
        print(f"Beam region of interest [sigma_z]:\t\t\t {self.main_beam_roi :.3f}")
        print(f"Beam slice thickness [um]:\t\t\t\t {self.beam_length_roi/self.num_beam_slice*1e6 :.3f}")
        print(f"Number of beam slices:\t\t\t\t\t {len(self.zs_main_cut) :d}")
        print(f"Number of beam particles:\t\t\t\t {len(main_beam.xs()) :d}\n")
        
        print(f"Initial drive beam x offset [um]:\t\t\t {drive_beam.x_offset()*1e6 :.3f}")
        print(f"Initial drive beam y offset [um]:\t\t\t {drive_beam.y_offset()*1e6 :.3f}")
        print(f"Initial drive beam z offset [um]:\t\t\t {drive_beam.z_offset()*1e6 :.3f}\n")
        
        print(f"Initial main beam x offset [um]:\t\t\t {main_beam.x_offset()*1e6 :.3f}")
        print(f"Initial main beam y offset [um]:\t\t\t {main_beam.y_offset()*1e6 :.3f}")
        print(f"Initial main beam z offset [um]:\t\t\t {main_beam.z_offset()*1e6 :.3f}\n")
        
        print(f"Initial normalised x emittance [mm mrad]:\t\t {main_beam.norm_emittance_x()*1e6 :.3f}")
        print(f"Initial normalised y emittance [mm mrad]:\t\t {main_beam.norm_emittance_y()*1e6 :.3f}\n")
        
        print(f"Initial matched beta function [mm]:\t\t\t {beta_matched(self.plasma_density, np.mean(main_beam.Es()))*1e3 :.3f}")
        print(f"Initial x beta function [mm]:\t\t\t\t {main_beam.beta_x()*1e3 :.3f}")
        print(f"Initial y beta function [mm]:\t\t\t\t {main_beam.beta_y()*1e3 :.3f}\n")
        
        print(f"Initial x beam size [um]:\t\t\t\t {main_beam.beam_size_x()*1e6 :.3f}")
        print(f"Initial y beam size [um]:\t\t\t\t {main_beam.beam_size_y()*1e6 :.3f}")
        print('----------------------------------------------------------------------')

    
    # ==================================================
    def print_current_summary(self, drive_beam, initial_main_beam, beam_out):
        print('======================================================================')
        print(f"Stage length [m]:\t\t\t\t\t {self.length :.3f}")
        print(f"Propagation length [m]:\t\t\t\t\t {beam_out.location :.3f}")
        print(f"Plasma density [m^-3]:\t\t\t\t\t {self.plasma_density :.3e}")
        print(f"Current mean gamma:\t\t\t\t\t {np.mean(beam_out.gamma()) :.3f}")
        print(f"Current mean energy  [GeV]:\t\t\t\t {np.mean(beam_out.Es())/1e9 :.3f}")
        print(f"Initial rms energy spread [%]:\t\t\t\t {initial_main_beam.rel_energy_spread()*1e2 :.3f}")
        print(f"Current rms energy spread [%]:\t\t\t\t {beam_out.rel_energy_spread()*1e2 :.3f}\n")

        print(f"Current rms main beam length [um]:\t\t\t {beam_out.bunch_length()*1e6 :.3f}")
        print(f"Initial rms main beam length [um]:\t\t\t {initial_main_beam.bunch_length()*1e6 :.3f}")
        print(f"Beam region of interest [sigma_z]:\t\t\t {self.main_beam_roi :.3f}")
        print(f"Beam slice thickness [um]:\t\t\t\t {self.beam_length_roi/self.num_beam_slice*1e6 :.3f}")
        print(f"Number of beam slices:\t\t\t\t\t {len(self.zs_main_cut) :d}")
        print(f"Initial number of beam particles:\t\t\t {len(initial_main_beam.xs()) :d}")
        print(f"Current number of beam particles:\t\t\t {len(beam_out.xs()) :d}\n")

        print(f"Initial drive beam x offset [um]:\t\t\t {drive_beam.x_offset()*1e6 :.3f}")
        print(f"Initial drive beam y offset [um]:\t\t\t {drive_beam.y_offset()*1e6 :.3f}")
        print(f"Initial drive beam z offset [um]:\t\t\t {drive_beam.z_offset()*1e6 :.3f}\n")
        
        print(f"Initial main beam x offset [um]:\t\t\t {initial_main_beam.x_offset()*1e6 :.3f}")
        print(f"Initial main beam y offset [um]:\t\t\t {initial_main_beam.y_offset()*1e6 :.3f}")
        print(f"Initial main beam z offset [um]:\t\t\t {initial_main_beam.z_offset()*1e6 :.3f}\n")
        
        print(f"Initial normalised x emittance [mm mrad]:\t\t {initial_main_beam.norm_emittance_x()*1e6 :.3f}")
        print(f"Current normalised x emittance [mm mrad]:\t\t {beam_out.norm_emittance_x()*1e6 :.3f}")
        print(f"Initial normalised y emittance [mm mrad]:\t\t {initial_main_beam.norm_emittance_y()*1e6 :.3f}")
        print(f"Current normalised y emittance [mm mrad]:\t\t {beam_out.norm_emittance_y()*1e6 :.3f}\n")
        
        print(f"Initial x beta function [mm]:\t\t\t\t {initial_main_beam.beta_x()*1e3 :.3f}")
        print(f"Current x beta function [mm]:\t\t\t\t {beam_out.beta_x()*1e3 :.3f}")
        print(f"Initial y beta function [mm]:\t\t\t\t {initial_main_beam.beta_y()*1e3 :.3f}")
        print(f"Current y beta function [mm]:\t\t\t\t {beam_out.beta_y()*1e3 :.3f}\n")
        
        print(f"Initial x beam size [um]:\t\t\t\t {initial_main_beam.beam_size_x()*1e6 :.3f}")
        print(f"Current x beam size [um]:\t\t\t\t {beam_out.beam_size_x()*1e6 :.3f}")
        print(f"Initial y beam size [um]:\t\t\t\t {initial_main_beam.beam_size_y()*1e6 :.3f}")
        print(f"Current y beam size [um]:\t\t\t\t {beam_out.beam_size_y()*1e6 :.3f}\n")

        initial_norm_amp_x = self.calc_norm_amp(particle_offsets=initial_main_beam.xs(), particle_angles=initial_main_beam.xps())
        current_norm_amp_x = self.calc_norm_amp(particle_offsets=beam_out.xs(), particle_angles=beam_out.xps())
        initial_norm_amp_y = self.calc_norm_amp(particle_offsets=initial_main_beam.ys(), particle_angles=initial_main_beam.yps())
        current_norm_amp_y = self.calc_norm_amp(particle_offsets=beam_out.ys(), particle_angles=beam_out.yps())
        
        print(f"Initial normalised amplitude in x:\t\t\t {initial_norm_amp_x :.3f}")
        print(f"Normalised amplitude in x after current stage:\t\t {current_norm_amp_x :.3f}")
        print(f"Normalised amplitude factor in x after current stage:\t {current_norm_amp_x/initial_norm_amp_x :.3f}")
        print(f"Initial normalised amplitude in y:\t\t\t {initial_norm_amp_y :.3f}")
        print(f"Normalised amplitude in y after current stage:\t\t {current_norm_amp_y :.3f}")
        print(f"Normalised amplitude factor in y after current stage:\t {current_norm_amp_y/initial_norm_amp_y :.3f}")
        #print(f"Number of beam particles:\t\t {N}")
        print('----------------------------------------------------------------------')


