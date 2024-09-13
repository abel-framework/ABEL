"""
Ion wakefield perturbation caused by ion motion


Ben Chen, 12 June 2024, University of Oslo
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # For customising colour maps
import scipy.constants as SI
from scipy.interpolate import RegularGridInterpolator
import copy
from joblib import Parallel, delayed
import os
import time, warnings

from abel.classes.beam import Beam
from abel.utilities.plasma_physics import k_p
from abel.utilities.statistics import weighted_std
from abel.utilities.other import find_closest_value_in_arr
from abel.apis.rf_track.rf_track_api import calc_sc_fields_obj



class IonMotionConfig():
    
    # ==================================================
    def __init__(self, drive_beam, main_beam, plasma_ion_density, ion_charge_num=1.0, ion_mass=None, num_z_cells_main=None, num_x_cells_rft=50, num_y_cells_rft=50, num_xy_cells_probe=41, uniform_z_grid=False, driver_x_jitter=0.0, driver_y_jitter=0.0, update_factor=1.0, update_ion_wakefield=False):
        """
        Contains calculation configuration for calculating the ion wakefield perturbation.
        
        Parameters
        ----------
        drive_beam: Beam object of drive beam.

        main_beam: Beam object of main beam.
        
        plasma_ion_density: [m^-3] float
            Plasma ion density.
        
        ion_charge_num: [e] float
            Plasma io charge in units of the elementary charge.
        
        num_z_cells_main: float
            Number of grid cells along z.

        num_x(y)_cells_rft: float
            Number of grid cells along x and y used in RF-Track for calculating beam electric fields.
            
        num_xy_cells_probe: float
            Number of grid cells along x and y used to probe beam electric fields calculated by RF-Track.
        
        uniform_z_grid: bool
            Determines whether the grid along z is uniform (True) or finely resolved along the drive beam and main beam regions, while the region between the beams are coarsely resolved (False).

        update_factor: float
            Update ion wakefield perturbation when beam sizes have changed by this factor.

        update_ion_wakefield: bool
            ...

        xs_probe: [m] 1D float array
            x-coordinates used to probe beam electric fields calculated by RF-Track.

        ys_probe: [m] 1D float array
            y-coordinates used to probe beam electric fields calculated by RF-Track.

        zs_probe: [m] 1D float array
            z-coordinates used to probe beam electric fields calculated by RF-Track.

        grid_size_z: [m] float or 1D float array
            The size of grid cells along z. If uniform_z_grid is False, this becomes a 1D float array containing the distance between elements in zs_probe.
        """

        #self.drive_beam = drive_beam
        self.plasma_ion_density = plasma_ion_density  # [m^-3]
        self.ion_charge_num = ion_charge_num
        
        # Set the ion mass to default if necessary
        if ion_mass is None:
            ion_mass = 4.002602 * SI.physical_constants['atomic mass constant'][0]  # [kg], He mass.
        self.ion_mass = ion_mass

        # Set the number of cells
        if num_z_cells_main is None:
            num_z_cells_main = round(np.sqrt( len(drive_beam)+len(main_beam) )/2)
        self.num_z_cells_main = num_z_cells_main
        self.num_x_cells_rft = num_x_cells_rft
        self.num_y_cells_rft = num_y_cells_rft
        self.num_xy_cells_probe = num_xy_cells_probe
        self.uniform_z_grid = uniform_z_grid

        self.xs_probe = None
        self.ys_probe = None
        self.zs_probe = None
        self.zs_probe_main = None
        self.zs_probe_separation = None
        self.zs_probe_driver = None
        self.xlims_driver_sc = None
        self.ylims_driver_sc = None
        self.grid_size_z = None

        # Store the std of driver jitters for transverse mesh refinement
        self.driver_x_jitter = driver_x_jitter
        self.driver_y_jitter = driver_y_jitter

        self.update_factor = update_factor
        self.update_ion_wakefield = update_ion_wakefield
        
        self.Wx_perts = None
        self.Wy_perts = None
        
        # Set the coordinates used to probe beam electric fields from RF-Track
        self.set_probing_coordinates(drive_beam, main_beam, set_driver_sc_coords=True)
        driver_sc_fields_obj = self.assemble_driver_sc_fields_obj(drive_beam)
        self.driver_sc_fields_obj = driver_sc_fields_obj

    
    # ==================================================
    def set_probing_coordinates(self, drive_beam, main_beam, set_driver_sc_coords=False):
        """
        Sets the coordinates used to probe the beam fields of the drive beam and main beam. Has to be called at every ion wakefield calculation step.
        """
        
        if drive_beam.zs().min() < main_beam.zs().max():
            raise ValueError('Beams have to propagate towards increasing z with drive beam placed in front of main beam.')

        if set_driver_sc_coords:
            # Set the xy limits used by assemble_driver_sc_fields_obj()
            x_max = np.max([drive_beam.xs().max(), main_beam.xs().max()])
            x_min = np.min([drive_beam.xs().min(), main_beam.xs().min()])
            y_max = np.max([drive_beam.ys().max(), main_beam.ys().max()])
            y_min = np.min([drive_beam.ys().min(), main_beam.ys().min()])
            xy_padding = 2.0
            x_max = self.pad_upwards(x_max, padding=xy_padding)
            x_min = self.pad_downwards(x_min, padding=xy_padding)
            y_max = self.pad_upwards(y_max, padding=xy_padding)
            y_min = self.pad_downwards(y_min, padding=xy_padding)
            self.xlims_driver_sc = np.array([x_min, x_max])  # [m]
            self.ylims_driver_sc = np.array([y_min, y_max])  # [m]
        
        # Transverse coordinates for probing the fields from RF-Track get_field()
        xs_probe = np.linspace( main_beam.xs().min(), main_beam.xs().max(), self.num_xy_cells_probe)  # [m]
        ys_probe = np.linspace( main_beam.ys().min(), main_beam.ys().max(), self.num_xy_cells_probe)  # [m]

        # Modify xs_probe and ys_probe if mesh refinement is required
        if self.driver_x_jitter != 0.0 or self.driver_y_jitter != 0.0:

            # Take care of when only one of the jitters is non-zero.
            driver_x_jitter = self.driver_x_jitter
            if driver_x_jitter == 0.0:
                driver_x_jitter = 0.1e-6
            driver_y_jitter = self.driver_y_jitter
            if driver_y_jitter == 0.0:
                driver_y_jitter = 0.1e-6
                
            dx_driver_middle = driver_x_jitter/5
            num_x_fine_middle_mesh = 7
            driver_x_middle_width = num_x_fine_middle_mesh * dx_driver_middle

            xs_driver_middle = np.linspace(drive_beam.x_offset()-driver_x_middle_width/2, drive_beam.x_offset()+driver_x_middle_width/2, num_x_fine_middle_mesh)
            indices = np.searchsorted(xs_probe, xs_driver_middle)
            xs_probe = np.insert(xs_probe, indices, xs_driver_middle)

            dy_driver_middle = driver_y_jitter/5
            num_y_fine_middle_mesh = 7
            driver_y_middle_width = num_y_fine_middle_mesh * dy_driver_middle
            
            ys_driver_middle = np.linspace(drive_beam.y_offset()-driver_y_middle_width/2, drive_beam.y_offset()+driver_y_middle_width/2, num_y_fine_middle_mesh)
            indices = np.searchsorted(ys_probe, ys_driver_middle)
            ys_probe = np.insert(ys_probe, indices, ys_driver_middle)

            if np.any(np.diff(xs_probe)) < 0 or np.any(np.diff(ys_probe)) < 0:
                raise ValueError('xs_probe and/or ys_probe are not in ascending order.')
            #if self.num_xy_cells_probe_updated is False:
            #    self.num_xy_cells_probe = len(xs_probe)
            #    self.num_xy_cells_probe_updated = True

        self.xs_probe = xs_probe
        self.ys_probe = ys_probe

        if self.xs_probe.min() < self.xlims_driver_sc.min() or self.xs_probe.max() > self.xlims_driver_sc.max() or self.ys_probe.min() < self.ylims_driver_sc.min() or self.ys_probe.max() > self.ylims_driver_sc.max():
            print('xs_probe.min:', self.xs_probe.min(), 'xlims_driver_sc.min:', self.xlims_driver_sc.min(), 'xs_probe.max:', self.self.xs_probe.max(), 'ys_probe.min:', self.ys_probe.min(), 'ylims_driver_sc.min:', self.self.ylims_driver_sc.min(), 'ys_probe.max:', self.ys_probe.max())
            warnings.warn("The range of the probing coordinates is larger than the driver probing coordinates. This may lead to a slower performance due to extrapolations when extracting the driver beam fields.", UserWarning)

        # Set the z-coordinates used to probe beam electric fields from RF-Track
        self.zs_probe_main = np.linspace( main_beam.zs().max(), main_beam.zs().min(), self.num_z_cells_main )  # [m], beam head facing start of array.
        
        if self.uniform_z_grid:
            self.grid_size_z = np.abs(np.diff(self.zs_probe_main)[0])  # [m], the size of grid cells along z
            driver_num_z = int( (drive_beam.zs().max() - self.pad_upwards(main_beam.zs().max()))/self.grid_size_z )
            self.zs_probe_driver = np.linspace( drive_beam.zs().max(), drive_beam.zs().min(), driver_num_z )  # [m], beam head facing start of array.

            sep_num_z = int( (self.pad_downwards(self.zs_probe_driver.min()) - self.pad_upwards(self.zs_probe_main.max()) )/self.grid_size_z )
            self.zs_probe_separation = np.linspace( self.pad_downwards(self.zs_probe_driver.min()), self.pad_upwards(self.zs_probe_main.max()), sep_num_z )  # [m], beam head facing start of array. Space between driver and main beam.
            
            self.zs_probe = np.concatenate( [self.zs_probe_driver, self.zs_probe_separation, self.zs_probe_main] )  # [m], beam head facing start of array.
            
        else:
            dz = weighted_std(drive_beam.zs(), drive_beam.weightings())/5
            driver_num_z = int( (drive_beam.zs().max() - drive_beam.zs().min())/dz )
            self.zs_probe_driver = np.linspace( drive_beam.zs().max(), drive_beam.zs().min(), driver_num_z )  # [m], beam head facing start of array.

            sep_num_z = int( (self.pad_downwards(self.zs_probe_driver.min()) - self.pad_upwards(self.zs_probe_main.max()) )/dz )
            self.zs_probe_separation = np.linspace( self.pad_downwards(self.zs_probe_driver.min()), self.pad_upwards(self.zs_probe_main.max()), sep_num_z )  # [m], beam head facing start of array. Space between driver and main beam.
            self.zs_probe = np.concatenate( [self.zs_probe_driver, self.zs_probe_separation, self.zs_probe_main] )  # [m], beam head facing start of array.
            self.grid_size_z = np.abs( np.insert(np.diff(self.zs_probe), 0, 0.0) )

        if np.any(np.diff(self.zs_probe) > 0):
            raise ValueError('zs_probe is not in descending order.')

    
    # ==================================================
    def assemble_driver_sc_fields_obj(self, drive_beam):
        """
        Creates a RF-Track SpaceCharge_Field object for the drive beam. Only needs to be called at the start of a stage for non-evolving drive beam.
        """
        
        x_min, x_max = self.xlims_driver_sc
        y_min, y_max = self.ylims_driver_sc
        
        z_end = self.pad_downwards(self.zs_probe_driver.min())
        z_start = self.pad_upwards(self.zs_probe_driver.max())
        
        X, Y, Z = np.meshgrid([x_min, x_max], [y_min, y_max], [z_start, z_end], indexing='ij')

        # Set up a beam with 0 charge consisting of 8 particles to enlarge region of the RF-Track SpaceCharge_Field object to avoid extrapolation later
        empty_beam = Beam()
        empty_beam.set_phase_space(Q=0,
                                   xs=X.flatten(),
                                   ys=Y.flatten(),
                                   zs=Z.flatten(), 
                                   uxs=np.ones_like(X.flatten())*drive_beam.uxs()[0],
                                   uys=np.ones_like(X.flatten())*drive_beam.uys()[0],
                                   uzs=np.ones_like(X.flatten())*drive_beam.uzs()[0],
                                   weightings=np.ones_like(X.flatten())*drive_beam.weightings()[0],
                                   particle_mass=drive_beam.particle_mass)
        
        combined_beam = drive_beam + empty_beam
        combined_beam.particle_mass = drive_beam.particle_mass

        # Set the resolution for the RF-Track SpaceCharge_Field object 
        dx = drive_beam.beam_size_x()/6
        num_x = int( (combined_beam.xs().max() - combined_beam.xs().min())/dx )
        dy = drive_beam.beam_size_y()/6
        num_y = int( (combined_beam.ys().max() - combined_beam.ys().min())/dy )
        dz = drive_beam.bunch_length()/10
        num_z = int( (combined_beam.zs().max() - combined_beam.zs().min())/dz )
        
        return calc_sc_fields_obj(combined_beam, num_x, num_y, num_z, num_t_bins=1)

    
    # ==================================================
    def pad_downwards(self, arr_min, padding=0.05):
        if padding < 0.0:
            padding = np.abs(padding)
        return arr_min*(1.0 - np.sign(arr_min)*padding)


    # ==================================================
    def pad_upwards(self, arr_max, padding=0.05):
        if padding < 0.0:
            padding = np.abs(padding)
        return arr_max*(1.0 + np.sign(arr_max)*padding)



###################################################
# Benedetti's ion motion quantifier Gamma
def ion_motion_quantifier(bunch_length, beam_peak_dens, plasma_density, ion_charge_num=1.0, ion_mass=None):
    if ion_mass is None:
        ion_mass = 39.95*SI.physical_constants['atomic mass constant'][0]  # [kg], Ar mass.
    return ion_charge_num*SI.m_e/ion_mass * beam_peak_dens/plasma_density * (k_p(plasma_density) * bunch_length)**2



###################################################
# Mehrling's ion motion quantifier Lambda
def ion_motion_quantifier2(beam_size_x, beam_size_y, bunch_length, beam_peak_current, ion_charge_num=1.0, ion_mass=None):
    if ion_mass is None:
        ion_mass = 39.95*SI.physical_constants['atomic mass constant'][0]  # [kg], Ar mass.
        
    I_A = 4*np.pi *SI.epsilon_0 *SI.m_e* SI.c**3 / SI.e  # [A], Alfvén current.
    
    return ion_charge_num*SI.m_e/ion_mass * beam_peak_current/I_A * bunch_length**2 / beam_size_x / beam_size_y



###################################################
def probe_driver_beam_field(ion_motion_config, driver_sc_fields_obj):
    """
    Probes the drive beam SpaceCharge_Field object and returns the drive beam E-fields stored in 3D numpy arrays where the first, second and third dimensions correspond to positions along x, y and z. Needs to be called at every ion wakefield calculation step.

    Parameters
    ----------
    ion_motion_config: IonMotionConfig object
        Contains the configurations for calculating the ion wakefield perturbation.
    
    driver_sc_fields_obj: RF-Track SpaceCharge_Field object.
        Contains the beam electric and magnetic fields for the drive beam calculated by RF-Track.
        
    Returns
    ----------
    driver_Exs_3d, driver_Eys_3d: [V/m] 3D float array
        Contain the beam E-field components where the first, second and third dimensions correspond to positions along x, y and z.
    """
    
    # Set up a 3D mesh grid
    xs_probe = ion_motion_config.xs_probe  # [m]
    ys_probe = ion_motion_config.ys_probe  # [m]
    zs_probe = ion_motion_config.zs_probe_driver  # [m]
    X, Y, Z = np.meshgrid(xs_probe, ys_probe, zs_probe, indexing='ij')
    xs_grid_flat = X.flatten()*1e3  # [mm]
    ys_grid_flat = Y.flatten()*1e3  # [mm]
    zs_grid_flat = Z.flatten()*1e3  # [mm]
        
    # Probe beam E-field
    E_fields_beam, _ = driver_sc_fields_obj.get_field(xs_grid_flat, ys_grid_flat, zs_grid_flat, np.zeros(len(zs_grid_flat)))  # [V/m]

    driver_Exs = E_fields_beam[:,0]
    driver_Eys = E_fields_beam[:,1]
    
    # Reshape the field component into a 3D array
    driver_Exs_3d = driver_Exs.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
    driver_Eys_3d = driver_Eys.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
    
    return driver_Exs_3d, driver_Eys_3d



###################################################
def construct_empty_field(ion_motion_config):
    """
    Returns a zero 3D numpy where the first, second and third dimensions correspond to positions along x, y and z. Needs to be called at every ion wakefield calculation step.
    """

    # Get the dimensions
    xs_probe = ion_motion_config.xs_probe  # [m]
    ys_probe = ion_motion_config.ys_probe  # [m]
    zs_probe = ion_motion_config.zs_probe_separation  # [m]
    num_x = len(xs_probe)
    num_y = len(ys_probe)
    num_z = len(zs_probe)

    return np.zeros((num_x, num_y, num_z))



###################################################
def assemble_main_sc_fields_obj(ion_motion_config, main_beam):
    """
    Creates a SpaceCharge_Field object for the main beam. Needs to be called at every ion wakefield calculation step.
    """
    
    # Slightly enlarge the transverse region by constructing empty_beam to avoid extrapolation when evaluating the beam fields.
    x_min = ion_motion_config.pad_downwards(ion_motion_config.xs_probe.min(), padding=0.05)
    x_max = ion_motion_config.pad_upwards(ion_motion_config.xs_probe.max(), padding=0.05)
    y_min = ion_motion_config.pad_downwards(ion_motion_config.ys_probe.min(), padding=0.05)
    y_max = ion_motion_config.pad_upwards(ion_motion_config.ys_probe.max(), padding=0.05)
    z_end = main_beam.zs().min()
    z_start = main_beam.zs().max()
    
    X, Y, Z = np.meshgrid([x_min, x_max], [y_min, y_max], [z_start, z_end], indexing='ij')
    
    empty_beam = Beam()
    empty_beam.set_phase_space(Q=0,
                               xs=X.flatten(),
                               ys=Y.flatten(),
                               zs=Z.flatten(), 
                               uxs=np.ones_like(X.flatten())*main_beam.uxs()[0],
                               uys=np.ones_like(X.flatten())*main_beam.uys()[0],
                               uzs=np.ones_like(X.flatten())*main_beam.uzs()[0],
                               weightings=np.ones_like(X.flatten())*main_beam.weightings()[0],
                               particle_mass=main_beam.particle_mass)
    
    combined_beam = main_beam + empty_beam
    combined_beam.particle_mass = main_beam.particle_mass
    
    return calc_sc_fields_obj(combined_beam, ion_motion_config.num_x_cells_rft, ion_motion_config.num_y_cells_rft, ion_motion_config.num_z_cells_main, num_t_bins=1)



###################################################
def probe_main_beam_field(ion_motion_config, main_sc_fields_obj):
    """
    Probes the main beam SpaceCharge_Field object and returns the main beam E-field component stored in a 3D numpy array. Needs to be called at every ion wakefield calculation step.

    Parameters
    ----------
    ion_motion_config: IonMotionConfig object
        Contains the configurations for calculating the ion wakefield perturbation.
    
    main_sc_fields_obj: RF-Track SpaceCharge_Field object.
        Contains the beam electric and magnetic fields for the main beam calculated by RF-Track.
        
    Returns
    ----------
    Exs_3d, Eys_3d: [V/m] 3D float array
        Contain the beam E-field components where the first, second and third dimensions correspond to positions along x, y and z.
    """
    
    # Set up a 3D mesh grid
    xs_probe = ion_motion_config.xs_probe  # [m]
    ys_probe = ion_motion_config.ys_probe  # [m]
    zs_probe = ion_motion_config.zs_probe_main  # [m]
    X, Y, Z = np.meshgrid(xs_probe, ys_probe, zs_probe, indexing='ij')
    xs_grid_flat = X.flatten()*1e3  # [mm]
    ys_grid_flat = Y.flatten()*1e3  # [mm]
    zs_grid_flat = Z.flatten()*1e3  # [mm]
        
    # Probe beam E-field
    E_fields_beam, _ = main_sc_fields_obj.get_field(xs_grid_flat, ys_grid_flat, zs_grid_flat, np.zeros(len(zs_grid_flat)))  # [V/m]
    Exs = E_fields_beam[:,0]
    Eys = E_fields_beam[:,1]
    
    # Reshape the field component into a 3D array       
    Exs_3d = Exs.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
    Eys_3d = Eys.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
    #Eys_3d = Exs.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
    
    return Exs_3d, Eys_3d



###################################################
def ion_wakefield_perturbation(ion_motion_config, main_Exs_3d, main_Eys_3d, driver_Exs_3d, driver_Eys_3d):
    """
    Calculates the ion wakefield perturbation to the otherwise linear background focusing kp*E0*r/2 by integrating the beam electric fields along z.

    Parameters
    ----------
    ion_motion_config: IonMotionConfig object
        Contains the configurations for calculating the ion wakefield perturbation.

    main_Exs_3d, main_Eys_3d: 3D float array
        Contains the main beam E-field component where the first, second and third dimensions correspond to positions along x, y and z.
    
    driver_Exs_3d, driver_Eys_3d: 3D float array
        Contains the drive beam E-field component where the first, second and third dimensions correspond to positions along x, y and z.
        
    Returns
    ----------
    wakefield_perturbations: [V/m] 3D float array
        Contains the ion wakefield perturbation where the first, second and third dimensions correspond to positions along x, y and z. 
    """

    # Get parameters
    kp = k_p(ion_motion_config.plasma_ion_density)  # [m]
    ion_mass = ion_motion_config.ion_mass  # [kg]
    ion_charge_num = ion_motion_config.ion_charge_num
    zs_probe = ion_motion_config.zs_probe  # [m]
    grid_size_z = ion_motion_config.grid_size_z  # [m], the grid size along z

    # Get the empty region between the drive beam and main beam
    sep_E_fields_3d = construct_empty_field(ion_motion_config)

    # Concatenate the 3D beam field of the drive beam and main beam
    Exs_3d = np.concatenate([driver_Exs_3d, sep_E_fields_3d, main_Exs_3d], axis=2)  # [V/m], beam head facing start of array.
    Eys_3d = np.concatenate([driver_Eys_3d, sep_E_fields_3d, main_Eys_3d], axis=2)  # [V/m], beam head facing start of array.
    
    # Integration along z using by splitting up the convolution integral
    integral_x = np.cumsum( zs_probe * Exs_3d * grid_size_z, axis=2) - zs_probe * np.cumsum(Exs_3d * grid_size_z, axis=2)
    integral_y = np.cumsum( zs_probe * Eys_3d * grid_size_z, axis=2) - zs_probe * np.cumsum(Eys_3d * grid_size_z, axis=2)
    
    wakefield_x_perturbations = ion_charge_num * SI.m_e/ion_mass * kp**2 * integral_x
    wakefield_y_perturbations = ion_charge_num * SI.m_e/ion_mass * kp**2 * integral_y
    
    return wakefield_x_perturbations, wakefield_y_perturbations



###################################################
def intplt_ion_wakefield_perturbation(beam, wakefield_perturbations, ion_motion_config, intplt_beam_region_only=True):
    """
    Interpolate the ion wake field perturbation to beam macroparticle positions.
    """

    if intplt_beam_region_only:  # Perform the interpolation only in the beam's region
        zs_probe = ion_motion_config.zs_probe
        dz = ion_motion_config.grid_size_z
        
        if isinstance(dz, np.ndarray):
            dz = np.min(dz)
            
        head_idx, _ = find_closest_value_in_arr( zs_probe, beam.zs().max()+dz )
        tail_idx, zs_probe_val = find_closest_value_in_arr( zs_probe, beam.zs().min()-dz )
        
        wakefield_perturbations = wakefield_perturbations[:, :, head_idx:tail_idx+1]
        zs_probe = zs_probe[head_idx:tail_idx+1]
    
        if beam.zs().min() < zs_probe.min() or beam.zs().max() > zs_probe.max():
            print('Probe z min max:', zs_probe.min()*1e6, zs_probe.max()*1e6)
            print('beam z min max:', beam.zs().min()*1e6, beam.zs().max()*1e6)
            raise ValueError('z is out of bounds.')
    else:
        zs_probe = ion_motion_config.zs_probe
    
    # Create the interpolator
    interpolator = RegularGridInterpolator((ion_motion_config.xs_probe, ion_motion_config.ys_probe, zs_probe), wakefield_perturbations, method='linear', bounds_error=True, fill_value=np.nan)

    # Coordinates for interpolating the field values
    xs_intplt = beam.xs()  # [m]
    ys_intplt = beam.ys()  # [m]
    zs_intplt = beam.zs()  # [m]
    
    # Combine the coordinates into an array of points
    points = np.vstack((xs_intplt, ys_intplt, zs_intplt)).T
    
    # Interpolate the field values at the given coordinates
    interpolated_wakefield_perturbations = interpolator(points)  # 1D array

    return interpolated_wakefield_perturbations, interpolator



###################################################
def ion_wakefield_scatter(arr1, arr2, intpl_W_perts, label1=None, label2=None, clabel=None, n_th_particle=1):
    
    arr1 = arr1[::n_th_particle]
    arr2 = arr2[::n_th_particle]
    intpl_W_perts = intpl_W_perts[::n_th_particle]
    
    plt.figure()
    plt.scatter(arr1*1e6, arr2*1e6, c=intpl_W_perts/1e9, cmap='seismic', s=5.0)
    plt.xlabel(label1)
    plt.ylabel(label2)
    cbar = plt.colorbar()
    cbar.set_label(clabel)
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)

    
    
###################################################
def ion_wakefield_xy_scatter(beam, plasma_density, intpl_Wx_perts, intpl_Wy_perts, n_th_particle=1):

    # Define the color map and boundaries
    colors = ['yellow', 'orange', 'red', 'black']
    bounds = [0, 0.2, 0.4, 0.8, 1]
    cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=256)

    zs = beam.zs()
    zs = zs[::n_th_particle]
    xs = beam.xs()
    xs = xs[::n_th_particle]
    ys = beam.ys()
    ys = ys[::n_th_particle]
    intpl_Wx_perts = intpl_Wx_perts[::n_th_particle]
    intpl_Wy_perts = intpl_Wy_perts[::n_th_particle]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6*2, 5))
    
    plt.tight_layout(pad=6.0)  # Sets padding between the figure edge and the edges of subplots, as a fraction of the font size.
    fig.subplots_adjust(top=0.87)

    #sort_x_indices = np.argsort(xs)
    p = axes[0].scatter(xs*1e6, plasma_density*SI.e/(2*SI.epsilon_0)*xs/1e9 - intpl_Wx_perts/1e9, c=zs*1e6, cmap=cmap, s=5.0)
    axes[0].plot(xs*1e6, plasma_density*SI.e/(2*SI.epsilon_0)*xs/1e9)
    axes[0].set_xlabel('x [µm]')
    axes[0].set_ylabel('$\mathcal{W}_x$ [GV/m]')
    axes[0].grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
    twinax = axes[0].twinx()
    twinax.fill_between(x=beam.transverse_profile_x()[1]*1e6, y1=beam.charge_sign()*beam.transverse_profile_x()[0], y2=0, color='g', alpha=0.3)
    twinax.set_yticks([])
    
    #sort_y_indices = np.argsort(ys)
    axes[1].scatter(ys*1e6, plasma_density*SI.e/(2*SI.epsilon_0)*ys/1e9 - intpl_Wy_perts/1e9, c=zs*1e6, cmap=cmap, s=3.0)
    axes[1].plot(ys*1e6, plasma_density*SI.e/(2*SI.epsilon_0)*ys/1e9)
    axes[1].set_xlabel('y [µm]')
    axes[1].set_ylabel('$\mathcal{W}_y$ [GV/m]')
    axes[1].grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
    twinax = axes[1].twinx()
    twinax.fill_between(x=beam.transverse_profile_y()[1]*1e6, y1=beam.charge_sign()*beam.transverse_profile_y()[0], y2=0, color='g', alpha=0.3)
    twinax.set_yticks([])

    # Set label and other properties for the colorbar
    cbar_ax = fig.add_axes([0.15, 0.96, 0.7, 0.02])   # The four values in the list correspond to the left, bottom, width, and height of the new axes, respectively.
    fig.colorbar(p, cax=cbar_ax, orientation='horizontal', label=r'$z$ [µm]')



#vvvvvvvvvvvvvvvvvvvvvv Not currently in use vvvvvvvvvvvvvvvvvvvvvv
'''
    # ==================================================
    def wide_probe_driver_field(self, sc_fields_obj):
        """
        Probes the beam SpaceCharge_Field object over a wider region around the beam and returns the drive beam E-field components stored in a 3D numpy array where the first, second and third dimensions correspond to positions along x, y and z.
        """

        x_min, x_max = self.xlims_driver_sc
        y_min, y_max = self.ylims_driver_sc
        z_end = self.pad_downwards(self.zs_probe_driver.min())
        z_start = self.pad_upwards(self.zs_probe_driver.max())
        drive_beam = self.drive_beam
    
        # Set the resolution for the interpolator
        dx = drive_beam.beam_size_x()/6
        num_x = int( (x_max - x_min)/dx )
        dy = drive_beam.beam_size_y()/6
        num_y = int( (y_max - y_min)/dy )
        dz = drive_beam.bunch_length()/10
        num_z = int( (z_start - z_end)/dz )
    
        xs_grid = np.linspace(x_min, x_max, num_x)  # [m]
        ys_grid = np.linspace(y_min, y_max, num_y)  # [m]
        zs_grid = np.linspace(z_start, z_end, num_z)  # [m]
        
        # Set up a 3D mesh grid
        X, Y, Z = np.meshgrid(xs_grid, ys_grid, zs_grid, indexing='ij')
        xs_grid_flat = X.flatten()*1e3  # [mm]
        ys_grid_flat = Y.flatten()*1e3  # [mm]
        zs_grid_flat = Z.flatten()*1e3  # [mm]
            
        # Probe beam E-field
        E_fields_beam, _ = sc_fields_obj.get_field(xs_grid_flat, ys_grid_flat, zs_grid_flat, np.zeros(len(zs_grid_flat)))  # [V/m]
        Ex = E_fields_beam[:,0]
        Ey = E_fields_beam[:,1]
        
        # Reshape the field component into a 3D array       
        Ex_3d = Ex.reshape(len(xs_grid), len(ys_grid), len(zs_grid))
        Ey_3d = Ey.reshape(len(xs_grid), len(ys_grid), len(zs_grid))
        
        return Ex_3d, Ey_3d, xs_grid, ys_grid, zs_grid


    # ==================================================
    def beam_field_interpolators(self, xs_grid, ys_grid, zs_grid, Ex_3d, Ey_3d):
        """
        Creates an interpolator for the input E-fields stored as 3D numpy arrays where the first, second and third dimensions correspond to positions along x, y and z.
        """
        
        # Create the interpolators
        Ex_interpolator = RegularGridInterpolator((xs_grid, ys_grid, zs_grid), Ex_3d, method='linear', bounds_error=True, fill_value=np.nan)
        Ey_interpolator = RegularGridInterpolator((xs_grid, ys_grid, zs_grid), Ey_3d, method='linear', bounds_error=True, fill_value=np.nan)

        return Ex_interpolator, Ey_interpolator



###################################################
#def intpl_beam_field(Ex_interpolator, Ey_interpolator, xs, ys, zs):
#    points = np.vstack((xs, ys, zs)).T
#    
#    # Interpolate beam E-field
#    Exs = Ex_interpolator(points)
#    Eys = Ey_interpolator(points)
#    
#    return Exs, Eys



###################################################
def grid_intpl_driver_beam_field(ion_motion_config):

    # Set up a 3D mesh grid
    xs_probe = ion_motion_config.xs_probe  # [m]
    ys_probe = ion_motion_config.ys_probe  # [m]
    zs_probe = ion_motion_config.zs_probe_driver  # [m]
    X, Y, Z = np.meshgrid(xs_probe, ys_probe, zs_probe, indexing='ij')
    xs_grid_flat = X.flatten()  # [m]
    ys_grid_flat = Y.flatten()  # [m]
    zs_grid_flat = Z.flatten()  # [m]
    points = np.vstack((xs_grid_flat, ys_grid_flat, zs_grid_flat)).T
        
    ## Interpolate beam E-field
    ##if field_comp == 'x':
    ##    driver_E_fields_comp = ion_motion_config.driver_Ex_interpolator(points)
    ##elif field_comp == 'y':
    ##    driver_E_fields_comp = ion_motion_config.driver_Ey_interpolator(points)
    ##else:
    ##    raise ValueError('ion_wakefield_perturbation_parallel(): Choose a valid field_comp (''x'' or ''y'').')
    ##
    ## Reshape the field component into a 3D array       
    ##driver_E_fields_3d = driver_E_fields_comp.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
    ##
    ##return driver_E_fields_3d

    # Determine the total number of available cores
    #total_cores = os.cpu_count()
    #
    # Use 80% of available cores
    #n_jobs = max(1, int(total_cores * 0.8))
#
    # Split probe grid coordinates into n_jobs segments
    #xs_grid_flat_segments = np.array_split(xs_grid_flat, n_jobs)
    #ys_grid_flat_segments = np.array_split(ys_grid_flat, n_jobs)
    #zs_grid_flat_segments = np.array_split(zs_grid_flat, n_jobs)
    #
    # Interpolate beam E-field in parallel
    #results = Parallel(n_jobs=n_jobs, backend='threading')(
    #    delayed(intpl_beam_field)(ion_motion_config.driver_Ex_interpolator, ion_motion_config.driver_Ey_interpolator, xs_grid_flat_segments[i], ys_grid_flat_segments[i], zs_grid_flat_segments[i])
    #    for i in range(n_jobs)
    #)
#
    # Extract Ex_3d and Ey_3d from the results
    #driver_Ex_list = [res[0] for res in results]  # Extract all Ex_3d arrays
    #driver_Ey_list = [res[1] for res in results]  # Extract all Ey_3d arrays
    #
    # Concatenate the extracted arrays along axis 0
    #driver_Exs = np.concatenate(driver_Ex_list, axis=0)
    #driver_Eys = np.concatenate(driver_Ey_list, axis=0)

    driver_Exs = ion_motion_config.driver_Ex_interpolator(points)
    driver_Eys = ion_motion_config.driver_Ey_interpolator(points)

    # Reshape the field component into a 3D array       
    driver_Exs_3d = driver_Exs.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
    driver_Eys_3d = driver_Eys.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
    
    return driver_Exs_3d, driver_Eys_3d



###################################################
def probe_driver_beam_field_comp(ion_motion_config, driver_sc_fields_obj, field_comp):
    """
    Probes the drive beam SpaceCharge_Field object and returns the drive beam E-field component stored in a 3D numpy array where the first, second and third dimensions correspond to positions along x, y and z. Needs to be called at every ion wakefield calculation step.
    """
    
    # Set up a 3D mesh grid
    xs_probe = ion_motion_config.xs_probe  # [m]
    ys_probe = ion_motion_config.ys_probe  # [m]
    zs_probe = ion_motion_config.zs_probe_driver  # [m]
    X, Y, Z = np.meshgrid(xs_probe, ys_probe, zs_probe, indexing='ij')
    xs_grid_flat = X.flatten()*1e3  # [mm]
    ys_grid_flat = Y.flatten()*1e3  # [mm]
    zs_grid_flat = Z.flatten()*1e3  # [mm]
        
    # Probe beam E-field
    E_fields_beam, _ = driver_sc_fields_obj.get_field(xs_grid_flat, ys_grid_flat, zs_grid_flat, np.zeros(len(zs_grid_flat)))  # [V/m]
    
    if field_comp == 'x':
        E_fields_comp = E_fields_beam[:,0]
    elif field_comp == 'y':
        E_fields_comp = E_fields_beam[:,1]
    else:
        raise ValueError('ion_wakefield_perturbation_parallel(): Choose a valid field_comp (''x'' or ''y'').')
    
    # Reshape the field component into a 3D array       
    E_fields_comp_3d = E_fields_comp.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
    
    return E_fields_comp_3d



###################################################
#def probe_driver_beam_field_comp_parallel(ion_motion_config, driver_sc_fields_obj, field_comp):
#    "Probes the drive beam SpaceCharge_Field object and returns the drive beam E-field component stored in a 3D numpy array where the first, second and third dimensions correspond to positions along x, y and z. Needs to be called at every ion wakefield calculation step."
#
#    # Set up a 3D mesh grid
#    xs_probe = ion_motion_config.xs_probe  # [m]
#    ys_probe = ion_motion_config.ys_probe  # [m]
#    zs_probe = ion_motion_config.zs_probe_driver  # [m]
#    X, Y, Z = np.meshgrid(xs_probe, ys_probe, zs_probe, indexing='ij')
#    xs_grid_flat = X.flatten()*1e3  # [mm]
#    ys_grid_flat = Y.flatten()*1e3  # [mm]
#    zs_grid_flat = Z.flatten()*1e3  # [mm]
#
#    # Determine the total number of available cores
#    total_cores = os.cpu_count()
#    
#    # Use 80% of available cores
#    n_jobs = max(1, int(total_cores * 0.8))
#
#    # Split probe grid coordinates into n_jobs segments
#    xs_grid_flat_segments = np.array_split(xs_grid_flat, n_jobs)
#    ys_grid_flat_segments = np.array_split(ys_grid_flat, n_jobs)
#    zs_grid_flat_segments = np.array_split(zs_grid_flat, n_jobs)
#
#    # Probe beam E-field in parallel
#    results = Parallel(n_jobs=n_jobs, backend='threading')(
#        delayed(probe_beam_field_segment)(driver_sc_fields_obj, xs_grid_flat_segments[i], ys_grid_flat_segments[i], zs_grid_flat_segments[i])
#        for i in range(n_jobs)
#    )
#
#    # Re-assemble the beam field array
#    E_fields_beam = np.concatenate(results, axis=0)  # [V/m]
#    
#    if field_comp == 'x':
#        E_fields_comp = E_fields_beam[:,0]
#    elif field_comp == 'y':
#        E_fields_comp = E_fields_beam[:,1]
#    else:
#        raise ValueError('ion_wakefield_perturbation_parallel(): Choose a valid field_comp (''x'' or ''y'').')
#    
#    # Reshape the field component into a 3D array       
#    E_fields_comp_3d = E_fields_comp.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
#    
#    return E_fields_comp_3d



###################################################
#from concurrent.futures import ThreadPoolExecutor
#def probe_driver_beam_field_comp_threading(ion_motion_config, driver_sc_fields_obj, field_comp, n_jobs=4):
#    # Set up a 3D mesh grid
#    xs_probe = ion_motion_config.xs_probe  # [m]
#    ys_probe = ion_motion_config.ys_probe  # [m]
#    zs_probe = ion_motion_config.zs_probe_driver  # [m]
#    X, Y, Z = np.meshgrid(xs_probe, ys_probe, zs_probe, indexing='ij')
#    xs_grid_flat = X.flatten() * 1e3  # [mm]
#    ys_grid_flat = Y.flatten() * 1e3  # [mm]
#    zs_grid_flat = Z.flatten() * 1e3  # [mm]
#
#    # Split probe grid coordinates into segments for parallel processing
#    xs_grid_flat_segments = np.array_split(xs_grid_flat, n_jobs)
#    ys_grid_flat_segments = np.array_split(ys_grid_flat, n_jobs)
#    zs_grid_flat_segments = np.array_split(zs_grid_flat, n_jobs)
#
#    # Parallel execution using ThreadPoolExecutor
#    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
#        # Map the probe_beam_field_segment function to each set of segments
#        results = list(executor.map(lambda args: probe_beam_field_segment(driver_sc_fields_obj, *args),
#                                    zip(xs_grid_flat_segments, ys_grid_flat_segments, zs_grid_flat_segments)))
#
#    # Re-assemble the beam field array
#    E_fields_beam = np.concatenate(results, axis=0)  # [V/m]
#
#    # Select the appropriate component of the field
#    if field_comp == 'x':
#        E_fields_comp = E_fields_beam[:, 0]
#    elif field_comp == 'y':
#        E_fields_comp = E_fields_beam[:, 1]
#    else:
#        raise ValueError("Choose a valid field_comp ('x' or 'y').")
#
#    # Reshape the field component into a 3D array       
#    E_fields_comp_3d = E_fields_comp.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
#    
#    return E_fields_comp_3d



###################################################
#def probe_driver_beam_field_chunked(ion_motion_config, driver_sc_fields_obj):
#    xs_probe = ion_motion_config.xs_probe
#    ys_probe = ion_motion_config.ys_probe
#    zs_probe = ion_motion_config.zs_probe_driver
#    
#    def process_chunk(xs_chunk, ys_chunk, zs_chunk, field_comp):
#        ion_motion_config_chunk = copy.deepcopy(ion_motion_config)  # Assume you can copy the config
#        ion_motion_config_chunk.xs_probe = xs_chunk
#        ion_motion_config_chunk.ys_probe = ys_chunk
#        ion_motion_config_chunk.zs_probe_driver = zs_chunk
#        return probe_driver_beam_field_comp(ion_motion_config_chunk, driver_sc_fields_obj, field_comp)
#
#    # Use 80% of available cores
#    n_jobs = max(1, int(total_cores * 0.8))
#    
#    # Split the arrays into chunks
#    xs_chunks = np.array_split(xs_probe, n_jobs)
#    ys_chunks = np.array_split(ys_probe, n_jobs)
#    zs_chunks = np.array_split(zs_probe, n_jobs)
#    
#    # Process chunks in parallel
#    results_x = Parallel(n_jobs=n_jobs, backend='threading')(
#        delayed(process_chunk)(xs_chunks[i], ys_chunks[i], zs_chunks[i], 'x') for i in range(n_jobs)
#    )
#    results_y = Parallel(n_jobs=n_jobs, backend='threading')(
#        delayed(process_chunk)(xs_chunks[i], ys_chunks[i], zs_chunks[i], 'y') for i in range(n_jobs)
#    )
#    
#    # Combine results back into a full array
#    E_field_x_3d = np.concatenate(results_x, axis=0)
#    E_field_y_3d = np.concatenate(results_y, axis=0)
#    
#    return E_field_x_3d, E_field_y_3d



###################################################
def probe_main_beam_field(ion_motion_config, main_sc_fields_obj, field_comp):
    """
    Probes the main beam SpaceCharge_Field object and returns the main beam E-field component stored in a 3D numpy array. Needs to be called at every ion wakefield calculation step.

    Parameters
    ----------
    ion_motion_config: IonMotionConfig object
        Contains the configurations for calculating the ion wakefield perturbation.
    
    main_sc_fields_obj: RF-Track SpaceCharge_Field object.
        Contains the beam electric and magnetic fields for the main beam calculated by RF-Track.

    field_comp: string
        Specifies the transverse component ('x' or 'y') of the wakefield.
        
    Returns
    ----------
    E_fields_comp_3d: [V/m] 3D float array
        Contains the beam E-field component where the first, second and third dimensions correspond to positions along x, y and z.
    """
    
    # Set up a 3D mesh grid
    xs_probe = ion_motion_config.xs_probe  # [m]
    ys_probe = ion_motion_config.ys_probe  # [m]
    zs_probe = ion_motion_config.zs_probe_main  # [m]
    X, Y, Z = np.meshgrid(xs_probe, ys_probe, zs_probe, indexing='ij')
    xs_grid_flat = X.flatten()*1e3  # [mm]
    ys_grid_flat = Y.flatten()*1e3  # [mm]
    zs_grid_flat = Z.flatten()*1e3  # [mm]
        
    # Probe beam E-field
    E_fields_beam, _ = main_sc_fields_obj.get_field(xs_grid_flat, ys_grid_flat, zs_grid_flat, np.zeros(len(zs_grid_flat)))  # [V/m]
    if field_comp == 'x':
        E_fields_comp = E_fields_beam[:,0]
    elif field_comp == 'y':
        E_fields_comp = E_fields_beam[:,1]
    else:
        raise ValueError('ion_wakefield_perturbation_parallel(): Choose a valid field_comp (''x'' or ''y'').')
    
    # Reshape the field component into a 3D array       
    E_fields_comp_3d = E_fields_comp.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
    
    return E_fields_comp_3d



###################################################
def ion_wakefield_perturbation(ion_motion_config, main_E_fields_comp_3d, driver_E_fields_comp_3d):
    """
    Calculates the ion wakefield perturbation to the otherwise linear background focusing kp*E0*r/2 by integrating the beam electric fields along z.

    Parameters
    ----------
    ion_motion_config: IonMotionConfig object
        Contains the configurations for calculating the ion wakefield perturbation.

    main_E_fields_comp_3d: 3D float array
        Contains the main beam E-field component where the first, second and third dimensions correspond to positions along x, y and z.
    
    driver_E_fields_comp_3d: 3D float array
        Contains the drive beam E-field component where the first, second and third dimensions correspond to positions along x, y and z.
        
    Returns
    ----------
    wakefield_perturbations: [V/m] 3D float array
        Contains the ion wakefield perturbation where the first, second and third dimensions correspond to positions along x, y and z. 
    """

    # Get parameters
    kp = k_p(ion_motion_config.plasma_ion_density)  # [m]
    ion_mass = ion_motion_config.ion_mass  # [kg]
    ion_charge_num = ion_motion_config.ion_charge_num
    zs_probe = ion_motion_config.zs_probe  # [m]
    grid_size_z = ion_motion_config.grid_size_z  # [m], the grid size along z

    # Get the empty region between the drive beam and main beam
    sep_E_fields_3d = construct_empty_field(ion_motion_config)

    # Concatenate the 3D beam field of the drive beam and main beam
    E_fields_comp_3d = np.concatenate([driver_E_fields_comp_3d, sep_E_fields_3d, main_E_fields_comp_3d], axis=2)
    
    # Integration along z using by splitting up the convolution integral
    integral = np.cumsum( zs_probe * E_fields_comp_3d * grid_size_z, axis=2) - zs_probe * np.cumsum(E_fields_comp_3d * grid_size_z, axis=2)
    
    wakefield_perturbations = ion_charge_num * SI.m_e/ion_mass * kp**2 * integral
    
    return wakefield_perturbations



###################################################
## Function to perform cumulative sum on a segment of the beam field (3D array)
#def compute_cumsum_segment(E_slice, zs_probe, grid_size_z):
#    #print(f"Processing slice i = {i} on core/process ID = {process_id}", flush=True)
#    #process_id = multiprocessing.current_process().pid
#
#    #start_time = time.time()
#    #print(f"Processing segment on process ID = {process_id} at time {start_time:.2f}", flush=True)
#    
#    
#    # Integration along z using by splitting up the convolution integral
#    integral = (np.cumsum(zs_probe * E_slice * grid_size_z, axis=2) 
#            - zs_probe * np.cumsum(E_slice * grid_size_z, axis=2))
#
#    #end_time = time.time()
#    #print(f"Finished segment on process ID = {process_id} at time {end_time:.2f}, duration = {end_time - start_time:.3e} seconds", flush=True)
#    
#    return integral
#
## Function for extracting a segment of the beam field
#def probe_beam_field_segment(sc_fields_obj, xs_segment, ys_segment, zs_segment):
#    zeros_segment = np.zeros_like(xs_segment)
#    E_fields, _ = sc_fields_obj.get_field(xs_segment, ys_segment, zs_segment, zeros_segment)
#    return E_fields



###################################################
#def ion_wakefield_perturbation_parallel(ion_motion_config, sc_fields_obj, tr_direction):
#    """
#    Calculates the ion wakefield perturbation to the otherwise linear background focusing kp*E0*r/2 by integrating the beam electric fields along z.
#
#    Parameters
#    ----------
#    ion_motion_config: IonMotionConfig object
#        Contains the configurations for calculating the ion wakefield perturbation.
#    
#    sc_fields_obj: RF-Track SpaceCharge_Field object.
#        Contains the beam electric and magnetic fields calculated by RF-Track.
#
#    tr_direction: string
#        Specifies the transverse direction ('x' or 'y') of the wakefield.
#        
#    Returns
#    ----------
#    wakefield_perturbations: [V/m] 3D float array
#        Contains the ion wakefield perturbation where the first, second and third dimensions correspond to positions along x, y and z. 
#    """
#    
#    # Get parameters
#    kp = k_p(ion_motion_config.plasma_ion_density)  # [m]
#    ion_mass = ion_motion_config.ion_mass  # [kg]
#    ion_charge_num = ion_motion_config.ion_charge_num
#    xs_probe = ion_motion_config.xs_probe  # [m]
#    ys_probe = ion_motion_config.ys_probe  # [m]
#    zs_probe = ion_motion_config.zs_probe  # [m]
#    grid_size_z = ion_motion_config.grid_size_z  # [m], the grid size along z
#    
#    # Set up a 3D mesh grid
#    X, Y, Z = np.meshgrid(xs_probe, ys_probe, zs_probe, indexing='ij')
#    xs_grid_flat = X.flatten()*1e3  # [mm]
#    ys_grid_flat = Y.flatten()*1e3  # [mm]
#    zs_grid_flat = Z.flatten()*1e3  # [mm]
#
#    # Determine the total number of available cores
#    total_cores = os.cpu_count()
#    
#    # Use 80% of available cores
#    n_jobs = max(1, int(total_cores * 0.8))
#
#    #start_time = time.time()
#
#    # Split probe grid coordinates into n_jobs segments
#    xs_grid_flat_segments = np.array_split(xs_grid_flat, n_jobs)
#    ys_grid_flat_segments = np.array_split(ys_grid_flat, n_jobs)
#    zs_grid_flat_segments = np.array_split(zs_grid_flat, n_jobs)
#
#    # Probe beam E-field in parallel
#    results = Parallel(n_jobs=n_jobs, backend='threading')(
#        delayed(probe_beam_field_segment)(sc_fields_obj, xs_grid_flat_segments[i], ys_grid_flat_segments[i], zs_grid_flat_segments[i])
#        for i in range(n_jobs)
#    )
#    
#    E_fields_beam = np.concatenate(results, axis=0)
#
#    #end_time = time.time()
#    #print('Probe time taken:', end_time - start_time, 'seconds')
#
#    if tr_direction == 'x':
#        E_fields_comp = E_fields_beam[:,0]
#    elif tr_direction == 'y':
#        E_fields_comp = E_fields_beam[:,1]
#    else:
#        raise ValueError('ion_wakefield_perturbation_parallel(): Choose a valid tr_direction (''x'' or ''y'').')
#
#    #start_time = time.time()
#    
#    # Reshape the field component into a 3D array       
#    E_fields_comp_3d = E_fields_comp.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
#
#    # Split E_fields_comp_3d into n_jobs segments along axis 0
#    E_fields_segments = np.array_split(E_fields_comp_3d, n_jobs, axis=0)
#    
#    # Process each E_fields_segment in parallel
#    results = Parallel(n_jobs=n_jobs)(
#        delayed(compute_cumsum_segment)(E_fields_segment, zs_probe, grid_size_z)
#        for E_fields_segment in E_fields_segments
#    )
#
#    # Stack the results back into a 3D array
#    integral = np.concatenate(results, axis=0)
#
#    #end_time = time.time()
#    #print('Integration time taken:', end_time - start_time, 'seconds')
#    
#    wakefield_perturbations = ion_charge_num * SI.m_e/ion_mass * kp**2 * integral
#    
#    return wakefield_perturbations

'''
    