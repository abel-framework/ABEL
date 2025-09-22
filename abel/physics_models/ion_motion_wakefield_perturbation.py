"""
Ion wakefield perturbation caused by ion motion


Ben Chen, 12 June 2024, University of Oslo
"""


import numpy as np
import scipy.constants as SI
import copy, os, warnings

from abel.classes.beam import Beam
from abel.utilities.plasma_physics import k_p
from abel.utilities.statistics import weighted_std
from abel.utilities.other import find_closest_value_in_arr, pad_downwards, pad_upwards
from abel.wrappers.rf_track.rf_track_wrapper import calc_sc_fields_obj



class IonMotionConfig():
    
    # ==================================================
    def __init__(self, drive_beam, main_beam, plasma_ion_density, ion_charge_num=1.0, ion_mass=None, num_z_cells_main=None, num_x_cells_rft=50, num_y_cells_rft=50, num_xy_cells_probe=41, uniform_z_grid=False, driver_x_jitter=0.0, driver_y_jitter=0.0, ion_wkfld_update_period=1, drive_beam_update_period=0, wake_t_fields=None):
        """
        Contains calculation configuration for calculating the ion wakefield 
        perturbation.
        
        Parameters
        ----------
        drive_beam : ABEL ``Beam`` object
            Drive beam.

        main_beam : ABEL ``Beam`` object
            Main beam.
        
        plasma_ion_density : [m^-3] float
            Plasma ion density.
        
        ion_charge_num : [e] float, optional
            Plasma io charge in units of the elementary charge. Default set to 
            1.0.

        ion_mass : [kg] float, optional
            Mass of ions. Default set to helium mass.
        
        num_z_cells_main : int, optional
            Number of grid cells along z. If set to ``None``, the value is 
            calculated from the drive beam and main beam properties. 

        num_x(y)_cells_rft : int, optional
            Number of grid cells along x and y used in RF-Track for calculating 
            beam electric fields. Default set to 50.
            
        num_xy_cells_probe : int, optional
            Number of grid cells along x and y used to probe beam electric 
            fields calculated by RF-Track. Default set to 41.

        driver_x_jitter, driver_y_jitter : float, optional
            Standard deviation of driver xy-jitter used for mesh-refinement. 
            Default set to 0.0.
        
        uniform_z_grid : bool, optional
            Determines whether the grid along z is uniform (True) or finely 
            resolved along the drive beam and main beam regions, while the 
            region between the beams are coarsely resolved (False). Default set 
            to ``False``.

        ion_wkfld_update_period : int, optional
            Sets the update period for calculating the ion wakefield 
            perturbation in units of time step. E.g. 
            ``ion_wkfld_update_period=1`` updates the ion wakefield perturbation 
            every time step.

        ...
        """

        # Check the inputs
        if not isinstance(ion_wkfld_update_period, int):
            raise TypeError("ion_wkfld_update_period must be an integer.")
        elif ion_wkfld_update_period < 1:
            raise ValueError("ion_wkfld_update_period must be a positive integer >= 1.")
        
        if not isinstance(drive_beam_update_period, int):
            raise TypeError("drive_beam_update_period must be an integer.")
        elif drive_beam_update_period < 0:
            raise ValueError("drive_beam_update_period must be an integer >= 0.")
        
            # TODO: Add setters and getters that check the inputs.


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

        # Coordinates used to probe drive beam and main beam electric fields calculated by RF-Track. 
        # Span the both the beams and the region in between them in every direction.
        # Updated at every every ion wakefield calculation step.
        # [m] 1D float ndarray
        self.xs_probe = None
        self.ys_probe = None
        self.zs_probe = None

        # Separate z-coordinates spanning three regions
        self.zs_probe_main = None  # [m] 1D float ndarray, spans main_beam.zs().
        self.zs_probe_separation = None  # [m] 1D float ndarray, z-coordinates that span the region between the beams.
        self.zs_probe_driver = None  # [m] 1D float ndarray, spans drive_beam.zs().

        # The transverse extent of the drive beam [m]
        self.xlims_driver_sc = None
        self.ylims_driver_sc = None

        # The size of grid cells along z [m]. Can be non-uniform
        self.grid_size_z = None

        # Store the std of driver jitters for transverse mesh refinement
        self.driver_x_jitter = driver_x_jitter
        self.driver_y_jitter = driver_y_jitter

        # Ion wakefield perturbation update configuraiton quantities
        self.ion_wkfld_update_period = ion_wkfld_update_period
        self.update_ion_wakefield = True  # Internal variable that is True when it is time to update the ion wakefield perturbation.

        # Drive beam evolution quantities TODO: replace with new drive beam evolution modelling
        self.drive_beam_update_period = drive_beam_update_period
        self.wake_t_fields = wake_t_fields
        
        # Store ion wakefield perturbation for time steps that skip calculating the wakefield
        self.Wx_perts = None
        self.Wy_perts = None
        
        # Set the coordinates used to probe beam electric fields from RF-Track
        self.set_probing_coordinates(drive_beam, main_beam, set_driver_sc_coords=True)
        driver_sc_fields_obj = self.assemble_driver_sc_fields_obj(drive_beam)
        self.driver_sc_fields_obj = driver_sc_fields_obj

    
    # ==================================================
    def set_probing_coordinates(self, drive_beam, main_beam, set_driver_sc_coords=False):
        """
        Sets the coordinates used to probe the beam fields of the drive beam 
        and main beam. Has to be called at every ion wakefield calculation step.

        The probing xy-coordinates are determined by the ``main_beam`` 
        xy-coordinates, while the probing longitudinal coordinates span both the 
        drive beam, main beam and the separation region between them.

        Parameters
        ----------
        drive_beam : ABEL ``Beam`` object
            Drive beam.

        main_beam : ABEL ``Beam`` object
            Main beam.

        set_driver_sc_coords : bool, optional
            Sets the xy limits used by 
            ``IonMotionConfig.assemble_driver_sc_fields_obj()``.


        Returns
        ----------
        ``None``
        """
        
        if drive_beam.zs().min() < main_beam.zs().max():
            warnings.warn('There are some overlap between drive beam and main beam.', UserWarning)
        
            if drive_beam.z_offset()-2*drive_beam.bunch_length() < main_beam.zs().max():
                raise ValueError('There is significant overlap between drive beam and main beam.')

        if set_driver_sc_coords:
            # Set the xy limits used by assemble_driver_sc_fields_obj()
            x_max = np.max([drive_beam.xs().max(), main_beam.xs().max()])
            x_min = np.min([drive_beam.xs().min(), main_beam.xs().min()])
            y_max = np.max([drive_beam.ys().max(), main_beam.ys().max()])
            y_min = np.min([drive_beam.ys().min(), main_beam.ys().min()])
            xy_padding = 2.0
            x_max = pad_upwards(x_max, padding=xy_padding)
            x_min = pad_downwards(x_min, padding=xy_padding)
            y_max = pad_upwards(y_max, padding=xy_padding)
            y_min = pad_downwards(y_min, padding=xy_padding)
            self.xlims_driver_sc = np.array([x_min, x_max])  # [m]
            self.ylims_driver_sc = np.array([y_min, y_max])  # [m]
        
        # Transverse coordinates for probing the fields from RF-Track get_field()
        xs_probe = np.linspace( main_beam.xs().min(), main_beam.xs().max(), self.num_xy_cells_probe)  # [m]
        ys_probe = np.linspace( main_beam.ys().min(), main_beam.ys().max(), self.num_xy_cells_probe)  # [m]

        # Modify xs_probe and ys_probe if mesh refinement is required to resolve the drive beam xy-jitter
        if self.driver_x_jitter != 0.0 or self.driver_y_jitter != 0.0:

            # Take care of when only one of the jitters is non-zero in order to resolve both directions
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
            #print('xs_probe.min:', self.xs_probe.min(), 'xlims_driver_sc.min:', self.xlims_driver_sc.min(), 'xs_probe.max:', self.xs_probe.max(), 'ys_probe.min:', self.ys_probe.min(), 'ylims_driver_sc.min:', self.ylims_driver_sc.min(), 'ys_probe.max:', self.ys_probe.max())
            warnings.warn("The range of the probing coordinates is larger than the driver probing coordinates. This may lead to a slower performance due to extrapolations when extracting the driver beam fields.", UserWarning)

        # Set the z-coordinates used to probe beam electric fields from RF-Track
        self.zs_probe_main = np.linspace( main_beam.zs().max(), main_beam.zs().min(), self.num_z_cells_main )  # [m], beam head facing start of array.
        
        if self.uniform_z_grid:
            self.grid_size_z = np.abs(np.diff(self.zs_probe_main)[0])  # [m], the size of grid cells along z
            driver_num_z = int( (drive_beam.zs().max() - pad_upwards(main_beam.zs().max()))/self.grid_size_z )
            self.zs_probe_driver = np.linspace( drive_beam.zs().max(), drive_beam.zs().min(), driver_num_z )  # [m], beam head facing start of array.

            sep_num_z = int( (pad_downwards(self.zs_probe_driver.min()) - pad_upwards(self.zs_probe_main.max()) )/self.grid_size_z )
            self.zs_probe_separation = np.linspace( pad_downwards(self.zs_probe_driver.min()), pad_upwards(self.zs_probe_main.max()), sep_num_z )  # [m], beam head facing start of array. Space between driver and main beam.
            
            self.zs_probe = np.concatenate( [self.zs_probe_driver, self.zs_probe_separation, self.zs_probe_main] )  # [m], beam head facing start of array.
            
        else:
            dz = weighted_std(drive_beam.zs(), drive_beam.weightings())/5
            driver_num_z = int( (drive_beam.zs().max() - drive_beam.zs().min())/dz )
            self.zs_probe_driver = np.linspace( drive_beam.zs().max(), drive_beam.zs().min(), driver_num_z )  # [m], beam head facing start of array.

            sep_num_z = int( (pad_downwards(self.zs_probe_driver.min()) - pad_upwards(self.zs_probe_main.max()) )/dz )
            self.zs_probe_separation = np.linspace( pad_downwards(self.zs_probe_driver.min()), pad_upwards(self.zs_probe_main.max()), sep_num_z )  # [m], beam head facing start of array. Space between driver and main beam.
            self.zs_probe = np.concatenate( [self.zs_probe_driver, self.zs_probe_separation, self.zs_probe_main] )  # [m], beam head facing start of array.
            self.grid_size_z = np.abs( np.insert(np.diff(self.zs_probe), 0, 0.0) )

        if np.any(np.diff(self.zs_probe) > 0):
            raise ValueError('zs_probe is not in descending order.')

    
    # ==================================================
    def assemble_driver_sc_fields_obj(self, drive_beam):
        """
        Creates a RF-Track ``SpaceCharge_Field`` object for the drive beam. Only 
        needs to be called at the start of a stage for non-evolving drive beam.

        Parameters
        ----------
        drive_beam : ABEL ``Beam`` object
            Drive beam.
        
        
        Returns
        ----------
        RF-Track ``SpaceCharge_Field`` object
        """
        
        x_min, x_max = self.xlims_driver_sc
        y_min, y_max = self.ylims_driver_sc
        
        z_end = pad_downwards(self.zs_probe_driver.min())
        z_start = pad_upwards(self.zs_probe_driver.max())
        
        X, Y, Z = np.meshgrid([x_min, x_max], [y_min, y_max], [z_start, z_end], indexing='ij')


        # In order to avoid extrapolations when probing a RF-Track ``SpaceCharge_Field`` object, need to add eight "ghost particles" with zero charge in order to artificially enlarge the simulation box when constructing a RF-Track ``SpaceCharge_Field`` object (the range of a ``SpaceCharge_Field`` object only spands the particle coordinates of the particles).
        empty_beam = Beam()
        empty_beam.set_phase_space(Q=0,
                                   xs=X.flatten(),
                                   ys=Y.flatten(),
                                   zs=Z.flatten(), 
                                   uxs=np.ones_like(X.flatten())*drive_beam.uxs()[0],
                                   uys=np.ones_like(X.flatten())*drive_beam.uys()[0],
                                   uzs=np.ones_like(X.flatten())*drive_beam.uzs()[0],
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
    Probes the RF-Track drive beam ``SpaceCharge_Field`` object and returns the 
    drive beam E-fields stored in 3D numpy arrays where the first, second and 
    third dimensions correspond to positions along x, y and z. Needs to be 
    called at every ion wakefield calculation step.

    Parameters
    ----------
    ion_motion_config : ``IonMotionConfig`` object
        Contains the configurations for calculating the ion wakefield 
        perturbation.
    
    driver_sc_fields_obj : RF-Track ``SpaceCharge_Field`` object.
        Contains the beam electric and magnetic fields for the drive beam 
        calculated by RF-Track.
        
    Returns
    ----------
    driver_Exs_3d, driver_Eys_3d : [V/m] 3D float ndarray
        Contain the beam E-field components where the first, second and third 
        dimensions correspond to positions along x, y and z.
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
    Returns a zero 3D numpy where the first, second and third dimensions 
    correspond to positions along x, y and z. Needs to be called at every ion 
    wakefield calculation step.
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
    Creates a ``SpaceCharge_Field`` object for the main beam. Needs to be called 
    at every ion wakefield calculation step.

    Parameters
    ----------
    ion_motion_config : ``IonMotionConfig`` object
        Contains information on the size of the simulation box and number of 
        cells along each directions.
    
    main_beam : ABEL ``Beam`` object
        Main beam.
    
    
    Returns
    ----------
    RF-Track ``SpaceCharge_Field`` object
    """
    
    return calc_sc_fields_obj(main_beam, ion_motion_config.num_x_cells_rft, ion_motion_config.num_y_cells_rft, ion_motion_config.num_z_cells_main, num_t_bins=1)



###################################################
def probe_main_beam_field(ion_motion_config, main_sc_fields_obj):
    """
    Probes the main beam ``SpaceCharge_Field`` object and returns the main beam 
    E-field component stored in a 3D numpy array. Needs to be called at every 
    ion wakefield calculation step.

    Parameters
    ----------
    ion_motion_config : ``IonMotionConfig`` object
        Contains the configurations for calculating the ion wakefield 
        perturbation.
    
    main_sc_fields_obj : RF-Track ``SpaceCharge_Field`` object
        Contains the beam electric and magnetic fields for the main beam 
        calculated by RF-Track.
        
    Returns
    ----------
    Exs_3d, Eys_3d : [V/m] 3D float ndarray
        Contain the beam E-field components where the first, second and third 
        dimensions correspond to positions along x, y and z.
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
    
    return Exs_3d, Eys_3d



###################################################
def ion_wakefield_perturbation(ion_motion_config, main_Exs_3d, main_Eys_3d, driver_Exs_3d, driver_Eys_3d):
    """
    Calculates the ion wakefield perturbation to the otherwise linear background 
    focusing kp*E0*r/2 by integrating the beam electric fields along z. The 
    method is based on C. Benedetti's model [1]_.

    
    Parameters
    ----------
    ion_motion_config : ``IonMotionConfig`` object
        Contains the configurations for calculating the ion wakefield 
        perturbation.

    main_Exs_3d, main_Eys_3d : [V/m] 3D float ndarray
        Contains the main beam E-field component where the first, second and 
        third dimensions correspond to positions along x, y and z.
    
    driver_Exs_3d, driver_Eys_3d : [V/m] 3D float ndarray
        Contains the drive beam E-field component where the first, second and 
        third dimensions correspond to positions along x, y and z.
        

    Returns
    ----------
    wakefield_perturbations : [V/m] 3D float ndarray
        Contains the ion wakefield perturbation where the first, second and 
        third dimensions correspond to positions along x, y and z.

        
    References
    ----------
    .. [1] C. Benedetti, C. B. Schroeder CB, E. Esarey and W. P. Leemans, "Emittance preservation in plasma-based accelerators with ion motion," Phys. Rev. Accel. Beams. 20, 111301 (2017);. https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.20.111301
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

    from scipy.interpolate import RegularGridInterpolator
    
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

    import matplotlib.pyplot as plt
    
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

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap  # For customising colour maps
    
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
    axes[0].set_ylabel(r"$\mathcal{W}_x$ [GV/m]")
    axes[0].grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
    twinax = axes[0].twinx()
    twinax.fill_between(x=beam.transverse_profile_x()[1]*1e6, y1=beam.charge_sign()*beam.transverse_profile_x()[0], y2=0, color='g', alpha=0.3)
    twinax.set_yticks([])
    
    #sort_y_indices = np.argsort(ys)
    axes[1].scatter(ys*1e6, plasma_density*SI.e/(2*SI.epsilon_0)*ys/1e9 - intpl_Wy_perts/1e9, c=zs*1e6, cmap=cmap, s=3.0)
    axes[1].plot(ys*1e6, plasma_density*SI.e/(2*SI.epsilon_0)*ys/1e9)
    axes[1].set_xlabel('y [µm]')
    axes[1].set_ylabel(r"$\mathcal{W}_y$ [GV/m]")
    axes[1].grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
    twinax = axes[1].twinx()
    twinax.fill_between(x=beam.transverse_profile_y()[1]*1e6, y1=beam.charge_sign()*beam.transverse_profile_y()[0], y2=0, color='g', alpha=0.3)
    twinax.set_yticks([])

    # Set label and other properties for the colorbar
    cbar_ax = fig.add_axes([0.15, 0.96, 0.7, 0.02])   # The four values in the list correspond to the left, bottom, width, and height of the new axes, respectively.
    fig.colorbar(p, cax=cbar_ax, orientation='horizontal', label=r'$z$ [µm]')


    