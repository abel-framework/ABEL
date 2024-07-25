"""
Ion wakefield perturbation caused by ion motion


Ben Chen, 12 June 2024, University of Oslo
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as SI
from scipy.interpolate import RegularGridInterpolator

#from abel.classes.beam import Beam
from abel.utilities.plasma_physics import k_p
from abel.utilities.other import find_closest_value_in_arr
from abel.apis.rf_track.rf_track_api import calc_sc_fields_obj



# Contains calculation configurations for calculating the ion wakefield perturbation. Can be called when setting up a StagePrtclTransWakeInstability.
class IonMotionConfig():
    
    # ==================================================
    def __init__(self, drive_beam, main_beam, plasma_ion_density, ion_charge_num=1.0, ion_mass=None, num_z_cells=None, num_xy_cells_rft=50, num_xy_cells_probe=41, drive_beam_len_scale=1.0, update_factor=1.0):
        """
        Parameters
        ----------
        drive_beam: Beam object of drive beam.

        main_beam: Beam object of main beam.
        
        plasma_ion_density: [m^-3] float
            Plasma ion density.
        
        ion_charge_num: [e] float
            Plasma io charge in units of the elementary charge.
        
        num_z_cells: float
            Number of grid cells along z.

        num_xy_cells_rft: float
            Number of grid cells along x and y used in RF-Track for calculating beam electric fields.
            
        num_xy_cells_probe: float
            Number of grid cells along x and y used to probe beam electric fields calculated by RF-Track.

        update_factor: float
            Update ion wakefield perturbation when beam sizes have changed by this factor.

        xs_probe: [m] 1D float array
            x-coordinates used to probe beam electric fields calculated by RF-Track.

        ys_probe: [m] 1D float array
            y-coordinates used to probe beam electric fields calculated by RF-Track.

        zs_probe: [m] 1D float array
            z-coordinates used to probe beam electric fields calculated by RF-Track.

        grid_size_z: [m] float
            The size of grid cells along z.

        sc_fields_obj: RF-Track SpaceCharge_Field object.
            Contains the beam electric and magnatic fields calculated by RF-Track.
        """
    
        self.plasma_ion_density = plasma_ion_density  # [m^-3]
        self.ion_charge_num = ion_charge_num
    
        # Set the ion mass to default if necessary
        if ion_mass is None:
            ion_mass = 4.002602 * SI.physical_constants['atomic mass constant'][0]  # [kg], He mass.
        self.ion_mass = ion_mass

        # Set the number of cells
        if num_z_cells is None:
            num_z_cells = round(np.sqrt( len(drive_beam)+len(main_beam) )/2)
        self.num_z_cells = num_z_cells
        self.num_xy_cells_rft = num_xy_cells_rft
        self.num_xy_cells_probe = num_xy_cells_probe
        
        self.drive_beam_len_scale = drive_beam_len_scale
        self.update_factor = update_factor

        # Set the coordinates used to probe beam electric fields from RF-Track
        self.set_probing_coordinates(drive_beam, main_beam)

        # Set the RF-Track SpaceCharge_Field object
        self.set_sc_fields_obj(drive_beam, main_beam)

    
    # ==================================================
    def set_probing_coordinates(self, drive_beam, main_beam):

        drive_beam_len_scale = self.drive_beam_len_scale
        
        # Coordinates for probing the fields from RF-Track get_field()
        xy_min = np.min( [main_beam.xs().min(), main_beam.ys().min()] )
        xy_max = np.max( [main_beam.xs().max(), main_beam.ys().max()] )
        
        self.xs_probe = np.linspace( xy_min, xy_max, self.num_xy_cells_probe)  # [m]
        #self.ys_probe = np.linspace( xy_min, xy_max, self.num_xy_cells_probe)  # [m]
        self.ys_probe = np.linspace( main_beam.ys().min(), main_beam.ys().max(), self.num_xy_cells_probe)  # [m]

        zs_drive_beam = drive_beam.zs()

        # Find the drive beam head z-coordinate to be used for wakefield perturbation calculations
        _, z_eff_driver_head = find_closest_value_in_arr( zs_drive_beam, zs_drive_beam.min() + drive_beam_len_scale*(zs_drive_beam.max()-zs_drive_beam.min()) )
        
        z_max = z_eff_driver_head
        z_min = main_beam.zs().min()

        # Set the z-coordinates used to probe beam electric fields from RF-Track
        self.zs_probe = np.linspace(z_max, z_min, self.num_z_cells)  # [m], beam head facing start of array.

        # Set the size of grid cells along z
        self.grid_size_z = (self.zs_probe.max() - self.zs_probe.min()) / (self.num_z_cells-1)  # [m]

    
    # ==================================================
    def set_sc_fields_obj(self, drive_beam, main_beam):
        ion_motion_beam = drive_beam + main_beam
        ion_motion_beam.particle_mass = main_beam.particle_mass
        
        self.sc_fields_obj = calc_sc_fields_obj(ion_motion_beam, self.num_xy_cells_rft, self.num_xy_cells_rft, self.num_z_cells, num_t_bins=1)



# ==================================================
# Benedetti's ion motion quantifier Gamma
def ion_motion_quantifier(bunch_length, beam_peak_dens, plasma_density, ion_charge_num=1.0, ion_mass=None):
    if ion_mass is None:
        ion_mass = 39.95*SI.physical_constants['atomic mass constant'][0]  # [kg], Ar mass.
    return ion_charge_num*SI.m_e/ion_mass * beam_peak_dens/plasma_density * (k_p(plasma_density) * bunch_length)**2



# ==================================================
# Mehrling's ion motion quantifier Lambda
def ion_motion_quantifier2(beam_size_x, beam_size_y, bunch_length, beam_peak_current, ion_charge_num=1.0, ion_mass=None):
    if ion_mass is None:
        ion_mass = 39.95*SI.physical_constants['atomic mass constant'][0]  # [kg], Ar mass.
        
    I_A = 4*np.pi *SI.epsilon_0 *SI.m_e* SI.c**3 / SI.e  # [A], Alfvén current.
    
    return ion_charge_num*SI.m_e/ion_mass * beam_peak_current/I_A * bunch_length**2 / beam_size_x / beam_size_y



# ==================================================
def ion_wakefield_perturbation(ion_motion_config, sc_fields_obj, tr_direction):
    """
    Calculates the ion wakefield perturbation to the otherwise linear background focusing kp*E0*r/2 by integrating the beam electric fields along z.

    Parameters
    ----------
    ion_motion_config: IonMotionConfig object
        Contains the configurations for calculating the ion wakefield perturbation.
    
    sc_fields_obj: RF-Track SpaceCharge_Field object.
        Contains the beam electric and magnatic fields calculated by RF-Track.

    tr_direction: string
        Specifies the transverse direction ('x' or 'y') of the wakefield.
        
    Returns
    ----------
    wakefield_perturbations: [V/m] 3D float array
        Contains the ion wakefield perturbation where the first, second and third dimensions correspond to positions along x, y and z. 
    """

    # Set parameters
    skin_depth = 1/k_p(ion_motion_config.plasma_ion_density)  # [m]
    ion_mass = ion_motion_config.ion_mass  # [kg]
    ion_charge_num = ion_motion_config.ion_charge_num
    xs_probe = ion_motion_config.xs_probe  # [m]
    ys_probe = ion_motion_config.ys_probe  # [m]
    zs_probe = ion_motion_config.zs_probe  # [m]
    grid_size_z = ion_motion_config.grid_size_z  # [m], the grid size along z
    
    # Set up a 3D mesh grid
    X, Y, Z = np.meshgrid(xs_probe, ys_probe, zs_probe, indexing='ij')
    xs_grid_flat = X.flatten()*1e3  # [mm]
    ys_grid_flat = Y.flatten()*1e3  # [mm]
    zs_grid_flat = Z.flatten()*1e3  # [mm]

    # Probe beam E-field
    E_fields_beam, _ = sc_fields_obj.get_field(xs_grid_flat, ys_grid_flat, zs_grid_flat, np.zeros(len(zs_grid_flat)))  # [V/m]

    if tr_direction == 'x':
        E_fields_comp = E_fields_beam[:,0]
    elif tr_direction == 'y':
        E_fields_comp = E_fields_beam[:,1]
    else:
        raise ValueError('ion_wakefield_perturbation(): Choose a valid tr_direction (''x'' or ''y'').')
    
    # Reshape the field component into a 3D array       
    E_fields_comp_3d = E_fields_comp.reshape(len(xs_probe), len(ys_probe), len(zs_probe))
    
    # Integration along z using by splitting up to convolution integral
    integral = np.cumsum( zs_probe * E_fields_comp_3d, axis=2) * grid_size_z - zs_probe * np.cumsum(E_fields_comp_3d, axis=2) * grid_size_z
    
    wakefield_perturbations = ion_charge_num * SI.m_e/ion_mass/skin_depth**2 * integral
    
    return wakefield_perturbations



# ==================================================
# Interpolate the ion wake field perturbation to beam macroparticle positions.
def intplt_ion_wakefield_perturbation(beam, wakefield_perturbations, ion_motion_config):
    
    # Create the interpolator
    interpolator = RegularGridInterpolator((ion_motion_config.xs_probe, ion_motion_config.ys_probe, ion_motion_config.zs_probe), wakefield_perturbations, method='linear', bounds_error=True, fill_value=np.nan)

    # Coordinates for interpolating the field values
    xs_intplt = beam.xs()  # [m]
    ys_intplt = beam.ys()  # [m]
    zs_intplt = beam.zs()  # [m]
    
    # Combine the coordinates into an array of points
    points = np.vstack((xs_intplt, ys_intplt, zs_intplt)).T  # TODO: check array dimensions.
    
    # Interpolate the field values at the given coordinates
    interpolated_wakefield_perturbations = interpolator(points)  # 1D array

    return interpolated_wakefield_perturbations

