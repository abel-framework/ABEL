"""
Ion wakefield perturbation caused by ion motion


Ben Chen, 12 June 2024, University of Oslo
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as SI

#from abel.classes.beam import Beam
from abel.utilities.plasma_physics import k_p
from abel.apis.rf_track.rf_track_api import calc_sc_fields_obj



# Contains calculation configurations for calculating the ion wakefield perturbation. Can be called when setting up a StagePrtclTransWakeInstability.
class IonMotionConfig():
    
    # ==================================================
    def __init__(self, drive_beam, main_beam, plasma_ion_density, ion_charge_num=1.0, ion_mass=None, num_z_cells=None, num_xy_cells_rft=50, num_xy_cells_extract=41, update_factor=1.0):
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
            
        num_xy_cells_extract: float
            Number of grid cells along x and y used to extract beam electric fields from RF-Track.

        update_factor: float
            Update ion wakefield perturbation when beam sizes have changed by this factor.

        xs_extract: [m] 1D array
            x-coordinates used to extract beam electric fields from RF-Track.

        ys_extract: [m] 1D array
            y-coordinates used to extract beam electric fields from RF-Track.

        zs_extract: [m] 1D array
            z-coordinates used to extract beam electric fields from RF-Track.

        sc_fields_obj: RF-Track SpaceCharge_Field object.
            
        """
    
        self.plasma_ion_density = plasma_ion_density  # [m^-3]
        self.ion_charge_num = ion_charge_num
    
        # Set the ion mass to default if necessary
        if ion_mass is None:
            ion_mass = 4.002602 * SI.physical_constants['atomic mass constant'][0]  # [kg], He mass.
        self.ion_mass = ion_mass
    
        if num_z_cells is None:
            num_z_cells = round(np.sqrt( len(drive_beam)+len(main_beam) )/2)
        self.num_z_cells = num_z_cells
        self.num_xy_cells_rft = num_xy_cells_rft
        self.num_xy_cells_extract = num_xy_cells_extract
        
        self.update_factor = update_factor
        
        self.set_extraction_coordinates(drive_beam, main_beam)
        self.set_sc_fields_obj(drive_beam, main_beam)

    
    # ==================================================
    def set_extraction_coordinates(self, drive_beam, main_beam):
        
        # Coordinates for extracting the fields from RF-Track get_field()
        xy_min = np.min( [main_beam.xs().min(), main_beam.ys().min()] )
        xy_max = np.max( [main_beam.xs().max(), main_beam.ys().max()] )
        
        self.xs_extract = np.linspace( xy_min, xy_max, self.num_xy_cells_extract)  # [m]
        self.ys_extract = self.xs_extract  # [m]
        
        z_max = np.max( [drive_beam.zs().max(), main_beam.zs().max()] )
        z_min = np.min( [drive_beam.zs().min(), main_beam.zs().min()] )
        
        self.zs_extract = np.linspace(z_max, z_min, self.num_z_cells)  # [m], beam head facing start of array.
        self.grid_size_z = (self.zs_extract.max() - self.zs_extract.min()) / (self.num_z_cells-1)  # [m], the grid size along z

    
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
    """

    # Set parameters
    plasma_ion_density = ion_motion_config.plasma_ion_density  # [m^-3]
    ion_mass = ion_motion_config.ion_mass  # [kg]
    ion_charge_num = ion_motion_config.ion_charge_num
    xs_extract = ion_motion_config.xs_extract  # [m]
    ys_extract = ion_motion_config.ys_extract  # [m]
    zs_extract = ion_motion_config.zs_extract  # [m]
    grid_size_z = ion_motion_config.grid_size_z  # [m], the grid size along z
    skin_depth = 1/k_p(plasma_ion_density)  # [m]
    
    # Set up a 3D mesh grid
    X, Y, Z = np.meshgrid(xs_extract, ys_extract, zs_extract, indexing='ij')
    xs_grid_flat = X.flatten()*1e3  # [mm]
    ys_grid_flat = Y.flatten()*1e3  # [mm]
    zs_grid_flat = Z.flatten()*1e3  # [mm]

    # Extract beam E-field
    E_fields_beam, _ = sc_fields_obj.get_field(xs_grid_flat, ys_grid_flat, zs_grid_flat, np.zeros(len(zs_grid_flat)))  # [V/m]

    if tr_direction == 'x':
        E_fields_comp = E_fields_beam[:,0]
    elif tr_direction == 'y':
        E_fields_comp = E_fields_beam[:,1]
    else:
        raise ValueError('ion_wakefield_perturbation(): Choose a valid tr_direction (''x'' or ''y'').')
    
    # Reshape the field component into a 3D array       
    E_fields_comp_3d = E_fields_comp.reshape(len(xs_extract), len(ys_extract), len(zs_extract))

    wakefield_head_idx = 0
    E_fields_comp_3d = E_fields_comp_3d[ :, :, wakefield_head_idx:E_fields_comp_3d.shape[2] ]
    zs_extract = zs_extract[wakefield_head_idx:len(zs_extract)]
    
    # Integration along z using by splitting up to convolution integral
    integral = np.cumsum( zs_extract * E_fields_comp_3d, axis=2) * grid_size_z - zs_extract * np.cumsum(E_fields_comp_3d, axis=2) * grid_size_z
    
    wakefield_perturbations = ion_charge_num * SI.m_e/ion_mass/skin_depth**2 * integral

    return wakefield_perturbations



# ==================================================
# Interpolate the ion wake field perturbation to beam particle positions.
def interp_ion_wakefield_perturbation(beam, wakefield_perturbations):

    from scipy.interpolate import RegularGridInterpolator
    # Create the interpolator
    interpolator = RegularGridInterpolator((xs_extract, ys_extract, zs_extract), wakefield_perturbations, method='linear', bounds_error=True, fill_value=np.nan)

    # Coordinates where you want to interpolate the field values
    xs_interp = beam.xs()*1e3  # [mm]
    ys_interp = beam.ys()*1e3  # [mm]
    zs_interp = beam.zs()*1e3  # [mm]
    
    # Combine the coordinates into an array of points
    points = np.vstack((xs_interp, ys_interp, zs_interp)).T  # TODO: check array dimensions.
    
    # Interpolate the field values at the given coordinates
    interpolated_wakefield_perturbations = interpolator(points)  # 1D array

    return interpolated_wakefield_perturbations