from abel import Beam
import scipy.constants as SI
import numpy as np
import wake_t
from abel.utilities.plasma_physics import k_p, blowout_radius

# convert from WakeT particle bunch to ABEL beam
def wake_t_bunch2beam(bunch):
    
    # extract phase space (with charge)
    phasespace = bunch.get_6D_matrix_with_charge()
    
    # initialize beam
    beam = Beam()
    
    # set the phase space of the ABEL beam
    beam.set_phase_space(Q=sum(phasespace[6]),
                         xs=phasespace[0],
                         ys=phasespace[2],
                         zs=phasespace[4], 
                         pxs=phasespace[1]*SI.c*SI.m_e,
                         pys=phasespace[3]*SI.c*SI.m_e,
                         pzs=phasespace[5]*SI.c*SI.m_e)
    
    return beam
      
    
# convert from ABEL beam to WakeT particle bunch
def beam2wake_t_bunch(beam, name='beam'):
    
    # convert the beam
    bunch = wake_t.ParticleBunch(w=beam.weightings(),
                                 x=beam.xs(),
                                 y=beam.ys(),
                                 xi=beam.zs(), 
                                 px=beam.pxs()/(SI.c*SI.m_e),
                                 py=beam.pys()/(SI.c*SI.m_e),
                                 pz=beam.pzs()/(SI.c*SI.m_e),
                                 name=name)
    
    return bunch


# ==================================================
def plasma_stage_setup(plasma_density, abel_drive_beam, abel_main_beam, stage_length=None, lambda_betatron_frac=0.1):
    """
    Calculates step size, box sizes etc. to set up a Wake-T plasma acceleration stage (https://wake-t.readthedocs.io/en/latest/api/beamline_elements/_autosummary/wake_t.beamline_elements.PlasmaStage.html#plasmastage).
    
    Parameters
    ----------
    plasma_density: [m^-3] float
        Plasma density.
        
    abel_drive_beam: ABEL Beam object
        Drive beam.

    abel_main_beam: ABEL Beam object
        Main beam.

    stage_length: [m] float, optional
        Length of the plasma acceleration stage. If not given, is set to the same as one step size.

    lambda_betatron_frac: float, optional
        The fraction of betatron wavelength used to set the step size.
    
        
    Returns
    ----------
    plasma_stage: Wake-T plasma acceleration stage
    """

    # Find stepsize
    k_beta = k_p(plasma_density)/np.sqrt(2*min(abel_main_beam.gamma(), abel_drive_beam.gamma()/2))
    lambda_betatron = 2*np.pi/k_beta

    if stage_length is None:
        stage_length = lambda_betatron*lambda_betatron_frac

    # Determines how often the plasma wakefields should be updated. For example, if dz_fields=10e-6, the plasma wakefields are only updated every time the simulation window advances by 10 micron
    dz_fields = lambda_betatron*lambda_betatron_frac
    
    # Number of times along the stage in which the particle distribution should be returned
    n_out = 1

    # Set box ranges
    R_blowout = blowout_radius(plasma_density, abel_drive_beam.peak_current())
    box_size_r = np.max([4/k_p(plasma_density), 3*R_blowout])
    box_min_z = abel_drive_beam.z_offset() - 3.3 * R_blowout
    box_max_z = min(abel_drive_beam.z_offset() + 6 * abel_drive_beam.bunch_length(), np.max(abel_drive_beam.zs()) + 0.5*R_blowout)
    box_range_z = [box_min_z, box_max_z]
    
    # Calculate number of cells
    num_cell_xy = 256
    dr = box_size_r/num_cell_xy
    num_cell_z = round((box_max_z-box_min_z)/dr)

    # Set up a Wake-T plasma stage
    plasma_stage = wake_t.beamline_elements.PlasmaStage(length=stage_length, density=plasma_density, wakefield_model='quasistatic_2d',
                                                r_max=box_size_r, r_max_plasma=box_size_r, xi_min=box_min_z, xi_max=box_max_z, 
                                                n_out=n_out, n_r=num_cell_xy, n_xi=int(num_cell_z), dz_fields=dz_fields, ppc=1)

    return plasma_stage


# ==================================================
def extract_initial_and_final_Ez_rho(tmpfolder):
    """
    Extract the initial and final Ez and plasma charge densities.
    """
    
    from types import SimpleNamespace
    from openpmd_viewer import OpenPMDTimeSeries

    wake_t_evolution = SimpleNamespace()
    
    wake_t_evolution.initial = SimpleNamespace()
    wake_t_evolution.initial.plasma = SimpleNamespace()
    wake_t_evolution.initial.plasma.density = SimpleNamespace()
    wake_t_evolution.initial.plasma.wakefield = SimpleNamespace()
    wake_t_evolution.initial.plasma.wakefield.onaxis = SimpleNamespace()
    
    wake_t_evolution.final = SimpleNamespace()
    wake_t_evolution.final.plasma = SimpleNamespace()
    wake_t_evolution.final.plasma.density = SimpleNamespace()
    wake_t_evolution.final.plasma.wakefield = SimpleNamespace()
    wake_t_evolution.final.plasma.wakefield.onaxis = SimpleNamespace()
        
    # prepare to read simulation data
    source_path = tmpfolder + 'hdf5/'
    ts = OpenPMDTimeSeries(source_path)      # TODO: figure out how to skip this write and read

    # extract initial on-axis wakefield
    Ez0, metadata0_Ez = ts.get_field(field='E', coord='z', slice_across=['r'], iteration=min(ts.iterations))
    wake_t_evolution.initial.plasma.wakefield.onaxis.zs = metadata0_Ez.z
    wake_t_evolution.initial.plasma.wakefield.onaxis.Ezs = Ez0
    
    # extract final on-axis wakefield
    Ez, metadata_Ez = ts.get_field(field='E', coord='z', slice_across=['r'], iteration=max(ts.iterations))
    wake_t_evolution.final.plasma.wakefield.onaxis.zs = metadata_Ez.z
    wake_t_evolution.final.plasma.wakefield.onaxis.Ezs = Ez
    
    # extract initial plasma density
    rho0_plasma, metadata0_plasma = ts.get_field(field='rho', iteration=min(ts.iterations))
    wake_t_evolution.initial.plasma.density.extent = metadata0_plasma.imshow_extent
    wake_t_evolution.initial.plasma.density.rho = -(rho0_plasma/SI.e)
    wake_t_evolution.initial.plasma.density.metadata = metadata0_plasma

    # extract final plasma density
    rho_plasma, metadata_plasma = ts.get_field(field='rho', iteration=max(ts.iterations))
    wake_t_evolution.final.plasma.density.extent = metadata_plasma.imshow_extent
    wake_t_evolution.final.plasma.density.rho = -(rho_plasma/SI.e)
    wake_t_evolution.final.plasma.density.metadata = metadata_plasma

    return wake_t_evolution








