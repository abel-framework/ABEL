from abel import Beam
import scipy.constants as SI
import numpy as np
import wake_t
from abel.utilities.plasma_physics import k_p, blowout_radius
from abel.classes.stage.stage import SimulationDomainSizeError


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
    
    beam.location = bunch.prop_distance
    
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
def wake_t_hdf5_load(file_path, species='beam'):
    """
    Load an ABEL beam from a Wake-T HDF5 file (OpenPMD format).

    Parameters
    ----------
    file_path: string
        Path to the Wake-T HDF5 output file.

    species: string, optional
        Specifies the name of the beam to be extracted.


    Returns
    ----------
    beam: Beam object
    """

    import openpmd_api as io
    
    # load file
    series = io.Series(file_path, io.Access.read_only)
    
    # find index (use last one) by default
    *_, index = series.iterations
    
    # get particle data
    particles = series.iterations[index].particles[species]
    
    # get attributes
    charge = particles["charge"][io.Record_Component.SCALAR].get_attribute("value")
    mass = particles["mass"][io.Record_Component.SCALAR].get_attribute("value")
    
    # extract phase space
    if "id" in particles:
        ids = particles["id"][Record_Component.SCALAR].load_chunk()
    else:
        ids = None
    weightings = particles["weighting"][io.Record_Component.SCALAR].load_chunk()
    xs = particles['position']['x'].load_chunk()
    ys = particles['position']['y'].load_chunk()
    zs = particles['position']['z'].load_chunk()
    pxs_unscaled = particles['momentum']['x'].load_chunk()
    pys_unscaled = particles['momentum']['y'].load_chunk()
    pzs_unscaled = particles['momentum']['z'].load_chunk()
    series.flush()  # Synchronization mechanism to confirm that all requested data is loaded into memory and available for further processing.
    
    # apply SI scaling
    pxs = pxs_unscaled * particles['momentum']['x'].unit_SI
    pys = pys_unscaled * particles['momentum']['y'].unit_SI
    pzs = pzs_unscaled * particles['momentum']['z'].unit_SI
    
    # make beam
    beam = Beam()
    beam.set_phase_space(Q=np.sum(weightings*charge), xs=xs, ys=ys, zs=zs, pxs=pxs, pys=pys, pzs=pzs, weightings=weightings)
    beam.particle_mass = mass
    
    # add metadata to beam
    try:
        beam.location = particles['positionOffset']['z'].load_chunk()
    except:
        beam.location = None

    try:
        beam.trackable_number = series.iterations[index].get_attribute("trackable_number")
    except:
        beam.trackable_number = None
    
    try:
        beam.stage_number = series.iterations[index].get_attribute("stage_number")
    except:
        beam.stage_number = None
    
    try:
        beam.num_bunches_in_train = series.iterations[index].get_attribute("num_bunches_in_train")
    except:
        beam.num_bunches_in_train = None
    
    try:
        beam.bunch_separation = series.iterations[index].get_attribute("bunch_separation")
    except:
        beam.bunch_separation = None
        
    return beam


# ==================================================
def plasma_stage_setup(plasma_density, abel_drive_beam, abel_main_beam=None, stage_length=None, dz_fields=None, num_cell_xy=256, n_out=1, box_size_r=None, box_min_z=None, box_max_z=None):
    """
    Calculates step size, box sizes etc. to set up a Wake-T plasma acceleration stage (https://wake-t.readthedocs.io/en/latest/api/beamline_elements/_autosummary/wake_t.beamline_elements.PlasmaStage.html#plasmastage).
    
    Parameters
    ----------
    plasma_density: [m^-3] float
        Plasma density.
        
    abel_drive_beam: ABEL Beam object
        Drive beam.

    abel_main_beam: ABEL Beam object, optional
        Main beam, can be used to determine the betatreon wavelength.

    stage_length: [m] float, optional
        Length of the plasma acceleration stage. If not given, is set to the same as one step size.

    dz_fields: [m], float, optional
        Determines how often the plasma wakefields should be updated. For example, if ``dz_fields=10e-6``, the plasma wakefields are only updated every time the simulation window advances by 10 micro meter. The default is set to be slightly longer than ``stage_length`` such that the plasma wakefields are only calculated once.

    num_cell_xy: float, optional
        Number of grid elements along r to calculate the wakefields.

    n_out: int, optional
        Number of times along the stage in which the particle distribution should be returned.
        
    box_size_r: [m], float, optional
        Determines the transverse size of the simulation domain [``-box_size_r``, ``box_size_r``] where the plasma wakefield will be calculated.

    box_min_z: [m], float, optional
        Minimum longitudinal (speed of light frame) position for the simulation domain in which the plasma wakefield will be calculated.

    box_max_z: [m], float, optional
        Maximum longitudinal (speed of light frame) position for the simulation domain in which the plasma wakefield will be calculated.
    
        
    Returns
    ----------
    plasma_stage: Wake-T plasma acceleration stage
    """

    if abel_main_beam is None:
        driver_only = True
    else:
        driver_only = False
        
    # Find stepsize
    if stage_length is None:
        if driver_only:
            k_beta = k_p(plasma_density)/np.sqrt(2*abel_drive_beam.gamma())
        else:
            k_beta = k_p(plasma_density)/np.sqrt(2*min(abel_main_beam.gamma(), abel_drive_beam.gamma()/2))
        lambda_betatron = 2*np.pi/k_beta
        
        stage_length = 0.05*lambda_betatron

    if dz_fields is None:
        dz_fields = 1.1*stage_length

    # Set box ranges
    R_blowout = blowout_radius(plasma_density, abel_drive_beam.peak_current())
    if box_size_r is None:
        box_size_r = np.max([4/k_p(plasma_density), 3*R_blowout])
    if box_min_z is None:
        box_min_z = abel_drive_beam.z_offset() - 4 * R_blowout
    if box_max_z is None:
        box_max_z = min(abel_drive_beam.z_offset() + 6 * abel_drive_beam.bunch_length(), np.max(abel_drive_beam.zs()) + 0.5*R_blowout)
    box_range_z = [box_min_z, box_max_z]

    # Check the simulation domain
    if driver_only:
        if box_min_z > abel_drive_beam.zs().min() or box_max_z < abel_drive_beam.zs().max():
            raise SimulationDomainSizeError('The simulation box is too short.')
        if box_size_r < abel_drive_beam.rs().max():
            raise SimulationDomainSizeError('The simulation box is too narrow.')
    else:
        if box_min_z > abel_main_beam.zs().min() or box_max_z < abel_drive_beam.zs().max():
            raise SimulationDomainSizeError('The simulation box is too short.')
        max_r = np.max( [abel_drive_beam.rs().max(), abel_main_beam.rs().max()] )
        if box_size_r < max_r:
            raise SimulationDomainSizeError('The simulation box is too narrow.')
        
        
    # Calculate number of cells
    dr = box_size_r/num_cell_xy
    num_cell_z = round((box_max_z-box_min_z)/dr)

    # Construct a Wake-T plasma stage
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








