# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import scipy.constants as SI

def wake_t_bunch2beam(bunch):
    """
    Convert a Wake-T ``ParticleBunch`` to an ABEL ``Beam``.

    
    Parameters
    ----------
    bunch : Wake-T ``ParticleBunch``
        Wake-T ``ParticleBunch`` to be converted.

        
    Returns
    -------
    beam : ``Beam``
        The converted ABEL ``Beam``.

        
    References
    ----------
    .. [1] Wake-T documentation: https://wake-t.readthedocs.io/en/latest/index.html
    """

    from abel.classes.beam import Beam
    
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
      
    
# ==================================================
def beam2wake_t_bunch(beam, name='beam'):
    """
    Convert an ABEL ``Beam`` to a Wake-T ``ParticleBunch``.

    
    Parameters
    ----------
    beam : ``Beam``
        The ABEL ``Beam`` to be converted.

    name : str, optional
        The name for the Wake-T ``ParticleBunch``. Defaults to ``'beam'``.

        
    Returns
    -------
    bunch : Wake-T ``ParticleBunch``
        The converted Wake-T ``ParticleBunch``.

        
    References
    ----------
    .. [1] Wake-T documentation: https://wake-t.readthedocs.io/en/latest/index.html
    """
    
    import wake_t

    # convert the beam
    bunch = wake_t.ParticleBunch(w=beam.weightings(),
                                 x=beam.xs(),
                                 y=beam.ys(),
                                 xi=beam.zs(), 
                                 px=beam.pxs()/(SI.c*SI.m_e),
                                 py=beam.pys()/(SI.c*SI.m_e),
                                 pz=beam.pzs()/(SI.c*SI.m_e),
                                 name=name)
    
    bunch.prop_distance = beam.location
    
    return bunch


# ==================================================
def wake_t_hdf5_load(file_path, species='beam'):
    """
    Load an ABEL beam from a Wake-T HDF5 file (OpenPMD format).

    
    Parameters
    ----------
    file_path : str
        Path to the Wake-T HDF5 output file.

    species : str, optional
        Specifies the name of the beam to be extracted.

        
    Returns
    ----------
    beam : ``Beam``
        The converted ABEL ``Beam``.

        
    References
    ----------
    .. [1] Wake-T documentation: https://wake-t.readthedocs.io/en/latest/index.html
    """

    from abel import Beam
    import openpmd_api as io
    import numpy as np
    
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
    Calculates step size, box sizes etc. to set up a `Wake-T plasma acceleration stage <https://wake-t.readthedocs.io/en/latest/api/beamline_elements/_autosummary/wake_t.beamline_elements.PlasmaStage.html#plasmastage>`_. 
    

    Parameters
    ----------
    plasma_density : [m^-3] float
        Plasma density.
        
    abel_drive_beam : ``Beam``
        Drive beam.

    abel_main_beam : ``Beam``, optional
        Main beam, can be used to determine the betatreon wavelength.

    stage_length: [m] float, optional
        Length of the plasma acceleration stage. If not given, is set to the 
        same as one step size.

    dz_fields : [m], float, optional
        Determines how often the plasma wakefields should be updated. For 
        example, if ``dz_fields=10e-6``, the plasma wakefields are only updated 
        every time the simulation window advances by 10 micro meter. The default 
        is set to be slightly longer than ``stage_length`` such that the plasma 
        wakefields are only calculated once.

    num_cell_xy : float, optional
        Number of grid elements along r to calculate the wakefields.

    n_out : int, optional
        Number of times along the stage in which the particle distribution 
        should be returned.
        
    box_size_r : [m], float, optional
        Determines the transverse size of the simulation domain 
        [``-box_size_r``, ``box_size_r``] where the plasma wakefield will be 
        calculated.

    box_min_z : [m], float, optional
        Minimum longitudinal (speed of light frame) position for the simulation 
        domain in which the plasma wakefield will be calculated.

    box_max_z : [m], float, optional
        Maximum longitudinal (speed of light frame) position for the simulation 
        domain in which the plasma wakefield will be calculated.
    
        
    Returns
    ----------
    plasma_stage : Wake-T ``PlasmaStage``


    References
    ----------
    .. [1] Wake-T documentation: https://wake-t.readthedocs.io/en/latest/index.html
    """

    import numpy as np
    import warnings
    import wake_t
    from abel.utilities.plasma_physics import k_p, blowout_radius
    from abel.classes.stage.stage import SimulationDomainSizeError

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
        dz_fields = 1.1*stage_length  # Set to larger than the stage length so that the fields will only be calculated once.

    # Set box ranges
    R_blowout = blowout_radius(plasma_density, abel_drive_beam.peak_current())

    if box_size_r is None:
        box_size_r = np.max([2/k_p(plasma_density), 2*R_blowout, 1.2*abel_drive_beam.rs().max()])

    if box_min_z is None:
        if driver_only:
            box_min_z = abel_drive_beam.z_offset() - 4 * R_blowout
        else:
            box_min_z = min( abel_drive_beam.z_offset()-4*R_blowout, abel_main_beam.z_offset()-6*abel_main_beam.bunch_length() )
    if box_max_z is None:
            box_max_z = min( abel_drive_beam.z_offset()+6*abel_drive_beam.bunch_length(), np.max(abel_drive_beam.zs())+0.5*R_blowout )

    # Check the simulation domain
    if driver_only:
        if box_min_z > abel_drive_beam.zs().min() or box_max_z < abel_drive_beam.zs().max():
            raise SimulationDomainSizeError(f"The simulation box is too short. Min box z: {box_min_z*1e6 :.3f} um, max box z: {box_max_z*1e6 :.3f} um, min particle z: {abel_drive_beam.zs().min()*1e6 :.3f} um, max particle z: {abel_drive_beam.zs().max()*1e6 :.3f} um")
        if box_size_r < 1.2*abel_drive_beam.rs().max():
            #raise SimulationDomainSizeError('The simulation box is too narrow.')
            warnings.warn(f"The simulation box is too narrow. Max box r: {box_size_r*1e6 :.3f} um, max particle r: {abel_drive_beam.rs().max()*1e6 :.3f} um.")
    else:
        if box_min_z > abel_main_beam.zs().min() or box_max_z < abel_drive_beam.zs().max():
            raise SimulationDomainSizeError(f"The simulation box is too short. Min box z: {box_min_z*1e6 :.3f} um, max box z: {box_max_z*1e6 :.3f} um, min particle z: {abel_main_beam.zs().min()*1e6 :.3f} um, max particle z: {abel_drive_beam.zs().max()*1e6 :.3f} um")
        
        max_r = np.max( [abel_drive_beam.rs().max(), abel_main_beam.rs().max()] )
        if box_size_r < 1.2*max_r:
            warnings.warn(f"The simulation box is too narrow. Max box r: {box_size_r*1e6 :.3f} um, max particle r: {max_r*1e6 :.3f} um.")
        
    # Calculate number of cells
    dr = box_size_r/num_cell_xy
    num_cell_z = round((box_max_z-box_min_z)/dr)

    # Construct a Wake-T plasma stage
    print("Constructing - parameters are:", flush=True)
    print("stage_length=", stage_length, flush=True)
    print("plasma_density=", plasma_density, flush=True)
    print("box_size_r=", box_size_r, flush=True)
    print("box_min_z=", box_min_z, flush=True)
    print("box_max_z=", box_max_z, flush=True)
    print("n_out=", n_out, flush=True)
    print("num_cell_xy=", num_cell_xy, flush=True)
    print("num_cell_z=", num_cell_z, int(num_cell_z), flush=True)
    print("dz_fields=", dz_fields, flush=True)
    plasma_stage = wake_t.beamline_elements.PlasmaStage(length=stage_length, density=plasma_density, wakefield_model='quasistatic_2d',
                                                r_max=box_size_r, r_max_plasma=box_size_r, xi_min=box_min_z, xi_max=box_max_z, 
                                                n_out=n_out, n_r=num_cell_xy, n_xi=int(num_cell_z), dz_fields=dz_fields, ppc=1)
    print("Complete.",flush=True)
    return plasma_stage

    
# ==================================================
def run_single_step_wake_t(plasma_density, drive_beam, beam):
    """
    Sets up and runs a single time step Wake-T simulation to calculate the 
    wakefields.

    Note that any transverse offsets ``drive_beam`` may have is subtracted from
    both ``drive_beam`` and ``beam``, so that the simulation is performed in a 
    transversly shifted frame. The coordinates for the results are also given in 
    this shifted frame.

    
    Parameters
    ----------
    drive_beam : ``Beam``
        Drive beam.

    beam : ``Beam``
        Main beam.

        
    Returns
    ----------
    wake_t_evolution : :class:`types.SimpleNamespace`
        Contains the 2D plasma density and wakefields for the initial and 
        final time steps.


    References
    ----------
    .. [1] Wake-T documentation: https://wake-t.readthedocs.io/en/latest/index.html
    """

    import os, uuid, shutil
    from abel.utilities.plasma_physics import k_p
    from abel.CONFIG import CONFIG

    print("IN SINGLE_STEP_WAKE_T", flush=True)

    # The drive beam must be centred around r=0 before being used in Wake-T
    driver_x_offset = drive_beam.x_offset()
    driver_y_offset = drive_beam.y_offset()
    drive_beam_wakeT_xs = drive_beam.xs()
    drive_beam.set_xs(drive_beam_wakeT_xs - driver_x_offset)
    drive_beam_wakeT_ys = drive_beam.ys()
    drive_beam.set_ys(drive_beam_wakeT_ys - driver_y_offset)

    # Also subtract the drive beam offset from the main beam used in Wake-T
    beam0_xs = beam.xs()
    beam.set_xs(beam0_xs - driver_x_offset)
    beam0_ys = beam.ys()
    beam.set_ys(beam0_ys - driver_y_offset)
    
    # Construct a Wake-T plasma acceleration stage
    wakeT_xy_res = 0.1*beam.bunch_length()
    wakeT_max_box_r = 4/k_p(plasma_density)
    wakeT_num_cell_xy = int(wakeT_max_box_r/wakeT_xy_res)
    print("PLASMA_STAGE_SETUP...", flush=True)
    plasma_stage = plasma_stage_setup(plasma_density, drive_beam, beam, stage_length=None, dz_fields=None, num_cell_xy=wakeT_num_cell_xy)

    print("MAKE TEMP FOLDER...", flush=True)
    # Make temp folder
    if not os.path.exists(CONFIG.temp_path):
        os.mkdir(CONFIG.temp_path)
    tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
    if not os.path.exists(tmpfolder):
        os.mkdir(tmpfolder)

    print("CONVERT BEMAS TO WAKE T BUNCHES...", flush=True)
    # Convert beams to Wake-T bunches
    driver0_wake_t = beam2wake_t_bunch(drive_beam, name='driver')
    beam0_wake_t = beam2wake_t_bunch(beam, name='beam')

    # Perform the Wake-T simulation
    print("PLASMA_STAGE.TRACK()...", flush=True)
    plasma_stage.track([driver0_wake_t, beam0_wake_t], opmd_diag=True, diag_dir=tmpfolder, show_progress_bar=False)
    print("DONE PLASMA_STAGE.TRACK()...", flush=True)
    
    # Extract the fields
    wake_t_evolution = extract_initial_and_final_Ez_rho(tmpfolder)

    print("EXTRACED FIELDS...", flush=True)
    
    # Remove temporary directory
    shutil.rmtree(tmpfolder)

    return wake_t_evolution


# ==================================================
def extract_initial_and_final_Ez_rho(tmpfolder):
    """
    Extract the initial and final Ez and plasma charge densities.

    
    Parameters
    ----------
    tmpfolder : str
        The path to the directory containing the Wake-T HDF5 files (OpenPMD 
        format).

        
    Returns
    ----------
    wake_t_evolution : :class:`types.SimpleNamespace`
        Contains the 2D plasma density and wakefields for the initial and 
        final time steps.


    References
    ----------
    .. [1] Wake-T documentation: https://wake-t.readthedocs.io/en/latest/index.html
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
    ts = OpenPMDTimeSeries(source_path)      # TODO: figure out how to skip this write and read. Look at e.g. plasma_stage.fields[0]

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


# ==================================================
def wake_t_remove_halo_particles(bunch, nsigma=20):
    """
    Bunch halo cleaning (outliers in x, xp, y and yp).

    
    Parameters
    ----------
    bunch : Wake-T ``ParticleBunch``
        Wake-T ``ParticleBunch`` to be cleaned.

    nsigma : float, optional
        Sets the filter to filter out particles that satisfy 
        ``bunch.x - x_offset > nsigma * beam_size_x`` or 
        ``bunch.y - y_offset > nsigma * beam_size_y`` or 
        ``bunch.px/bunch.pz - x_angle > nsigma * divergence_x`` or 
        ``bunch.py/bunch.pz - y_angle > nsigma * divergence_y``. Defaults to 20.

        
    Returns
    ----------
    filtered_bunch : Wake-T ``ParticleBunch``
        Cleaned Wake-T ``ParticleBunch``.


    References
    ----------
    .. [1] Wake-T documentation: https://wake-t.readthedocs.io/en/latest/index.html
    """

    import numpy as np
    import wake_t
    from abel.utilities.statistics import weighted_mean, weighted_std

    weightings = bunch.w
    x_offset = weighted_mean(bunch.x, weightings, clean=True)
    y_offset = weighted_mean(bunch.y, weightings, clean=True)
    xps = bunch.px/bunch.pz
    yps = bunch.py/bunch.pz
    x_angle = weighted_mean(xps, weightings, clean=True)
    y_angle = weighted_mean(yps, weightings, clean=True)

    beam_size_x = weighted_std(bunch.x, weightings, clean=True)
    beam_size_y = weighted_std(bunch.y, weightings, clean=True)
    divergence_x = weighted_std(xps, weightings, clean=True)
    divergence_y = weighted_std(yps, weightings, clean=True)
    
    xfilter = np.abs(bunch.x-x_offset) > nsigma*beam_size_x
    xpfilter = np.abs(xps-x_angle) > nsigma*divergence_x
    yfilter = np.abs(bunch.y-y_offset) > nsigma*beam_size_y
    ypfilter = np.abs(yps-y_angle) > nsigma*divergence_y
    filter = np.logical_or(np.logical_or(xfilter, xpfilter), np.logical_or(yfilter, ypfilter))

    if hasattr(filter, 'len'):
        if len(filter) == len(bunch):
            filter = np.where(filter)
    
    phase_space = bunch.get_6D_matrix_with_charge()
    filtered_phasespace = np.ascontiguousarray(np.delete(phase_space, filter, 1))

    filtered_bunch = wake_t.ParticleBunch(w=filtered_phasespace[6] / bunch.q_species,
                                 x=filtered_phasespace[0],
                                 y=filtered_phasespace[2],
                                 xi=filtered_phasespace[4], 
                                 px=filtered_phasespace[1],
                                 py=filtered_phasespace[3],
                                 pz=filtered_phasespace[5],
                                 name=bunch.name)
    
    filtered_bunch.prop_distance = bunch.prop_distance

    return filtered_bunch



# ==================================================
def wakeT_r_E_filter(bunch, r_thres, pz_thres):
    """
    Bunch cleaning for particles located further away than a given distance 
    ``r_thres`` from axis. Also discards particles that have lower momentum than 
    pz_thres.

    
    Parameters
    ----------
    bunch : Wake-T ``ParticleBunch``
        
    r_thres : [m] float
        Transverse coordinate (x^2+y^2) threshold for selecting particles.
        
    pz_thres : [beta*gamma] float
        Momentum threshold for selecting particles.

        
    Returns
    ----------
    filtered_bunch : Wake-T ``ParticleBunch``
        Cleaned Wake-T ``ParticleBunch``.


    References
    ----------
    .. [1] Wake-T documentation: https://wake-t.readthedocs.io/en/latest/index.html
    """

    import numpy as np
    import wake_t

    rs = np.sqrt(bunch.x**2 + bunch.y**2)
    r_filter = rs > r_thres

    #Es = momentum2energy(bunch.pz*SI.c*SI.m_e)
    pzs = bunch.pz
    pz_filter = pzs < pz_thres

    filter = np.logical_or(r_filter, pz_filter)

    if hasattr(filter, 'len'):
        if len(filter) == len(bunch):
            filter = np.where(filter)
    
    phase_space = bunch.get_6D_matrix_with_charge()
    filtered_phasespace = np.ascontiguousarray(np.delete(phase_space, filter, 1))

    filtered_bunch = wake_t.ParticleBunch(w=filtered_phasespace[6] / bunch.q_species,
                                 x=filtered_phasespace[0],
                                 y=filtered_phasespace[2],
                                 xi=filtered_phasespace[4], 
                                 px=filtered_phasespace[1],
                                 py=filtered_phasespace[3],
                                 pz=filtered_phasespace[5],
                                 name=bunch.name)
    
    filtered_bunch.prop_distance = bunch.prop_distance

    return filtered_bunch


