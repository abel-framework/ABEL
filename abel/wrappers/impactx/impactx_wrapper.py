# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import numpy as np
from abel.CONFIG import CONFIG
import uuid, os, shutil
import scipy.constants as SI
from types import SimpleNamespace

def run_impactx(lattice, beam0, nom_energy=None, runnable=None, keep_data=False, save_beams=False, space_charge=False, csr=False, isr=False, isr_on_ref_part=True, verbose=False):
    """
    Run an ImpactX [1]_ particle-tracking simulation using a specified lattice and 
    input beam. 
    
    This function 

    1. Sets up an ImpactX simulation specifying allocated resources.
    
    2. Configures relevant physics options including space charge effects, CSR 
       (coherent synchrotron radiation) and ISR (incoherent synchrotron 
       radiation). 
    
    3. Executes the simulation, converts the results back into an ABEL beam 
       object, and returns the tracked beam and its evolution data.
       
    Parameters
    ----------
    lattice : list
        Contains a sequence consisting of ImpactX beamline elements (e.g., 
        drifts, bends, lenses, or multipoles). Defines the lattice through which 
        ``beam0`` is tracked.
        
    beam0 : ``Beam``
        Input ABEL beam object representing the particle distribution before 
        entering the lattice.
        
    nom_energy : [eV] float, optional
        Nominal beam energy used to determine the ISR order and scaling. If 
        ``None``, first order ISR effects are always used. If greater than 
        1 TeV, third order ISR effects are enabled. Defaults to ``None``.

    runnable : ``Runnable``, optional
        ABEL ``Runnable`` object. If provided and ``save_beams`` is ``True``, 
        the beam files are saved to the shot directory specified by 
        :func:`runnable.shot_path() <abel.Runnable.shot_path>`. The pickled file 
        ``runnable.obj`` is also saved to the same directory.

    keep_data : bool, optional
        If ``True``, the ImpactX run directory containing simulation outputs and 
        the pickled file ``runnable.obj`` is preserved after the simulation 
        completes. If ``False``, it is deleted automatically. Defaults to 
        ``False``.

    save_beams : bool, optional
        If ``True``, saves beam snapshots for each simulation step via 
        ``extract_beams()``. Defaults to ``False``.

    space_charge : bool, optional
        Enable space charge effects in the simulation. Defaults to ``False``.

    csr : bool, optional
        Enable CSR modeling. When enabled, the ImpactX simulation parameter 
        ``sim.csr_bins`` is set to 150. Defaults to ``False``.

    isr : bool, optional
        Enable ISR modeling. The ISR order is automatically determined based on 
        ``nom_energy``. Defaults to ``False``.

    verbose : bool, optional
        If ``True``, prints detailed ImpactX tracking output to the console. 
        Defaults to ``False``.


    Returns
    -------
    beam : ``Beam``
        Output ABEL ``Beam`` object after transport through the given lattice.

    evol : :class:`types.SimpleNamespace`
        Beam parameter evolution data extracted from the ImpactX run (e.g. 
        emittances, beta functions, centroid offsets).

    References
    ----------
    .. [1] ImpactX Documentation: https://impactx.readthedocs.io/en/latest/
    """
    
    # create a new directory
    original_folder = os.getcwd()
    if runnable is not None:
        runfolder = CONFIG.temp_path + str(uuid.uuid4())
    else:
        runfolder = 'impactx_sims'
    if not os.path.exists(runfolder):
        os.makedirs(runfolder)
    os.chdir(runfolder)
    
    # make simulation
    sim = initialize_impactx_sim(verbose=verbose)
    
    # physics flags
    sim.space_charge = space_charge
    sim.csr = csr
    if csr:
        sim.csr_bins = 150
    sim.isr = isr
    if isr:
        sim.isr_on_ref_part = isr_on_ref_part
        if nom_energy is not None and nom_energy > 1e12:
            sim.isr_order = 3
        else:
            sim.isr_order = 1
    
    # convert to ImpactX particle container
    _, sim = beam2particle_container(beam0, sim=sim, verbose=verbose, nom_energy=nom_energy)
    
    # assign the lattice
    sim.lattice.extend(lattice)
    
    # run simulation
    sim.verbose = int(verbose)
    if verbose:
        sim.track_particles()
    else:
        eval('sim.track_particles()')
    
    # convert back to ABEL beam
    beam = particle_container2beam(sim.particle_container())

    # clean shutdown
    sim.finalize()

    # extract evolution
    evol = extract_evolution();

    # copy meta data from input beam (will be iterated by super)
    beam.trackable_number = beam0.trackable_number
    beam.stage_number = beam0.stage_number
    beam.location = beam0.location

    # change back to original directory
    os.chdir(original_folder)

    # save beams if requested
    if runnable is not None and save_beams:
        beams = extract_beams(runfolder, beam0=beam0)
        for beam_step in beams:
            beam_step.save(runnable=runnable)
        
    # delete run folder (or move it to the run folder)
    if not keep_data:
        shutil.rmtree(runfolder)
    else:
        new_dir = shutil.move(runfolder, runnable.shot_path()) # TODO: what if runnable is None?
        os.rename(new_dir, os.path.join(os.path.dirname(new_dir),'impactx_sims'))

    return beam, evol


# ==================================================
def run_envelope_impactx(lattice, distr, nom_energy=None, peak_current=None, space_charge="2D", runnable=None, keep_data=False, verbose=False):
    """
    Run an ImpactX envelope-tracking simulation using a given lattice and 
    initial beam distribution.

    Parameters
    ----------
    lattice : list
		Contains a sequence consisting of ImpactX beamline elements (e.g., 
        drifts, bends, lenses, or multipoles). Defines the lattice through which 
        ``beam0`` is tracked.

    distr : ImpcatX ``distribution``
        ImpactX distribution function specifying the initial beam envelope 
        parameters and correlations.

    nom_energy : [eV] float, optional
        Nominal kinetic energy of the reference particle. Used to set 
        relativistic scaling. Defaults to ``None``.

    peak_current : [A] float, optional
        Peak current of the beam. Defaults to ``None``.

    space_charge : str | bool, optional
        Space charge model to use. Options include ``'2D'``, ``'3D'``, or 
        ``False`` to disable. Defaults to ``'2D'``.

    runnable : ``Runnable``, optional
		ABEL ``Runnable`` object. Used to specify the run folder if provided.

    keep_data : bool, optional
        If ``True``, the temporary ImpactX run directory is preserved after 
        simulation. Defaults to ``False``.

    verbose : bool, optional
        If ``True``, prints ImpactX progress and diagnostic information. 
        Defaults to ``False``.

    Returns
    -------
    evol : :class:`types.SimpleNamespace`
        Object containing the envelope evolution along the lattice, including 
        beam size, emittance, and energy as a function of the longitudinal 
        coordinate.
    """
    
    # create a new directory
    original_folder = os.getcwd()
    if runnable is not None:
        runfolder = CONFIG.temp_path + str(uuid.uuid4())
    else:
        runfolder = 'impactx_sims'
    if not os.path.exists(runfolder):
        os.makedirs(runfolder)
    os.chdir(runfolder)

    # make simulation
    sim = initialize_impactx_sim(verbose=verbose)

    # physics flags
    sim.space_charge = space_charge
    
    # reference particle
    ref = sim.particle_container().ref_particle()
    ref.set_charge_qe(1.0).set_mass_MeV(SI.m_e*SI.c**2/SI.e/1e6).set_kin_energy_MeV(nom_energy/1e6) # TODO: what if nom_energy is None?

    # initialize the envelope
    sim.init_envelope(ref, distr, peak_current) # TODO: what if peak_current is None?

    # assign the lattice
    sim.lattice.extend(lattice)
    
    # run simulation
    sim.verbose = int(verbose)
    if verbose:
        sim.track_envelope()
    else:
        eval('sim.track_envelope()')
    
    # clean shutdown
    sim.finalize()

    # extract evolution
    evol = extract_evolution();
    
    # change back to original directory
    os.chdir(original_folder)

    # delete run folder
    if not keep_data:
        shutil.rmtree(runfolder)

    return evol


def initialize_impactx_sim(verbose=False):
    """Initialize the ImpactX simulation."""
    
    #import amrex.space3d as amr
    from impactx import ImpactX

    # make simulation object
    sim = ImpactX()

    # serial run on one CPU core
    sim.omp_threads = 1

    # set ImpactX verbosity
    sim.verbose = int(verbose)
    sim.tiny_profiler = verbose
    
    # enable diagnostics
    sim.diagnostics = True
    #   Note: Diagnostics in every element's slice steps is verbose.
    #         Disable for speed if if only beam monitors and final results are needed.
    sim.slice_step_diagnostics = True

    # set numerical parameters and IO control
    sim.particle_shape = 2  # B-spline order
    
    # domain decomposition & space charge mesh
    sim.init_grids()
    
    return sim


# ==================================================
def extract_beams(path='', runnable=None, beam0=None):
    """
    Extract the saved beam snapshots from ImpactX monitor OpenPMD files.

    This function reads the particle data stored by ImpactX in
    ``diags/openPMD/monitor.h5`` and reconstructs a list of ``Beam`` objects 
    representing the beam at each diagnostic step.

    Parameters
    ----------
    path : str, optional
        Path to the ImpactX simulation directory containing the OpenPMD output.
        Defaults to `''`.

    runnable : ``Runnable``, optional
        ABEL ``Runnable`` object used to locate the simulation directory 
        (``runnable.shot_path()/impactx_sims``). If provided, ``path`` is 
        ignored.

    beam0 : ``Beam``, optional
        Reference beam whose metadata (e.g., ``beam0.trackable_number`` and 
        ``beam0.stage_number``) are used to tag the extracted beams.

    Returns
    -------
    beams : list of ``Beam``
        List of reconstructed ``Beam`` objects, one for each saved monitor 
        snapshot.
    """
    
    from abel.classes.beam import Beam
    import openpmd_api as io

    if runnable is not None:
        path = os.path.join(runnable.shot_path(), 'impactx_sims') # TODO: even if path is provided, this will overwrite it.
        
    # load OpenPMD series
    series = io.Series(os.path.join(path,'diags/openPMD/monitor.h5'), io.Access.read_only)
    steps = list(series.iterations)
    
    ss = np.zeros_like(steps, dtype=np.float64)
    dispx2 = np.zeros_like(steps, dtype=np.float64)

    beams = []
    for i in range(len(steps)):
        
        beam_raw = series.iterations[steps[i]].particles["beam"]
        
        ss[i] = beam_raw.get_attribute("s_ref")
        pz_ref = beam_raw.get_attribute("pz_ref")
        
        beam_df = beam_raw.to_df()
        Q = np.sum(beam_df.qm*beam_df.weighting)*(SI.m_e*SI.c**2)
        
        xs = np.array(beam_df.position_x)
        ys = np.array(beam_df.position_y)
        zs = -np.array(beam_df.position_t)
        uxs = pz_ref*(np.array(beam_df.momentum_x))*SI.c
        uys = pz_ref*(np.array(beam_df.momentum_y))*SI.c
        uzs = pz_ref*(1-np.array(beam_df.momentum_t))*SI.c
        
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uxs=uxs, uys=uys, uzs=uzs)
        beam.location = ss[i]
        if beam0 is not None:
            beam.trackable_number = beam0.trackable_number+1
            beam.stage_number = beam0.stage_number+1

        beams.append(beam)

    return beams
        

# ==================================================
def extract_evolution(path=''):
    """
    Extract the beam evolution from the ImpactX reduced beam diagnostics.

    This function reads the reduced diagnostic files generated by ImpactX,
    typically stored in ``diags/reduced_beam_characteristics.*`` and 
    ``diags/ref_particle.*``. It compiles the evolution of beam parameters such as
    emittance, betatron functions, beam size, bunch length, dispersion, 
    and energy as a function of longitudinal position ``s``.
    
    The function automatically detects the ImpactX version and adapts
    to the diagnostic format changes introduced in ImpactX 25.10+.
    

    Parameters
    ----------
    path : str, optional
        Path to the directory containing the ImpactX diagnostic files.
        Should include the trailing slash, e.g. ``'impactx_sims/'``.
        Defaults to ``''`` (current working directory).


    Returns
    -------
    evol : :class:`types.SimpleNamespace`
        Data structure containing beam evolution parameters, including:

        - ``location`` : [m] numpy.ndarray — Longitudinal position.
        - ``emit_nx``, ``emit_ny`` : [m rad] numpy.ndarray — Normalized emittances.
        - ``beta_x``, ``beta_y`` : [m] numpy.ndarray — Beta functions.
        - ``beam_size_x``, ``beam_size_y`` : [m] numpy.ndarray — RMS beam sizes.
        - ``bunch_length`` : [m] numpy.ndarray — RMS bunch length.
        - ``x``, ``y``, ``z`` : [m] numpy.ndarray — Mean centroid positions.
        - ``energy`` : [eV] numpy.ndarray — Mean beam energy.
        - ``rel_energy_spread`` : numpy.ndarray — Relative RMS energy spread.
        - ``dispersion_x``, ``dispersion_y`` : [m] numpy.ndarray — Dispersion functions.
        - ``charge`` : [C] numpy.ndarray — Total beam charge.
    """

    """Extract the beam evolution from the reduced beam diagnostics."""
    
    from abel.utilities.relativity import gamma2energy
    import pandas as pd
    import impactx

    # read CSV file
    try:
        ref = pd.read_csv(path+"diags/ref_particle.0.0", delimiter=r"\s+")
        diags = pd.read_csv(path+"diags/reduced_beam_characteristics.0.0", delimiter=r"\s+")
    except:
        ref = pd.read_csv(path+"diags/ref_particle.0", delimiter=r"\s+")
        diags = pd.read_csv(path+"diags/reduced_beam_characteristics.0", delimiter=r"\s+")
    
    # breaking change in CSV files in ImpactX 25.10+
    impactx_major, impactx_minor = impactx.__version__.split(".")
    is_new_impactx = (
        float(impactx_major) == 25 and float(impactx_minor) >= 10
    ) or float(impactx_major) >= 26 

    # extract numbers
    evol = SimpleNamespace()
    evol.location = diags["s"]
    evol.emit_nx = diags["emittance_xn"]
    evol.emit_ny = diags["emittance_yn"]
    evol.beta_x = diags["beta_x"]
    evol.beta_y = diags["beta_y"]
    if is_new_impactx:
        evol.rel_energy_spread = diags["sigma_pt"]
        evol.beam_size_x = diags["sigma_x"]
        evol.beam_size_y = diags["sigma_y"]
        evol.bunch_length = diags["sigma_t"]
        evol.x = diags["mean_x"]
        evol.y = diags["mean_y"]
        evol.z = diags["mean_t"]
        evol.energy = gamma2energy(-ref["pt"]*(1-diags["mean_pt"]))#ref["pt"])#gamma2energy(diags["mean_pt"]-ref["pt"])
    else:
        evol.rel_energy_spread = diags["sig_pt"]
        evol.beam_size_x = diags["sig_x"]
        evol.beam_size_y = diags["sig_y"]
        evol.bunch_length = diags["sig_t"]
        evol.x = diags["x_mean"]
        evol.y = diags["y_mean"]
        evol.z = diags["t_mean"]
        evol.energy = gamma2energy(-ref["pt"]*(1-diags["pt_mean"]))#ref["pt"])#gamma2energy(diags["mean_pt"]-ref["pt"])
    evol.charge = diags["charge_C"]
    evol.dispersion_x = diags["dispersion_x"]
    evol.dispersion_y = diags["dispersion_y"]
    
    return evol


# ==================================================    
# convert from ImpactX particle container to ABEL beam
def particle_container2beam(particle_container):
    """
    Convert from ImpactX particle container to an ABEL beam object.

    Parameters
    ----------
    particle_container : ImpactX ``ParticleContainer``
        ImpactX ``ParticleContainer`` to be converted.

    Returns
    -------
    beam : ``Beam``
        ABEL ``Beam`` object.
    """
    
    from abel.classes.beam import Beam
    
    beam = Beam()
    beam.reset_phase_space(int(particle_container.total_number_of_particles()))
    
    array = particle_container.to_df().to_numpy()
    ref = particle_container.ref_particle()
    
    beam.set_xs(array[:,1])
    beam.set_ys(array[:,2])
    beam.set_zs(-array[:,3])
    
    beam.set_uxs(ref.pz*(array[:,4])*SI.c)
    beam.set_uys(ref.pz*(array[:,5])*SI.c)
    beam.set_uzs(ref.pz*(1-array[:,6])*SI.c)
    
    q_particle = array[0,7]*(SI.m_e*SI.c**2/SI.e)*SI.e
    beam.set_qs(array[:,8]*q_particle)
    
    beam.location = ref.s
    
    return beam


# ==================================================
# convert from ABEL beam to ImpactX particle container
def beam2particle_container(beam, nom_energy=None, sim=None, verbose=False):
    """
    Convert from an ABEL beam object to an ImpactX particle container.

    Parameters
    ----------
    beam : ``Beam``
        ABEL ``Beam`` object.

    nom_energy : [eV] float, optional
        Nominal kinetic energy of the reference particle. Used to set reference 
        particle kinetic energy. When ``None``, will use ``beam.energy()``. 
        Defaults to ``None``.

    sim : ImpactX simulation object, optional
        If ``None``, one is created via :func:`initialize_impactx_sim`. Defaults 
        to ``None``.

    verbose : bool, optional
        If ``True``, prints ImpactX progress and diagnostic information. 
        Defaults to ``False``.


    Returns
    -------
    particle_container : ImpactX ``ParticleContainer``
        ImpactX ``ParticleContainer`` object.

    sim : ImpactX simulation object 
    """

    import amrex.space3d as amr
    from impactx import Config
    import abel.wrappers.impactx.transformation_utilities as pycoord
    
    # make simulation object if not already existing
    if sim is None:
        sim = initialize_impactx_sim(verbose=verbose)

    # select beam energy as nominal if none given
    if nom_energy is None:
        nom_energy = beam.energy()
        
    # reference particle
    ref = sim.particle_container().ref_particle()
    ref.set_charge_qe(beam.charge_sign())
    ref.set_mass_MeV(0.510998950)
    ref.set_kin_energy_MeV(nom_energy/1e6)
    ref.s = beam.location
    
    dx, dy, dz, dpx, dpy, dpz = pycoord.to_ref_part_t_from_global_t(ref, beam.xs(), beam.ys(), beam.zs(), beam.uxs()/SI.c, beam.uys()/SI.c, beam.uzs()/SI.c)
    dx, dy, dt, dpx, dpy, dpt = pycoord.to_s_from_t(ref, dx, dy, dz, dpx, dpy, dpz)
    
    if not Config.have_gpu:  # initialize using cpu-based PODVectors
        dx_podv = amr.PODVector_real_std()
        dy_podv = amr.PODVector_real_std()
        dt_podv = amr.PODVector_real_std()
        dpx_podv = amr.PODVector_real_std()
        dpy_podv = amr.PODVector_real_std()
        dpt_podv = amr.PODVector_real_std()
    else:  # initialize on device using arena/gpu-based PODVectors
        dx_podv = amr.PODVector_real_arena()
        dy_podv = amr.PODVector_real_arena()
        dt_podv = amr.PODVector_real_arena()
        dpx_podv = amr.PODVector_real_arena()
        dpy_podv = amr.PODVector_real_arena()
        dpt_podv = amr.PODVector_real_arena()
    
    for p_dx in dx:
        dx_podv.push_back(p_dx)
    for p_dy in dy:
        dy_podv.push_back(p_dy)
    for p_dt in dt:
        dt_podv.push_back(p_dt)
    for p_dpx in dpx:
        dpx_podv.push_back(p_dpx)
    for p_dpy in dpy:
        dpy_podv.push_back(p_dpy)
    for p_dpt in dpt:
        dpt_podv.push_back(p_dpt)

    particle_container = sim.particle_container()
    qm_eev = -1.0 / 0.510998950e6  # electron charge/mass in e / eV
    particle_container.add_n_particles(dx_podv, dy_podv, dt_podv, dpx_podv, dpy_podv, dpt_podv, qm_eev, beam.abs_charge())

    return particle_container, sim

