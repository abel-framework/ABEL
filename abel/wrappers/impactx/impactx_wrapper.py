import numpy as np
from abel.CONFIG import CONFIG
import uuid, os, shutil
import scipy.constants as SI
from types import SimpleNamespace

def run_impactx(lattice, beam0, nom_energy=None, runnable=None, keep_data=False, space_charge=False, csr=False, isr=False, verbose=False):

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
        if nom_energy is not None and nom_energy > 1e12:
            sim.isr_order = 3
        else:
            sim.isr_order = 1
    
    # convert to ImpactX particle container
    _, sim = beam2particle_container(beam0, sim=sim, verbose=verbose)
    
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
    finalize_impactx_sim(sim, verbose=verbose)

    # extract evolution
    evol = extract_evolution();

    # copy meta data from input beam (will be iterated by super)
    beam.trackable_number = beam0.trackable_number
    beam.stage_number = beam0.stage_number
    beam.location = beam0.location

    # change back to original directory
    os.chdir(original_folder)

    # delete run folder
    if not keep_data:
        shutil.rmtree(runfolder)

    return beam, evol


def run_envelope_impactx(lattice, distr, nom_energy=None, peak_current=None, space_charge="2D", runnable=None, keep_data=False, verbose=False):

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
    ref.set_charge_qe(1.0).set_mass_MeV(SI.m_e*SI.c**2/SI.e/1e6).set_kin_energy_MeV(nom_energy/1e6)

    # initialize the envelope
    sim.init_envelope(ref, distr, peak_current)

    # assign the lattice
    sim.lattice.extend(lattice)
    
    # run simulation
    sim.verbose = int(verbose)
    if verbose:
        sim.track_envelope()
    else:
        eval('sim.track_envelope()')
    
    # clean shutdown
    finalize_impactx_sim(sim, verbose=verbose)

    # extract evolution
    evol = extract_evolution();
    
    # change back to original directory
    os.chdir(original_folder)

    # delete run folder
    if not keep_data:
        shutil.rmtree(runfolder)

    return evol


def initialize_impactx_sim(verbose=False):

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


def finalize_impactx_sim(sim, verbose=False):
    """finalize and delete the simulation"""

    sim.finalize()


def extract_evolution(path='', second_order=False):
    
    from abel.utilities.relativity import gamma2energy
    import pandas as pd

    # read CSV file
    try:
        ref = pd.read_csv(path+"diags/ref_particle.0.0", delimiter=r"\s+")
        diags = pd.read_csv(path+"diags/reduced_beam_characteristics.0.0", delimiter=r"\s+")
    except:
        ref = pd.read_csv(path+"diags/ref_particle.0", delimiter=r"\s+")
        diags = pd.read_csv(path+"diags/reduced_beam_characteristics.0", delimiter=r"\s+")
    
    # extract numbers
    evol = SimpleNamespace()
    evol.location = diags["s"]
    evol.emit_nx = diags["emittance_xn"]
    evol.emit_ny = diags["emittance_yn"]
    evol.beta_x = diags["beta_x"]
    evol.beta_y = diags["beta_y"]
    evol.rel_energy_spread = diags["sig_pt"]
    evol.beam_size_x = diags["sig_x"]
    evol.beam_size_y = diags["sig_y"]
    evol.bunch_length = diags["sig_t"]
    evol.x = diags["x_mean"]
    evol.y = diags["y_mean"]
    evol.z = diags["t_mean"]
    evol.energy = gamma2energy(-ref["pt"]*(1-diags["pt_mean"]))#ref["pt"])#gamma2energy(diags["pt_mean"]-ref["pt"])
    evol.charge = diags["charge_C"]
    evol.dispersion_x = diags["dispersion_x"]
    evol.dispersion_y = diags["dispersion_y"]

    if second_order:

        import openpmd_api as io
        
        # load OpenPMD series
        series = io.Series(path+"diags/openPMD/monitor.h5", io.Access.read_only)
        steps = list(series.iterations)
        
        ss = np.zeros_like(steps, dtype=np.float64)
        dispx2 = np.zeros_like(steps, dtype=np.float64)
        
        for i in range(len(steps)):
            
            beam_raw = series.iterations[steps[i]].particles["beam"]
            
            ss[i] = beam_raw.get_attribute("s_ref")
            
            beam = beam_raw.to_df()
            x = np.array(beam.position_x)
            delta = np.array(beam.momentum_t)

            # set the fit order based on the energy spread
            if np.std(delta) > 0.02:
                ordermax = 4
            else:
                ordermax = 2
            pfit = np.polyfit(delta, x, ordermax)
            dispx2[i] = pfit[ordermax-2]*np.math.factorial(2)
        
        evol.second_order_dispersion_x = np.interp(evol.location, ss, dispx2)
    else:
        evol.second_order_dispersion_x = np.empty_like(evol.location)
    
    return evol

    
# convert from ImpactX particle container to ABEL beam
def particle_container2beam(particle_container):

    from abel.classes.beam import Beam
    
    beam = Beam()
    beam.reset_phase_space(int(particle_container.total_number_of_particles()))
    
    array = particle_container.to_df().to_numpy()
    ref = particle_container.ref_particle()
    
    beam.set_xs(array[:,1])
    beam.set_ys(array[:,2])
    beam.set_zs(array[:,3])
    
    beam.set_uxs(ref.pz*(array[:,4])*SI.c)
    beam.set_uys(ref.pz*(array[:,5])*SI.c)
    beam.set_uzs(ref.pz*(1-array[:,6])*SI.c)
    
    q_particle = array[0,7]*(SI.m_e*SI.c**2/SI.e)*SI.e
    beam.set_qs(array[:,8]*q_particle)
    
    beam.location = ref.s
    
    return beam

    
# convert from ABEL beam to ImpactX particle container
def beam2particle_container(beam, sim=None, verbose=False):

    import amrex.space3d as amr
    from impactx import Config
    import abel.wrappers.impactx.transformation_utilities as pycoord
    
    # make simulation object if not already existing
    if sim is None:
        sim = initialize_impactx_sim(verbose=verbose)
    
    # reference particle
    ref = sim.particle_container().ref_particle()
    ref.set_charge_qe(-1.0)
    ref.set_mass_MeV(0.510998950)
    ref.set_kin_energy_MeV(beam.energy()/1e6)
    ref.s = beam.location
    
    dx, dy, dz, dpx, dpy, dpz = pycoord.to_ref_part_t_from_global_t(ref, beam.xs(), beam.ys(), -beam.zs(), beam.uxs()/SI.c, beam.uys()/SI.c, beam.uzs()/SI.c)
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

