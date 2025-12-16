# -*- coding: utf-8 -*-
# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

"""
ABEL : StageReducedModels unit tests
"""

import pytest
from abel import *
import shutil

def setup_trapezoid_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=0.0, y_angle=0.0):
    driver = SourceTrapezoid()
    driver.current_head = 0.1e3                                                   # [A]
    driver.bunch_length = 1150e-6                                                 # [m] This value is for trapezoid.
    driver.z_offset = 1724e-6                                                     # [m]
    driver.x_angle = x_angle                                                      # [rad]
    driver.y_angle = y_angle                                                      # [rad]

    driver.num_particles = 30000                                                 
    driver.charge = 5.0e10 * -SI.e                                                # [C]
    driver.energy = 4.9e9                                                         # [eV] 
    driver.gaussian_blur = 50e-6                                                  # [m]
    driver.rel_energy_spread = 0.01                                              

    driver.emit_nx, driver.emit_ny = 50e-6, 100e-6                                # [m rad]
    driver.beta_x, driver.beta_y = 0.5, 0.5                                       # [m]

    if enable_xy_jitter:
        driver.jitter.x = 100e-9                                                  # [m], std
        driver.jitter.y = 100e-9                                                  # [m], std

    if enable_xpyp_jitter:
        driver.jitter.xp = 1.0e-6                                                 # [rad], std
        driver.jitter.yp = 1.0e-6                                                 # [rad], std

    driver.symmetrize = True

    return driver


def setup_basic_main_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3.0e9):
    from abel.utilities.plasma_physics import beta_matched

    main = SourceBasic()
    main.bunch_length = 40.0e-06                                                  # [m], rms. Standard value
    main.num_particles = 5000                                               
    main.charge = -SI.e * 1.0e10                                                  # [C]

    # Energy parameters
    main.energy = energy                                                          # [eV], HALHF v2 first stage nominal input energy as default.
    main.rel_energy_spread = 0.02                                                 # Relative rms energy spread

    # Emittances
    #main.emit_nx, main.emit_ny = 90.0e-6, 0.32e-6                                 # [m rad], budget value
    main.emit_nx, main.emit_ny = 15e-6, 0.1e-6                                    # [m rad]

    # Beta functions
    main.beta_x = beta_matched(plasma_density, main.energy) * ramp_beta_mag       # [m]
    main.beta_y = main.beta_x                                                     # [m]

    # Offsets
    main.z_offset = 0.00e-6                                                       # [m] # Standard value

    # Other
    main.symmetrize_6d = True

    return main


def setup_StageReducedModels(driver_source, main_source, plasma_density=6.0e20, ramp_beta_mag=5.0, enable_tr_instability=True, enable_radiation_reaction=True, enable_ion_motion=False, use_ramps=False, drive_beam_update_period=0, return_tracked_driver=False, store_beams_for_tests=False, length_flattop=7.8):

    stage = StageReducedModels()
    stage.time_step_mod = 0.03                                                    # In units of betatron wavelengths/c.
    stage.length_flattop = length_flattop                                         # [m]
    if length_flattop is not None:
        stage.nom_energy_gain = length_flattop*1e9                                # [eV]
    stage.plasma_density = plasma_density                                         # [m^-3]
    stage.driver_source = driver_source
    stage.main_source = main_source
    stage.ramp_beta_mag = ramp_beta_mag
    stage.enable_tr_instability = enable_tr_instability 
    stage.enable_radiation_reaction = enable_radiation_reaction

    stage.enable_ion_motion = enable_ion_motion
    stage.ion_charge_num = 1.0
    ion_mass = 4.002602 * SI.physical_constants['atomic mass constant'][0]        # [kg], He mass
    stage.ion_mass = ion_mass
    stage.num_z_cells_main = 51
    stage.num_y_cells_rft = 50
    stage.num_x_cells_rft = 50
    stage.num_xy_cells_probe = 41
    stage.ion_wkfld_update_period = 1  # Updates the ion wakefield perturbation every nth time step.
    stage.drive_beam_update_period = drive_beam_update_period  # Updates the drive beam every nth time step.

    stage.probe_evol_period = 3
    stage.make_animations = False
    stage.save_final_step = False

    stage._return_tracked_driver = return_tracked_driver
    stage.store_beams_for_tests = store_beams_for_tests

    # Set up ramps after the stage is fully configured
    if use_ramps:
        stage.upramp = PlasmaRamp()
        stage.downramp = PlasmaRamp()
    
    return stage


@pytest.mark.StageReducedModels
def test_stage_length_gradient_energyGain():
    """
    Tests ensuring that the flattop length and total length of the stage as well 
    as nominal gradient and nominal energy gain are set correctly.
    """

    np.random.seed(42)

    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = False
    enable_xpyp_jitter = False
    enable_tr_instability = False
    enable_radiation_reaction = False
    enable_ion_motion = False
    use_ramps = True
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)


    # ========== Set stage length and nominal energy gain ==========
    stage = setup_StageReducedModels(driver_source, main_source, plasma_density, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period, return_tracked_driver=False, store_beams_for_tests=False)

    stage.length_flattop = 7.8                                                    # [m]
    stage.nom_energy_gain = 7.8e9                                                 # [eV]

    linac = PlasmaLinac(source=main_source, stage=stage, num_stages=1)
    linac.run('test_stage_length_gradient_energyGain', overwrite=True, verbose=False)

    assert np.allclose(stage.nom_energy_gain_flattop, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length_flattop, 7.8, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient_flattop, 1.0e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length, stage.length_flattop + stage.upramp.length_flattop + stage.downramp.length_flattop, rtol=1e-15, atol=0.0)


    # ========== Set nominal energy gain and flattop nominal acceleration gradient ==========
    stage = setup_StageReducedModels(driver_source, main_source, plasma_density, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period, return_tracked_driver=False, store_beams_for_tests=False, length_flattop=None)

    stage.nom_energy_gain = 7.8e9                                                 # [eV]
    stage.nom_energy_gain_flattop = 7.8e9                                         # [eV]
    stage.nom_accel_gradient_flattop = 1.0e9                                      # [V/m]

    linac = PlasmaLinac(source=main_source, stage=stage, num_stages=1)
    return
    linac.run('test_stage_length_gradient_energyGain', overwrite=True, verbose=False)
    assert np.allclose(stage.nom_energy_gain_flattop, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length_flattop, 7.8, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient_flattop, 1.0e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length, stage.length_flattop + stage.upramp.length_flattop + stage.downramp.length_flattop, rtol=1e-15, atol=0.0)

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageReducedModels
def test_driver_unrotation():
    """
    Tests for checking the driver being correctly un-rotated back to its 
    original coordinate system.
    """

    np.random.seed(42)

    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = True
    enable_xpyp_jitter = True
    enable_tr_instability = False
    enable_radiation_reaction = False
    enable_ion_motion = False
    use_ramps = True
    drive_beam_update_period = 0
    return_tracked_driver = True

    # ========== Driver jitter, no angular offset ==========
    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StageReducedModels(driver_source, main_source, plasma_density, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period, return_tracked_driver, store_beams_for_tests=True)

    stage.nom_energy = 369.6e9                                                    # [eV], HALHF v2 last stage nominal input energy
    _, driver = stage.track(main_source.track())
    driver0 = stage.driver_incoming

    x_drift = stage.length * np.tan(driver0.x_angle())
    y_drift = stage.length * np.tan(driver0.y_angle())
    xs = driver0.xs()
    ys = driver0.ys()
    driver0.set_xs(xs + x_drift)
    driver0.set_ys(ys + y_drift)

    assert np.allclose(driver.x_angle(), driver0.x_angle(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.y_angle(), driver0.y_angle(), rtol=1e-11, atol=0.0)

    assert np.allclose(driver.qs(), driver0.qs(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.weightings(), driver0.weightings(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.xs(), driver0.xs(), rtol=0.0, atol=1e-7)
    assert np.allclose(driver.ys(), driver0.ys(), rtol=0.0, atol=1e-7)
    assert np.allclose(driver.zs(), driver0.zs(), rtol=0.0, atol=1e-7)
    assert np.allclose(driver.uxs(), driver0.uxs(), rtol=1e-12, atol=0.0)
    assert np.allclose(driver.uys(), driver0.uys(), rtol=1e-12, atol=0.0)
    assert np.allclose(driver.uzs(), driver0.uzs(), rtol=1e-12, atol=0.0)
    assert np.allclose(driver.particle_mass, driver0.particle_mass, rtol=1e-13, atol=0.0)


    # ========== No jitter, no angular offset ==========
    driver_source = setup_trapezoid_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StageReducedModels(driver_source, main_source, use_ramps=True, return_tracked_driver=True, store_beams_for_tests=True)

    stage.nom_energy = 7.8e9                                                      # [eV]
    _, driver = stage.track(main_source.track())
    driver0 = stage.driver_incoming

    Beam.comp_beams(driver, driver0, comp_location=False, rtol=1e-13, atol=0.0)


    # ========== No jitter, large angular offset ==========
    x_angle = 5e-6                                                                # [rad]
    y_angle = 2e-5                                                                # [rad]
    driver_source = setup_trapezoid_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=x_angle, y_angle=y_angle)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StageReducedModels(driver_source, main_source, use_ramps=True, return_tracked_driver=True, store_beams_for_tests=True)

    stage.nom_energy = 7.8e9                                                      # [eV]
    _, driver = stage.track(main_source.track())
    driver0 = stage.driver_incoming

    x_drift = stage.length * np.tan(x_angle)
    y_drift = stage.length * np.tan(y_angle)
    xs = driver0.xs()
    ys = driver0.ys()
    driver0.set_xs(xs + x_drift)
    driver0.set_ys(ys + y_drift)

    assert np.allclose(driver.x_angle(), driver0.x_angle(), rtol=1e-10, atol=0.0)
    assert np.allclose(driver.y_angle(), driver0.y_angle(), rtol=1e-13, atol=0.0)

    assert np.allclose(driver.qs(), driver0.qs(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.weightings(), driver0.weightings(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.xs(), driver0.xs(), rtol=0.0, atol=1e-7)
    assert np.allclose(driver.ys(), driver0.ys(), rtol=0.0, atol=1e-7)
    assert np.allclose(driver.zs(), driver0.zs(), rtol=0.0, atol=1e-7)
    assert np.allclose(driver.uxs(), driver0.uxs(), rtol=1e-10, atol=0.0)
    assert np.allclose(driver.uys(), driver0.uys(), rtol=1e-10, atol=0.0)
    assert np.allclose(driver.uzs(), driver0.uzs(), rtol=1e-12, atol=0.0)
    assert np.allclose(driver.particle_mass, driver0.particle_mass, rtol=1e-13, atol=0.0)

    
@pytest.mark.StageReducedModels
def test_copy_config2blank_stage():
    """
    Tests for ``StageReducedModels.copy_config2blank_stage()``.
    """

    np.random.seed(42)

    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = True
    enable_xpyp_jitter = True
    enable_tr_instability = True
    enable_radiation_reaction = True
    enable_ion_motion = True
    use_ramps = True
    drive_beam_update_period = 0
    return_tracked_driver = False

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StageReducedModels(driver_source, main_source, plasma_density, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period, return_tracked_driver, store_beams_for_tests=True)

    stage_copy = stage.copy_config2blank_stage()

    assert stage_copy.plasma_density is None
    assert stage_copy.ramp_beta_mag is None
    assert stage_copy.length is None
    assert stage_copy.length_flattop is None
    assert stage_copy.nom_energy_gain is None
    assert stage_copy.nom_energy_gain_flattop is None
    assert stage_copy.nom_energy is None
    assert stage_copy.nom_energy_flattop is None
    assert stage_copy.nom_accel_gradient is None
    assert stage_copy.nom_accel_gradient_flattop is None

    assert stage_copy._length_calc is None
    assert stage_copy._length_flattop_calc is None
    assert stage_copy._nom_energy_gain_calc is None
    assert stage_copy._nom_energy_gain_flattop_calc is None
    assert stage_copy._nom_energy_calc is None
    assert stage_copy._nom_energy_flattop_calc is None
    assert stage_copy._nom_accel_gradient_calc is None
    assert stage_copy._nom_accel_gradient_flattop_calc is None

    assert stage_copy.driver_source is None
    assert stage_copy.upramp is None
    assert stage_copy.downramp is None
    assert stage_copy.has_ramp() is False
    assert stage_copy.is_upramp() is False
    assert stage_copy.is_downramp() is False

    assert stage_copy.enable_tr_instability is True
    assert stage_copy.enable_radiation_reaction is True
    assert stage_copy.enable_ion_motion is True
    assert stage_copy.drive_beam_update_period == 0
    assert stage_copy._return_tracked_driver is False
    assert stage_copy.probe_evol_period == 1
    assert stage_copy.make_animations is False  # Currently does not support animations in ramps, as they get overwritten.
    assert stage_copy.show_prog_bar is False


@pytest.mark.StageReducedModels
def test_rb_Ez_tracing():
    """
    Tests for checking correct tracing of bubble radius and axial electric field 
    by ``StageReducedModels`` methods by comparing against the data in  
    reference files.
    """

    import os, copy, uuid
    from abel.wrappers.wake_t.wake_t_wrapper import beam2wake_t_bunch, plasma_stage_setup, extract_initial_and_final_Ez_rho
    from abel.utilities.plasma_physics import blowout_radius

    np.random.seed(42)

    # ========== Load data from the reference file ==========
    file_path = '.' + os.sep + 'tests' + os.sep + 'data' + os.sep + 'test_StageReducedModels' + os.sep + 'test_rb_Ez_tracing' + os.sep + 'bubble_radius_axial_Ez.npz'

    data = np.load(file_path)
    arrays = data["array"]
    zs_ref = arrays[:,0]
    rb_ref = arrays[:,1]
    zs_Ez_ref = arrays[:,2]
    Ez_ref = arrays[:,3]
    box_size_r = data["box_size_r"]
    num_cell_xy = data["num_cell_xy"]  # Transverse resolution for the Wake-T simulation. Same resolution in the longitudinal direction.


    # ========== Set up a Wake-T simulation ==========
    driver_source = setup_trapezoid_driver_source()
    main_source = setup_basic_main_source()
    ramp_beta_mag = 5.0
    plasma_density = 6.0e20
    stage = setup_StageReducedModels(driver_source, main_source, length_flattop=0.01)

    drive_beam = driver_source.track()
    beam = main_source.track()
    beam.magnify_beta_function(1/ramp_beta_mag, axis_defining_beam=drive_beam)
    drive_beam_ramped = copy.deepcopy(drive_beam)
    drive_beam_ramped.magnify_beta_function(1/ramp_beta_mag, axis_defining_beam=drive_beam)

    plasma_stage = plasma_stage_setup(plasma_density, drive_beam_ramped, beam, stage_length=None, dz_fields=None, num_cell_xy=int(num_cell_xy), n_out=1, box_size_r=box_size_r, box_min_z=zs_ref.min(), box_max_z=zs_ref.max())

    # Make temp folder
    if not os.path.exists(CONFIG.temp_path):
        os.mkdir(CONFIG.temp_path)
    tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
    if not os.path.exists(tmpfolder):
        os.mkdir(tmpfolder)

    # Convert beams to Wake-T bunches
    driver0_wake_t = beam2wake_t_bunch(drive_beam_ramped, name='driver')
    beam0_wake_t = beam2wake_t_bunch(beam, name='beam')

    plasma_stage.track([driver0_wake_t, beam0_wake_t], opmd_diag=True, diag_dir=tmpfolder, show_progress_bar=False)
    
    wake_t_evolution = extract_initial_and_final_Ez_rho(tmpfolder)
    
    # remove temporary directory
    shutil.rmtree(tmpfolder)

    # Read the Wake-T simulation data
    Ez_axis_wakeT = wake_t_evolution.initial.plasma.wakefield.onaxis.Ezs
    zs_Ez_wakeT = wake_t_evolution.initial.plasma.wakefield.onaxis.zs
    plasma_num_density = wake_t_evolution.initial.plasma.density.rho/plasma_density
    info_rho = wake_t_evolution.initial.plasma.density.metadata
    zs_rho = info_rho.z
    rs_rho = info_rho.r


    # ========== Compare the data along the whole simulation box ==========
    rb = stage.trace_bubble_radius_WakeT(plasma_num_density=plasma_num_density, plasma_tr_coord=rs_rho, plasma_z_coord=zs_rho, drive_beam_peak_current=drive_beam_ramped.peak_current(), threshold=0.8)

    assert np.allclose(zs_rho, zs_ref, rtol=1e-13, atol=0.0)
    RMSE_rb = np.sqrt(np.mean((rb-rb_ref)**2))
    assert RMSE_rb < 5e-6

    assert np.allclose(zs_Ez_wakeT, zs_Ez_ref, rtol=1e-13, atol=0.0)
    RMSE_Ez = np.sqrt(np.mean((Ez_axis_wakeT-Ez_ref)**2))
    assert RMSE_Ez < 0.2e9


    # ========== Compare the data only in the region of interest ==========
    file_path = '.' + os.sep + 'tests' + os.sep + 'data' + os.sep + 'test_StageReducedModels' + os.sep + 'test_rb_Ez_tracing' + os.sep + 'bubble_radius_axial_Ez_roi.npz'

    data_roi = np.load(file_path)
    arrays_roi = data_roi["array"]
    zs_ref_roi = arrays_roi[:,0]
    rb_ref_roi = arrays_roi[:,1]
    Ez_ref_roi = arrays_roi[:,2]

    # Cut out axial Ez over the ROI
    Ez_roi, Ez_fit = stage.Ez_shift_fit(Ez_axis_wakeT, zs_Ez_wakeT, beam, zs_ref_roi)

    # Cut out bubble radius over the ROI
    R_blowout = blowout_radius(plasma_density, drive_beam_ramped.peak_current())
    stage.estm_R_blowout = R_blowout
    rb_roi, rb_fit = stage.rb_shift_fit(rb, zs_rho, beam, zs_ref_roi)
    print(rb_roi[1:10])
    print(rb_ref_roi[1:10])

    assert np.allclose(Ez_roi, Ez_ref_roi, rtol=0.0, atol=0.1e9)
    assert np.allclose(rb_roi, rb_ref_roi, rtol=0.0, atol=20e-6)
    

@pytest.mark.StageReducedModels
def test_longitudinal_number_distribution():
    """
    Tests for ``StageReducedModels.longitudinal_number_distribution()``.
    """

    np.random.seed(42)

    driver_source = setup_trapezoid_driver_source()
    main_source = setup_basic_main_source()
    stage = setup_StageReducedModels(driver_source, main_source)

    beam = main_source.track()
    main_num_profile, z_slices = stage.longitudinal_number_distribution(beam=beam)
    drive_beam = driver_source.track()
    drive_beam_num_profile, driver_z_slices = stage.longitudinal_number_distribution(beam=drive_beam)

    # Calculate values for comparison
    zs = beam.zs()
    weights = beam.weightings()  # The weight of each macroparticle.
    bin_number = stage.FD_rule_num_slice(zs)
    num_profile, edges = np.histogram(zs, weights=weights, bins=bin_number)  # Compute the histogram of z using bin_number bins.
    z_ctrs = (edges[0:-1] + edges[1:])/2  # Centres of the bins (zs).
    driver_zs = drive_beam.zs()
    driver_weights = drive_beam.weightings()  # The weight of each macroparticle.
    driver_bin_number = stage.FD_rule_num_slice(driver_zs)
    driver_num_profile, driver_edges = np.histogram(driver_zs, weights=driver_weights, bins=driver_bin_number)  # Compute the histogram of z using bin_number bins.
    driver_z_ctrs = (driver_edges[0:-1] + driver_edges[1:])/2  # Centres of the bins (zs).

    assert np.allclose(main_num_profile, num_profile, rtol=1e-15, atol=0.0)
    assert np.allclose(z_slices, z_ctrs, rtol=1e-15, atol=0.0)
    assert np.allclose(drive_beam_num_profile, driver_num_profile, rtol=1e-15, atol=0.0)
    assert np.allclose(driver_z_slices, driver_z_ctrs, rtol=1e-15, atol=0.0)


@pytest.mark.StageReducedModels
def test_matched_beta_function():
    """
    Tests for ``Stage.matched_beta_function()``.
    """
    #TODO: move to tests for the Stage class

    from abel.utilities.relativity import energy2gamma

    np.random.seed(42)

    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0

    driver_source = setup_trapezoid_driver_source()
    main_source = setup_basic_main_source()
    stage = setup_StageReducedModels(driver_source, main_source, plasma_density, ramp_beta_mag)

    kp = np.sqrt(plasma_density*SI.e**2/(SI.epsilon_0*SI.m_e*SI.c**2))  # plasma wavenumber [m^-1]
    beta_mat = np.sqrt(2*energy2gamma(main_source.energy))/kp
    assert np.allclose(stage.matched_beta_function(main_source.energy), beta_mat, rtol=1e-15, atol=0.0)

    stage_upramp = setup_StageReducedModels(driver_source, main_source, plasma_density, ramp_beta_mag)
    stage_upramp.upramp = PlasmaRamp()
    beta_mat_upramp = np.sqrt(2*energy2gamma(main_source.energy))/kp * ramp_beta_mag
    assert np.allclose(stage_upramp.matched_beta_function(main_source.energy), beta_mat_upramp, rtol=1e-15, atol=0.0)

    stage_ramped = setup_StageReducedModels(driver_source, main_source, plasma_density, ramp_beta_mag)
    stage_ramped.upramp = PlasmaRamp()
    stage_ramped.downramp = PlasmaRamp()
    stage_ramped.downramp.ramp_beta_mag = 4.0
    beta_mat_ramped = np.sqrt(2*energy2gamma(main_source.energy))/kp * 4.0
    assert np.allclose(stage_ramped.matched_beta_function(main_source.energy, match_entrance=False), beta_mat_ramped, rtol=1e-15, atol=0.0)


@pytest.mark.StageReducedModels
def test_trim_attr_reduce_pickle_size():
    """
    Tests for ``StageReducedModels.trim_attr_reduce_pickle_size()``.
    """

    np.random.seed(42)

    driver_source = setup_trapezoid_driver_source()
    main_source = setup_basic_main_source()
    stage = setup_StageReducedModels(driver_source, main_source, use_ramps=True)

    # Before calling StageReducedModels.trim_attr_reduce_pickle_size()
    assert stage.upramp is not None
    assert stage.upramp.initial is not None
    assert stage.upramp.final is not None
    assert stage.downramp is not None
    assert stage.downramp.initial is not None
    assert stage.downramp.final is not None

    stage.trim_attr_reduce_pickle_size()

    # After calling StageReducedModels.trim_attr_reduce_pickle_size()
    assert stage.upramp.drive_beam is None
    assert stage.upramp.initial is None
    assert stage.upramp.final is None
    assert stage.downramp.drive_beam is None
    assert stage.downramp.initial is None
    assert stage.downramp.final is None


@pytest.mark.StageReducedModels
def test_plotting_methods():
    """
    Tests for various ``StageReducedModels`` plotting methods.
    """

    import matplotlib.pyplot as plt
    np.random.seed(42)

    driver_source = setup_trapezoid_driver_source()
    main_source = setup_basic_main_source()
    stage = setup_StageReducedModels(driver_source, main_source, use_ramps=True, length_flattop=1.0)
    stage.save_final_step = True

    linac = PlasmaLinac(source=main_source, stage=stage, num_stages=1)
    linac.run('test_plotting_methods', overwrite=True, verbose=False)
    beam = linac.get_beam(-1)

    linac.stages[-1].plot_wakefield(saveToFile=None, includeWakeRadius=False)
    linac.stages[-1].plot_wakefield(saveToFile=None, includeWakeRadius=True)
    linac.stages[-1].plot_wake(show_Ez=False, trace_rb=False, savefig=None, aspect='auto')
    linac.stages[-1].plot_wake(show_Ez=True, trace_rb=True, savefig=None, aspect='auto')
    linac.stages[-1].plot_Ez_rb_cut()
    linac.stages[-1].plot_flattop_evolution()

    # Close all plots
    plt.close('all')

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageReducedModels
def test_print_stage_beam_summary():
    """
    Tests for ``StageReducedModels.print_stage_beam_summary()``.
    """

    np.random.seed(42)

    driver_source = setup_trapezoid_driver_source()
    main_source = setup_basic_main_source()
    stage = setup_StageReducedModels(driver_source, main_source, use_ramps=True)

    stage.print_stage_beam_summary(initial_main_beam=main_source.track(), beam_out=main_source.track(), clean=True)


@pytest.mark.StageReducedModels
def test_print_summary():
    """
    Tests for ``StageReducedModels.print_summary()``.
    """

    np.random.seed(42)

    driver_source = setup_trapezoid_driver_source()
    main_source = setup_basic_main_source()
    stage = setup_StageReducedModels(driver_source, main_source, use_ramps=True)

    stage.print_summary()


@pytest.mark.StageReducedModels
def test_make_animations():
    """
    Tests for ``StageReducedModels`` functions for making animations.
    """
    from pathlib import Path
    import os

    np.random.seed(42)

    driver_source = setup_trapezoid_driver_source()
    main_source = setup_basic_main_source(energy=300.0e9)  # High energy for faster tracking.
    stage = setup_StageReducedModels(driver_source, main_source, use_ramps=True)
    stage.time_step_mod = 0.03                                                    # In units of betatron wavelengths/c.
    stage.probe_evol_period = 18
    stage.make_animations = True
    stage.run_path = Path("tests/run_data")
    os.makedirs(stage.run_path, exist_ok=True)

    stage.nom_energy = main_source.energy                                         # [eV]
    stage.track(main_source.track())

    shutil.rmtree( os.path.join(stage.run_path, 'plots') )
