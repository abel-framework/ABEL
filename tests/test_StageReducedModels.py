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
