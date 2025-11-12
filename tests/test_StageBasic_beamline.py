# -*- coding: utf-8 -*-
# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

"""
ABEL : StageBasic beamline integration tests
"""

import pytest
from abel import *
import shutil
import numpy as np


def setup_basic_driver_source(enable_xt_jitter=False, enable_xpyp_jitter=False, enable_norm_emittance_jitter=False, x_angle=0.0, y_angle=0.0):
    driver = SourceBasic()
    driver.bunch_length = 42e-6                                                     # [m] This value is for trapezoid.
    driver.z_offset = 300e-6                                                        # [m]
    driver.x_angle = x_angle                                                        # [rad]
    driver.y_angle = y_angle                                                        # [rad]

    driver.num_particles = 100000                                                 
    driver.charge = -2.7e10 * SI.e                                                  # [C]
    driver.energy = 31.25e9                                                         # [eV]
    driver.rel_energy_spread = 0.01                                                 # Relative rms energy spread

    driver.emit_nx, driver.emit_ny = 80e-6, 80e-6                                   # [m rad]
    driver.beta_x, driver.beta_y = 0.2, 0.2                                         # [m]

    if enable_xt_jitter:
        driver.jitter.x = 1.0e-6                                                    # [m], std
        driver.jitter.y = 10e-15                                                    # [m], std

    if enable_xpyp_jitter:
        driver.jitter.xp = 1.0e-6                                                   # [rad], std
        driver.jitter.yp = 1.0e-6                                                   # [rad], std

    if enable_norm_emittance_jitter:
        driver.norm_jitter_emittance_x = 1e-12                                      # [m rad]
        driver.norm_jitter_emittance_y = 1e-12                                      # [m rad]

    driver.symmetrize = True

    return driver


def setup_basic_main_source(plasma_density=7.0e21, ramp_beta_mag=1.0):
    from abel.utilities.plasma_physics import beta_matched

    main = SourceBasic()
    main.bunch_length = 18e-6                                                       # [m], rms. Standard value
    main.num_particles = 10000                                               
    main.charge = -SI.e * 1.0e10                                                    # [C]

    # Energy parameters
    main.energy = 5.0e9                                                             # [eV]
    main.rel_energy_spread = 0.01                                                   # Relative rms energy spread

    # Emittances
    main.emit_nx, main.emit_ny = 160e-6, 0.56e-6                                    # [m rad]

    # Beta functions
    main.beta_x = beta_matched(plasma_density, main.energy) * ramp_beta_mag         # [m]
    main.beta_y = main.beta_x                                                       # [m]

    # Offsets
    main.z_offset = -36e-6                                                          # [m] # Standard value

    # Other
    main.symmetrize = True

    return main


def setup_StageBasic(driver_source=None, nom_accel_gradient=6.4e9, nom_energy_gain=31.9e9, plasma_density=7.0e21, use_ramps=False, transformer_ratio=1, depletion_efficiency=0.75, probe_evolution=False, return_tracked_driver=False, store_beams_for_tests=False):
    
    stage = StageBasic()
    stage.nom_accel_gradient = nom_accel_gradient                                   # [GV/m]
    stage.nom_energy_gain = nom_energy_gain                                         # [eV]
    stage.plasma_density = plasma_density                                           # [m^-3]
    stage.driver_source = driver_source
    if use_ramps:
        stage.ramp_beta_mag = 10.0
    else:
        stage.ramp_beta_mag = 1.0
    stage.store_beams_for_tests = store_beams_for_tests
    stage.transformer_ratio = transformer_ratio
    stage.depletion_efficiency = depletion_efficiency
    stage.probe_evolution = probe_evolution
    stage._return_tracked_driver = return_tracked_driver

    # Set up ramps after the stage is fully configured
    if use_ramps:
        stage.upramp = PlasmaRamp() 
        stage.downramp = PlasmaRamp()
    
    return stage


def setup_interstage(stage):
    interstage = InterstagePlasmaLensBasic()
    interstage.beta0 = lambda E: stage.matched_beta_function(E)
    interstage.length_dipole = lambda E: 1 * np.sqrt(E/10e9)                        # [m(eV)]
    interstage.field_dipole = lambda E: np.min([1.0, 100e9/E])                      # [T]

    return interstage


@pytest.mark.StageBasic_linac
def test_baseline_linac():
    """
    Tests a linac with ``StageBasic`` plasma stages. No driver jitter, no ramps. 
    Checks whether various linac attributes are correctly set and compares some 
    output beam parameters against expected values.
    """

    np.random.seed(42)

    num_stages = 5
    enable_xt_jitter = False
    enable_xpyp_jitter = False
    
    driver_source = setup_basic_driver_source(enable_xt_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(ramp_beta_mag=1.0)
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=False, probe_evolution=False)
    interstage = setup_interstage(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages, alternate_interstage_polarity=False)

    # Perform tracking
    linac.run('test_baseline_linac', overwrite=True, verbose=False)
    
    # Check the outputs
    stages = linac.stages
    interstages = linac.interstages
    assert len(stages) == num_stages
    assert len(interstages) == num_stages - 1
    assert np.isclose(linac.get_length(), 93.0738788160906, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].nom_energy, 5.0e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[1].nom_energy, 36.9e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[2].nom_energy, 68.8e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[3].nom_energy, 100.7e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[4].nom_energy, 132.6e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[4].nom_energy_gain, 31.9e9, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].nom_energy, stages[1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[2].nom_energy, stages[3].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].nom_energy, stages[4].nom_energy, rtol=1e-15, atol=0.0)
    interstage_nom_energy = interstages[0].nom_energy
    assert np.isclose(interstages[0].beta0, stage.matched_beta_function(interstage_nom_energy), rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].length_dipole, np.sqrt(interstage_nom_energy/10e9), rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].field_dipole, np.min([1.0, 100e9/interstage_nom_energy]), rtol=1e-15, atol=0.0)
    interstage_nom_energy = interstages[3].nom_energy
    assert np.isclose(interstages[3].beta0, stage.matched_beta_function(interstage_nom_energy), rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].length_dipole, np.sqrt(interstage_nom_energy/10e9), rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].field_dipole, np.min([1.0, 100e9/interstage_nom_energy]), rtol=1e-15, atol=0.0)

    initial_beam = linac.get_beam(0)
    assert np.isclose(initial_beam.energy(), stages[0].nom_energy, rtol=1e-3, atol=0.0)
    assert np.isclose(initial_beam.beta_x(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)
    assert np.isclose(initial_beam.beta_y(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)

    final_beam = linac.get_beam(-1)
    final_beam.beam_name = 'Test beam'
    assert final_beam.stage_number == 5
    assert np.isclose(final_beam.location, linac.get_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.energy(), stages[4].nom_energy + stages[4].nom_energy_gain, rtol=1e-4, atol=0.0)
    assert np.isclose(final_beam.bunch_length(), main_source.bunch_length, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.charge(), main_source.charge, rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.rel_energy_spread(), 0.00030271, rtol=1e-1, atol=0.0)

    nom_beam_size_x = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_x()
    nom_beam_size_y = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_y()
    assert np.isclose(final_beam.beam_size_x(), nom_beam_size_x, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.beam_size_y(), nom_beam_size_y, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_x(), main_source.emit_nx, rtol=1e-2, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_y(), main_source.emit_ny, rtol=1e-1, atol=0.0)

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageBasic_linac
def test_linac_plots():
    """
    Tests a linac with ``StageBasic`` plasma stages. No driver jitter, with 
    ramps. ``StageBasic.probe_evolution=True``.

    Checks that plotting methods can be executed.
    """

    np.random.seed(42)
    from matplotlib import pyplot as plt

    num_stages = 5
    
    driver_source = setup_basic_driver_source(enable_xt_jitter=False, enable_xpyp_jitter=False)
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, probe_evolution=False)
    stage.probe_evolution = True
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)
    interstage = setup_interstage(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages, alternate_interstage_polarity=False)

    # Perform tracking
    linac.run('test_linac_plots', overwrite=True, verbose=False)

    # Check the outputs
    stages = linac.stages
    interstages = linac.interstages
    assert len(stages) == num_stages
    assert len(interstages) == num_stages - 1
    assert np.isclose(linac.get_length(), 93.0738788160906, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].nom_energy, 5.0e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[1].nom_energy, 36.9e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[2].nom_energy, 68.8e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[3].nom_energy, 100.7e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[4].nom_energy, 132.6e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[4].nom_energy_gain, 31.9e9, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].nom_energy, stages[1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[2].nom_energy, stages[3].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].nom_energy, stages[4].nom_energy, rtol=1e-15, atol=0.0)
    interstage_nom_energy = interstages[0].nom_energy
    assert np.isclose(interstages[0].beta0, stage.matched_beta_function(interstage_nom_energy), rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].length_dipole, np.sqrt(interstage_nom_energy/10e9), rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].field_dipole, np.min([1.0, 100e9/interstage_nom_energy]), rtol=1e-15, atol=0.0)
    interstage_nom_energy = interstages[3].nom_energy
    assert np.isclose(interstages[3].beta0, stage.matched_beta_function(interstage_nom_energy), rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].length_dipole, np.sqrt(interstage_nom_energy/10e9), rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].field_dipole, np.min([1.0, 100e9/interstage_nom_energy]), rtol=1e-15, atol=0.0)
    # assert np.isclose(interstages[0].beta0, 0.24137835827389, rtol=1e-15, atol=0.0)
    # assert np.isclose(interstages[0].length_dipole, 1.9209372712298547, rtol=1e-15, atol=0.0)
    # assert np.isclose(interstages[0].field_dipole, 1.0, rtol=1e-15, atol=0.0)
    # assert np.isclose(interstages[3].beta0, 0.45756933131960525, rtol=1e-15, atol=0.0)
    # assert np.isclose(interstages[3].length_dipole, 3.641428291206625, rtol=1e-15, atol=0.0)
    # assert np.isclose(interstages[3].field_dipole, 0.7541478129713424, rtol=1e-15, atol=0.0)

    initial_beam = linac.get_beam(0)
    assert np.isclose(initial_beam.energy(), stages[0].nom_energy, rtol=1e-3, atol=0.0)
    assert np.isclose(initial_beam.beta_x(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)
    assert np.isclose(initial_beam.beta_y(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)

    final_beam = linac.get_beam(-1)
    final_beam.beam_name = 'Test beam'
    assert final_beam.stage_number == 5
    assert np.isclose(final_beam.location, linac.get_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.energy(), stages[4].nom_energy + stages[4].nom_energy_gain, rtol=1e-3, atol=0.0)
    assert np.isclose(final_beam.bunch_length(), main_source.bunch_length, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.charge(), main_source.charge, rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.rel_energy_spread(), 0.00030271, rtol=1e-1, atol=0.0)

    nom_beam_size_x = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_x()
    nom_beam_size_y = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_y()
    assert np.isclose(final_beam.beam_size_x(), nom_beam_size_x, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.beam_size_y(), nom_beam_size_y, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_x(), main_source.emit_nx, rtol=1e-2, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_y(), main_source.emit_ny, rtol=1e-1, atol=0.0)

    # Plot evolution in a stage
    plt.ion()
    linac.stages[1].plot_evolution()  # Need to set plt.show(block=False)

    # Plot linac survey
    linac.plot_survey()

    # Plot beam evolution
    linac.plot_evolution()
    linac.plot_waterfalls()

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageBasic_linac
def test_ramped_linac():
    """
    Tests a linac with ``StageBasic`` plasma stages. No driver jitter, with 
    ramps. 
    
    Checks whether various linac attributes are correctly set and compares some 
    output beam parameters against expected values.
    """

    np.random.seed(42)

    num_stages = 5
    enable_xt_jitter = False
    enable_xpyp_jitter = False
    
    driver_source = setup_basic_driver_source(enable_xt_jitter, enable_xpyp_jitter)
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, probe_evolution=False)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)
    interstage = setup_interstage(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages, alternate_interstage_polarity=False)

    # Perform tracking
    linac.run('test_ramped_linac', overwrite=True, verbose=False)
    
    # Check the outputs
    stages = linac.stages
    interstages = linac.interstages
    assert len(stages) == num_stages
    assert len(interstages) == num_stages - 1
    assert np.isclose(linac.get_length(), 93.0738788160906, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].nom_energy, 5.0e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[1].nom_energy, 36.9e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[2].nom_energy, 68.8e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[3].nom_energy, 100.7e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[4].nom_energy, 132.6e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[4].nom_energy_gain, 31.9e9, rtol=1e-15, atol=0.0)

    assert np.isclose(stages[0].upramp.length, 0.04413570063824205, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].upramp.length_flattop, stages[0].upramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].upramp.nom_energy, stages[0].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].upramp.nom_energy_flattop, stages[0].upramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].upramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].upramp.nom_energy_gain_flattop, stages[0].upramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].upramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].upramp.nom_accel_gradient_flattop, stages[0].upramp.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.length, 0.11989973028624575, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.length_flattop, stages[0].downramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.nom_energy, stages[1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.nom_energy_flattop, stages[0].downramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.nom_energy_gain_flattop, stages[0].downramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.nom_accel_gradient_flattop, stages[0].downramp.nom_accel_gradient, rtol=1e-15, atol=0.0)

    assert np.isclose(stages[-1].upramp.length, 0.22728814548579593, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.length_flattop, stages[-1].upramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy, stages[-1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_flattop, stages[-1].upramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_gain_flattop, stages[-1].upramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_accel_gradient_flattop, stages[-1].upramp.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.length, 0.2531558538336775, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.length_flattop, stages[-1].downramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy, stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_flattop, stages[-1].downramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_gain_flattop, stages[-1].downramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_accel_gradient_flattop, stages[-1].downramp.nom_accel_gradient, rtol=1e-15, atol=0.0)

    assert np.isclose(interstages[0].nom_energy, stages[1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[2].nom_energy, stages[3].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].nom_energy, stages[4].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].beta0, 0.24137835827389, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].length_dipole, 1.9209372712298547, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].field_dipole, 1.0, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].beta0, 0.45756933131960525, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].length_dipole, 3.641428291206625, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].field_dipole, 0.7541478129713424, rtol=1e-15, atol=0.0)

    initial_beam = linac.get_beam(0)
    assert np.isclose(initial_beam.energy(), stages[0].nom_energy, rtol=1e-3, atol=0.0)
    assert np.isclose(initial_beam.beta_x(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)
    assert np.isclose(initial_beam.beta_y(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)

    final_beam = linac.get_beam(-1)
    final_beam.beam_name = 'Test beam'
    assert final_beam.stage_number == 5
    assert np.isclose(final_beam.location, linac.get_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.energy(), stages[-1].nom_energy + stages[-1].nom_energy_gain + stages[-1].upramp.nom_energy_gain + stages[-1].downramp.nom_energy_gain, rtol=1e-3, atol=0.0)
    assert np.isclose(final_beam.bunch_length(), main_source.bunch_length, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.charge(), main_source.charge, rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.rel_energy_spread(), 0.00030198, rtol=1e-1, atol=0.0)

    nom_beam_size_x = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_x()
    nom_beam_size_y = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_y()
    assert np.isclose(final_beam.beam_size_x(), nom_beam_size_x, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.beam_size_y(), nom_beam_size_y, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_x(), main_source.emit_nx, rtol=1e-2, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_y(), main_source.emit_ny, rtol=1e-1, atol=0.0)

    final_beam.print_summary()

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageBasic_linac
def test_ramped_linac_vs_old_method():
    """
    Tests a linac with ``StageBasic`` plasma stages. No driver jitter, with 
    ramps. 
    
    Compares the output beam of linac with ``PlasmRamp``ramps and linac with 
    ramps defined with ``stage.__class__()``.
    """

    np.random.seed(42)

    num_stages = 5
    enable_xt_jitter = False
    enable_xpyp_jitter = False
    
    driver_source = setup_basic_driver_source(enable_xt_jitter, enable_xpyp_jitter)
    
    # Ramps constructed with PlasmRamp
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, probe_evolution=False)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)
    interstage = setup_interstage(stage)
    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages, alternate_interstage_polarity=False)
    linac.run('test_baseline_linac', overwrite=True, verbose=False)

    # Ramps constructed with stage.__class__()
    stage_old = setup_StageBasic(driver_source=driver_source, use_ramps=False, probe_evolution=False)
    stage_old.upramp = stage_old.__class__()
    stage_old.downramp = stage_old.__class__()
    stage_old.upramp.ramp_beta_mag = 10.0
    stage_old.downramp.ramp_beta_mag = 10.0
    interstage_old = setup_interstage(stage_old)
    linac_old = PlasmaLinac(source=main_source, stage=stage_old, interstage=interstage_old, num_stages=num_stages, alternate_interstage_polarity=False)
    linac_old.run('test_baseline_linac', overwrite=True, verbose=False)
    
    # Check the output beams
    final_beam = linac.get_beam(-1)
    final_beam.beam_name = 'Beam from linac with PlasmaRamp ramps'
    final_beam_old = linac_old.get_beam(-1)
    final_beam_old.beam_name = 'Beam from linac with stage.__class__() ramps'

    final_beam.print_summary()
    final_beam_old.print_summary()

    Beam.comp_beams(final_beam, final_beam_old, comp_location=True)

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageBasic_linac
def test_ramped_norm_emitt_jitter_linac():
    """
    Tests a linac with ``StageBasic`` plasma stages. Enabled driver normalised 
    emittance jitters, with ramps. 
    
    Only checks whether the linac can run without crashing.
    """

    np.random.seed(42)

    num_stages = 3
    
    driver_source = setup_basic_driver_source(enable_xt_jitter=False, enable_xpyp_jitter=False, enable_norm_emittance_jitter=True)
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, probe_evolution=False)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)
    interstage = setup_interstage(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages, alternate_interstage_polarity=False)

    # Perform tracking
    linac.run('test_ramped_norm_emitt_jitter_linac', overwrite=True, verbose=False)

    
@pytest.mark.StageBasic_linac
def test_ramped_jitter_linac():
    """
    Tests a linac with ``StageBasic`` plasma stages. Enabled driver jitters, 
    with ramps. 
    
    Checks whether various linac attributes are correctly set and compares some 
    output beam parameters against expected values.
    """

    np.random.seed(42)

    num_stages = 16
    enable_xt_jitter = True
    enable_xpyp_jitter = True
    
    driver_source = setup_basic_driver_source(enable_xt_jitter, enable_xpyp_jitter)
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, probe_evolution=False)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)
    interstage = setup_interstage(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages, alternate_interstage_polarity=False)

    # Perform tracking
    linac.run('test_ramped_jitter_linac', overwrite=True, verbose=False)

    # Check the outputs
    stages = linac.stages
    interstages = linac.interstages
    assert len(stages) == num_stages
    assert len(interstages) == num_stages - 1
    assert np.isclose(linac.get_length(), 518.752417450173, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].nom_energy, 5.0e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[11].nom_energy, 355.9e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[12].nom_energy, 387.8e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[13].nom_energy, 419.7e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[15].nom_energy, 483.5e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[15].nom_energy_gain, 31.9e9, rtol=1e-15, atol=0.0)

    assert np.isclose(stages[-1].upramp.length, 0.4340135238090143, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.length_flattop, stages[-1].upramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy, stages[-1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_flattop, stages[-1].upramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_gain_flattop, stages[-1].upramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_accel_gradient_flattop, stages[-1].upramp.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.length, 0.44810235895496436, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.length_flattop, stages[-1].downramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy, stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_flattop, stages[-1].downramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_gain_flattop, stages[-1].downramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_accel_gradient_flattop, stages[-1].downramp.nom_accel_gradient, rtol=1e-15, atol=0.0)

    assert np.isclose(interstages[10].nom_energy, stages[11].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[12].nom_energy, stages[13].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[14].nom_energy, stages[15].nom_energy, rtol=1e-15, atol=0.0)
    interstage_nom_energy = interstages[0].nom_energy
    assert np.isclose(interstages[0].beta0, stage.matched_beta_function(interstage_nom_energy), rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].length_dipole, np.sqrt(interstage_nom_energy/10e9), rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].field_dipole, np.min([1.0, 100e9/interstage_nom_energy]), rtol=1e-15, atol=0.0)
    interstage_nom_energy = interstages[14].nom_energy
    assert np.isclose(interstages[14].beta0, stage.matched_beta_function(interstage_nom_energy), rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[14].length_dipole, np.sqrt(interstage_nom_energy/10e9), rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[14].field_dipole, np.min([1.0, 100e9/interstage_nom_energy]), rtol=1e-15, atol=0.0)

    initial_beam = linac.get_beam(0)
    assert np.isclose(initial_beam.energy(), stages[0].nom_energy, rtol=1e-3, atol=0.0)
    assert np.isclose(initial_beam.beta_x(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)
    assert np.isclose(initial_beam.beta_y(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)

    final_beam = linac.get_beam(-1)
    final_beam.beam_name = 'Test beam'
    assert final_beam.stage_number == num_stages
    assert np.isclose(final_beam.location, linac.get_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.energy(), stages[-1].nom_energy + stages[-1].nom_energy_gain + stages[-1].upramp.nom_energy_gain + stages[-1].downramp.nom_energy_gain, rtol=1e-3, atol=0.0)
    assert np.isclose(final_beam.bunch_length(), main_source.bunch_length, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.charge(), main_source.charge, rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.rel_energy_spread(), 9.825586100176821e-05, rtol=1e-2, atol=0.0)

    nom_beam_size_x = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4) * initial_beam.beam_size_x()
    #nom_beam_size_y = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4) * initial_beam.beam_size_y()
    assert np.isclose(final_beam.beam_size_x(), nom_beam_size_x, rtol=1e-2, atol=0.0)
    assert np.isclose(final_beam.beam_size_y(), 1.2967329484466221e-06, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_x(), main_source.emit_nx, rtol=1e-2, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_y(), 1.084878053041328e-06, rtol=1e-2, atol=0.0)

    #ref_beam = Beam.load('./tests/data/')
    #ref_beam.beam_name = 'Reference beam'

    final_beam.print_summary()
    #ref_beam.print_summary()
    #Beam.comp_beam_params(final_beam, ref_beam, comp_location=True)  # Compare output beam with reference beam file.

    # Remove output directory
    shutil.rmtree(linac.run_path())

