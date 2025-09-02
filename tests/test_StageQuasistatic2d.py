# -*- coding: utf-8 -*-
"""
ABEL : StageQuasistatic2d unit tests
=======================================

This file is a part of ABEL.
Copyright 2022– C.A.Lindstrøm, J.B.B.Chen, O.G.Finnerud,
D.Kallvik, E.Hørlyk, K.N.Sjobak, E.Adli, University of Oslo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import pytest
from abel import *
import shutil
import numpy as np


def setup_basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=0.0, y_angle=0.0):
    driver = SourceBasic()
    driver.bunch_length = 42e-6                                                   # [m] This value is for trapezoid.
    driver.z_offset = 300e-6                                                      # [m]
    driver.x_angle = x_angle                                                      # [rad]
    driver.y_angle = y_angle                                                      # [rad]

    driver.num_particles = 100000                                                 
    driver.charge = -2.7e10 * SI.e                                                # [C]
    driver.energy = 31.25e9                                                       # [eV]
    driver.rel_energy_spread = 0.01                                               # Relative rms energy spread

    driver.emit_nx, driver.emit_ny = 80e-6, 80e-6                                 # [m rad]
    driver.beta_x, driver.beta_y = 0.2, 0.2                                       # [m]

    if enable_xy_jitter:
        driver.jitter.x = 100e-9                                                  # [m], std
        driver.jitter.y = 100e-9                                                  # [m], std

    if enable_xpyp_jitter:
        driver.jitter.xp = 1.0e-6                                                 # [rad], std
        driver.jitter.yp = 1.0e-6                                                 # [rad], std

    driver.symmetrize = True

    return driver


def setup_basic_main_source(plasma_density=7.0e21, ramp_beta_mag=1.0):
    from abel.utilities.plasma_physics import beta_matched

    main = SourceBasic()
    main.bunch_length = 18e-6                                                     # [m], rms. Standard value
    main.num_particles = 10000                                               
    main.charge = -SI.e * 1.0e10                                                  # [C]

    # Energy parameters
    main.energy = 5.0e9                                                           # [eV]
    main.rel_energy_spread = 0.01                                                 # Relative rms energy spread

    # Emittances
    main.emit_nx, main.emit_ny = 160e-6, 0.56e-6                                  # [m rad]

    # Beta functions
    main.beta_x = beta_matched(plasma_density, main.energy) * ramp_beta_mag       # [m]
    main.beta_y = main.beta_x                                                     # [m]

    # Offsets
    main.z_offset = -36e-6                                                        # [m] # Standard value

    # Other
    main.symmetrize_6d = True

    return main


def setup_StageQuasistatic2d(driver_source=None, nom_accel_gradient=6.4e9, nom_energy_gain=31.9e9, plasma_density=7.0e21, enable_radiation_reaction=False, probe_evolution=False, return_tracked_driver=False, use_ramps=False, store_beams_for_tests=False):

    stage = StageQuasistatic2d()
    stage.nom_accel_gradient = nom_accel_gradient                                 # [V/m]
    stage.nom_energy_gain = nom_energy_gain                                       # [eV]
    stage.plasma_density = plasma_density                                         # [m^-3]
    stage.driver_source = driver_source
    if use_ramps:
        stage.ramp_beta_mag = 10.0
    else:
        stage.ramp_beta_mag = 1.0
    stage.enable_radiation_reaction = enable_radiation_reaction
    stage.probe_evolution = probe_evolution
    stage._return_tracked_driver = return_tracked_driver
    stage.store_beams_for_tests = store_beams_for_tests

    # Set up ramps after the stage is fully configured
    if use_ramps:
        stage.upramp = PlasmaRamp() 
        stage.downramp = PlasmaRamp()
    
    return stage


@pytest.mark.StageQuasistatic2d
def test_driver_unrotation():
    """
    Tests for checking the driver being correctly un-rotated back to its 
    original coordinate system.
    """
    
    np.random.seed(42)

    # ========== No driver jitter, with angular offset, no ramps ==========
    x_angle = 1.3e-6                                                              # [rad]
    y_angle = 2e-6                                                                # [rad]
    driver_source = setup_basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=x_angle, y_angle=y_angle)
    stage = setup_StageQuasistatic2d(driver_source=driver_source, use_ramps=False, return_tracked_driver=True, store_beams_for_tests=True)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)

    stage.nom_energy = 51.4e9                                                       # [eV]
    _, driver = stage.track(main_source.track())
    driver0 = stage.driver_incoming

    x_drift = stage.length * np.tan(driver0.x_angle())
    y_drift = stage.length * np.tan(driver0.y_angle())
    xs = driver0.xs()
    ys = driver0.ys()
    driver0.set_xs(xs + x_drift)
    driver0.set_ys(ys + y_drift)
    
    # Cannot compare the whole phase space due to driver evolution
    assert np.isclose(driver.x_offset(), driver0.x_offset(), rtol=1e-4, atol=0.0)
    assert np.isclose(driver.y_offset(), driver0.y_offset(), rtol=1e-4, atol=0.0)
    assert np.isclose(driver.z_offset(), driver0.z_offset(), rtol=1e-9, atol=0.0)
    assert np.isclose(driver.x_angle(), driver0.x_angle(), rtol=1e-4, atol=0.0)
    assert np.isclose(driver.y_angle(), driver0.y_angle(), rtol=1e-4, atol=0.0)


    # ========== Driver jitter, no angular offset ==========
    driver_source = setup_basic_driver_source(enable_xy_jitter=True, enable_xpyp_jitter=True)
    stage = setup_StageQuasistatic2d(driver_source=driver_source, use_ramps=True, return_tracked_driver=True, store_beams_for_tests=True)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)

    stage.nom_energy = 51.4e9                                                       # [eV]
    _, driver = stage.track(main_source.track())
    driver0 = stage.driver_incoming

    x_drift = stage.length * np.tan(driver0.x_angle())
    y_drift = stage.length * np.tan(driver0.y_angle())
    xs = driver0.xs()
    ys = driver0.ys()
    driver0.set_xs(xs + x_drift)
    driver0.set_ys(ys + y_drift)

    # Cannot compare the whole phase space due to driver evolution
    assert np.isclose(driver.x_offset(), driver0.x_offset(), rtol=1e-1, atol=0.0)
    assert np.isclose(driver.y_offset(), driver0.y_offset(), rtol=1e-3, atol=0.0)
    assert np.isclose(driver.z_offset(), driver0.z_offset(), rtol=1e-9, atol=0.0)
    assert np.isclose(driver.x_angle(), driver0.x_angle(), rtol=1e-3, atol=0.0)
    assert np.isclose(driver.y_angle(), driver0.y_angle(), rtol=1e-3, atol=0.0)
    assert np.isclose(driver.norm_emittance_x(), driver0.norm_emittance_x(), rtol=1e-2, atol=0.0)
    assert np.isclose(driver.norm_emittance_y(), driver0.norm_emittance_y(), rtol=1e-2, atol=0.0)
    assert np.isclose(driver.bunch_length(), driver0.bunch_length(), rtol=1e-10, atol=0.0)
    assert np.isclose(driver.peak_current(), driver0.peak_current(), rtol=1e-3, atol=0.0)
    assert np.isclose(driver.particle_mass, driver0.particle_mass, rtol=1e-13, atol=0.0)
    assert np.allclose(driver.qs(), driver0.qs(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.weightings(), driver0.weightings(), rtol=1e-13, atol=0.0)


    # ========== No jitter, no angular offset ==========
    driver_source = setup_basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False)
    stage = setup_StageQuasistatic2d(driver_source=driver_source, use_ramps=True, return_tracked_driver=True, store_beams_for_tests=True)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)

    stage.nom_energy = 36.9e9                                                       # [eV]

    _, driver = stage.track(main_source.track())
    driver0 = stage.driver_incoming

    # Cannot compare the whole phase space due to driver evolution
    assert np.isclose(driver.x_offset(), driver0.x_offset(), rtol=0.0, atol=1e-20)
    assert np.isclose(driver.y_offset(), driver0.y_offset(), rtol=0.0, atol=1e-20)
    assert np.isclose(driver.z_offset(), driver0.z_offset(), rtol=1e-15, atol=0.0)
    assert np.isclose(driver.x_angle(), driver0.x_angle(), rtol=0.0, atol=1e-20)
    assert np.isclose(driver.y_angle(), driver0.y_angle(), rtol=0.0, atol=1e-20)
    assert np.isclose(driver.norm_emittance_x(), driver0.norm_emittance_x(), rtol=1e-2, atol=0.0)
    assert np.isclose(driver.norm_emittance_y(), driver0.norm_emittance_y(), rtol=1e-2, atol=0.0)
    assert np.isclose(driver.bunch_length(), driver0.bunch_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(driver.peak_current(), driver0.peak_current(), rtol=1e-15, atol=0.0)
    assert np.isclose(driver.particle_mass, driver0.particle_mass, rtol=1e-13, atol=0.0)
    assert np.allclose(driver.qs(), driver0.qs(), rtol=1e-15, atol=0.0)
    assert np.allclose(driver.weightings(), driver0.weightings(), rtol=1e-13, atol=0.0)


    # ========== No jitter, large angular offset ==========
    x_angle = 5e-6                                                                # [rad]
    y_angle = 2e-5                                                                # [rad]
    driver_source2 = setup_basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=x_angle, y_angle=y_angle)
    stage2 = setup_StageQuasistatic2d(driver_source=driver_source2, use_ramps=True, return_tracked_driver=True, store_beams_for_tests=True)
    main_source2 = setup_basic_main_source(ramp_beta_mag=stage2.ramp_beta_mag)

    stage2.nom_energy = 36.9e9                                                      # [eV]
    _, driver = stage2.track(main_source2.track())
    driver0 = stage2.driver_incoming

    x_drift = stage2.length * np.tan(x_angle)
    y_drift = stage2.length * np.tan(y_angle)
    xs = driver0.xs()
    ys = driver0.ys()
    driver0.set_xs(xs + x_drift)
    driver0.set_ys(ys + y_drift)

    # Cannot compare the whole phase space due to driver evolution
    assert np.isclose(driver.x_offset(), driver0.x_offset(), rtol=1e-4, atol=0.0)
    assert np.isclose(driver.y_offset(), driver0.y_offset(), rtol=1e-4, atol=0.0)
    assert np.isclose(driver.z_offset(), driver0.z_offset(), rtol=1e-9, atol=0.0)
    assert np.isclose(driver.x_angle(), driver0.x_angle(), rtol=1e-4, atol=0.0)
    assert np.isclose(driver.y_angle(), driver0.y_angle(), rtol=1e-3, atol=0.0)
    assert np.isclose(driver.norm_emittance_x(), driver0.norm_emittance_x(), rtol=1e-2, atol=0.0)
    assert np.isclose(driver.norm_emittance_y(), driver0.norm_emittance_y(), rtol=1e-2, atol=0.0)
    assert np.isclose(driver.bunch_length(), driver0.bunch_length(), rtol=1e-9, atol=0.0)
    assert np.isclose(driver.peak_current(), driver0.peak_current(), rtol=1e-2, atol=0.0)
    assert np.isclose(driver.particle_mass, driver0.particle_mass, rtol=1e-13, atol=0.0)
    assert np.allclose(driver.qs(), driver0.qs(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.weightings(), driver0.weightings(), rtol=1e-13, atol=0.0)


@pytest.mark.StageQuasistatic2d
def test_baseline_tracking():
    """
    Tests for tracking ``StageQuasistatic2d`` without ramps and radiation 
    reaction.

    Examines stage configuraions and the output main beam parameters.
    """

    np.random.seed(42)
    
    driver_source = setup_basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=0.0, y_angle=0.0)
    stage = setup_StageQuasistatic2d(driver_source=driver_source, enable_radiation_reaction=False, use_ramps=False)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)

    stage.nom_energy = 5.0e9                                                        # [eV]
    beam = stage.track(main_source.track())

    # Inspect stage configurations
    assert stage.has_ramp() is False
    assert stage.enable_radiation_reaction is False
    assert stage.probe_evolution is False
    assert stage._return_tracked_driver is False
    assert stage.store_beams_for_tests is False
    assert np.isclose(stage.plasma_density, 7.0e21, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy, 5.0e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_flattop, stage.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_gain, 31.9e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_gain_flattop, stage.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_accel_gradient, 6.4e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_accel_gradient_flattop, stage.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.length, stage.nom_energy_gain/stage.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.length_flattop, stage.length, rtol=1e-15, atol=0.0)

    # Examine output beam
    assert np.isclose(beam.energy(), main_source.energy + stage.nom_energy_gain, rtol=2e-2, atol=0.0)
    assert np.isclose(beam.charge(), main_source.charge, rtol=1e-15, atol=0.0)
    assert len(beam) == main_source.num_particles
    assert beam.stage_number == 1
    assert np.isclose(beam.location, stage.length, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.particle_mass, SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.x_offset(), main_source.x_offset, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.y_offset(), main_source.y_offset, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.z_offset(), main_source.z_offset, rtol=1e-3, atol=0.0)
    assert np.isclose(beam.x_angle(), main_source.x_angle, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.y_angle(), main_source.y_angle, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.norm_emittance_x(), main_source.emit_nx, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), main_source.emit_ny, rtol=7e-2, atol=0.0)
    nom_beta_x = np.sqrt(beam.energy()/main_source.energy) * main_source.beta_x
    nom_beta_y = np.sqrt(beam.energy()/main_source.energy) * main_source.beta_y
    assert np.isclose(beam.beta_x(), nom_beta_x, rtol=5e-2, atol=0.0)
    assert np.isclose(beam.beta_y(), nom_beta_y, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.bunch_length(), main_source.bunch_length, rtol=5e-2, atol=0.0)
    assert np.isclose(beam.rel_energy_spread(), 4.0e-2, rtol=1e-1, atol=0.0)


@pytest.mark.StageQuasistatic2d
def test_ramped_tracking():
    """
    Tests for tracking ``StageQuasistatic2d`` with ramps, but without radiation 
    reaction.

    Examines stage configuraions and the output main beam parameters.
    """

    np.random.seed(42)
    
    driver_source = setup_basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=0.0, y_angle=0.0)
    stage = setup_StageQuasistatic2d(driver_source=driver_source, enable_radiation_reaction=False, use_ramps=True, probe_evolution=True)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)

    stage.nom_energy = 5.0e9                                                        # [eV]
    beam = stage.track(main_source.track())

    # Inspect stage configurations
    assert stage.has_ramp() is True
    assert stage.enable_radiation_reaction is False
    assert stage.probe_evolution is True
    assert stage._return_tracked_driver is False
    assert stage.store_beams_for_tests is False
    assert np.isclose(stage.plasma_density, 7.0e21, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy, 5.0e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_flattop, stage.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_gain, 31.9e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_gain_flattop, stage.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_accel_gradient, 6.4e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_accel_gradient_flattop, 6.6e9, rtol=1e-2, atol=0.0)
    assert np.isclose(stage.length, stage.nom_energy_gain/stage.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.length_flattop, stage.nom_energy_gain_flattop/stage.nom_accel_gradient_flattop, rtol=1e-15, atol=0.0)

    # Examine output beam
    assert np.isclose(beam.energy(), main_source.energy + stage.nom_energy_gain, rtol=3e-2, atol=0.0)
    assert np.isclose(beam.charge(), main_source.charge, rtol=1e-15, atol=0.0)
    assert len(beam) == main_source.num_particles
    assert beam.stage_number == 1
    assert np.isclose(beam.location, stage.length, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.particle_mass, SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.x_offset(), main_source.x_offset, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.y_offset(), main_source.y_offset, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.z_offset(), main_source.z_offset, rtol=1e-3, atol=0.0)
    assert np.isclose(beam.x_angle(), main_source.x_angle, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.y_angle(), main_source.y_angle, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.norm_emittance_x(), main_source.emit_nx, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), main_source.emit_ny, rtol=7e-2, atol=0.0)
    nom_beta_x = np.sqrt(beam.energy()/main_source.energy) * main_source.beta_x
    nom_beta_y = np.sqrt(beam.energy()/main_source.energy) * main_source.beta_y
    assert np.isclose(beam.beta_x(), nom_beta_x, rtol=5e-2, atol=0.0)
    assert np.isclose(beam.beta_y(), nom_beta_y, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.bunch_length(), main_source.bunch_length, rtol=5e-2, atol=0.0)
    assert np.isclose(beam.rel_energy_spread(), 4.0e-2, rtol=1e-1, atol=0.0)

    # Also test Stage.plot_evolution()
    #import time
    #from matplotlib import pyplot as plt
    #plt.ion()
    #stage.plot_evolution()
    #time.sleep(5)  # pauses for 5 seconds


    






