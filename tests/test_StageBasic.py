# -*- coding: utf-8 -*-
"""
ABEL : StageBasic unit tests
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


def setup_StageBasic(driver_source=None, nom_accel_gradient=6.4e9, nom_energy_gain=31.9e9, plasma_density=7.0e21, use_ramps=False, transformer_ratio=1, depletion_efficiency=0.75, probe_evolution=False, return_tracked_driver=False, store_beams_for_tests=False):
    
    stage = StageBasic()
    stage.nom_accel_gradient = nom_accel_gradient                                 # [V/m]
    stage.nom_energy_gain = nom_energy_gain                                       # [eV]
    stage.plasma_density = plasma_density                                         # [m^-3]
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


@pytest.mark.StageBasic
def test_stage_length_gradient_energyGain():
    """
    Tests ensuring that the flattop length and total length of the stage as well 
    as nominal gradient and nominal energy gain are set correctly.
    """

    np.random.seed(42)
    driver_source = setup_basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False)


    # ========== Set flattop length and nominal energy gain ==========
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, store_beams_for_tests=False)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)

    stage.length_flattop = 4.82                                                   # [m]
    stage.nom_energy_gain = 31.9e9                                                # [eV]

    linac = PlasmaLinac(source=main_source, stage=stage, num_stages=1, alternate_interstage_polarity=False)
    linac.run('test_stage_length_gradient_energyGain', overwrite=True, verbose=False)

    assert np.allclose(stage.nom_energy_gain, 31.9e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length_flattop, 4.82, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient, stage.nom_energy_gain/stage.length, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient_flattop, stage.nom_energy_gain_flattop/stage.length_flattop, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length, stage.length_flattop + stage.upramp.length_flattop + stage.downramp.length_flattop, rtol=1e-15, atol=0.0)


    # ========== Set flattop nominal accelertaion gradient and nominal energy gain ==========
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, store_beams_for_tests=False)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)

    stage.nom_energy_gain = 7.8e9                                                 # [eV]
    stage.nom_accel_gradient = None                                               # [V/m]
    stage.nom_accel_gradient_flattop = 1.0e9                                      # [V/m]
    
    linac = PlasmaLinac(source=main_source, stage=stage, num_stages=1, alternate_interstage_polarity=False)
    linac.run('test_stage_length_gradient_energyGain', overwrite=True, verbose=False)

    assert np.allclose(stage.nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient_flattop, 1.0e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient, stage.nom_energy_gain/stage.length, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient_flattop, stage.nom_energy_gain_flattop/stage.length_flattop, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length, stage.length_flattop + stage.upramp.length_flattop + stage.downramp.length_flattop, rtol=1e-15, atol=0.0)

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageBasic
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
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=False, return_tracked_driver=True, store_beams_for_tests=True)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)

    stage.nom_energy = 51.4e9                                                    # [eV]
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

    # Deplete driver0 energy depletion for ramps and stage to be comparable to driver
    driver0.set_Es(driver0.Es()*(1-stage.depletion_efficiency))
    assert np.isclose(driver.norm_emittance_x(), driver0.norm_emittance_x(), rtol=1e-7, atol=0.0)
    assert np.isclose(driver.norm_emittance_y(), driver0.norm_emittance_y(), rtol=1e-7, atol=0.0)
    assert np.isclose(driver.bunch_length(), driver0.bunch_length(), rtol=1e-11, atol=0.0)
    assert np.isclose(driver.peak_current(), driver0.peak_current(), rtol=1e-6, atol=0.0)
    assert np.isclose(driver.particle_mass, driver0.particle_mass, rtol=1e-13, atol=0.0)

    assert np.allclose(driver.qs(), driver0.qs(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.weightings(), driver0.weightings(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.xs(), driver0.xs(), rtol=0.0, atol=1e-8)
    assert np.allclose(driver.ys(), driver0.ys(), rtol=0.0, atol=1e-8)
    assert np.allclose(driver.zs(), driver0.zs(), rtol=0.0, atol=1e-8)
    assert np.allclose(driver.uzs(), driver0.uzs(), rtol=1e-6, atol=0.0)


    # ========== Driver jitter, no angular offset ==========
    driver_source = setup_basic_driver_source(enable_xy_jitter=True, enable_xpyp_jitter=True)
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, return_tracked_driver=True, store_beams_for_tests=True)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)

    stage.nom_energy = 51.4e9                                                    # [eV]
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
    assert np.isclose(driver.y_angle(), driver0.y_angle(), rtol=1e-3, atol=0.0)
    assert np.isclose(driver.norm_emittance_x(), driver0.norm_emittance_x(), rtol=1e-7, atol=0.0)
    assert np.isclose(driver.norm_emittance_y(), driver0.norm_emittance_y(), rtol=1e-7, atol=0.0)
    assert np.isclose(driver.bunch_length(), driver0.bunch_length(), rtol=1e-11, atol=0.0)
    assert np.isclose(driver.peak_current(), driver0.peak_current(), rtol=1e-3, atol=0.0)
    assert np.isclose(driver.particle_mass, driver0.particle_mass, rtol=1e-13, atol=0.0)

    assert np.allclose(driver.qs(), driver0.qs(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.weightings(), driver0.weightings(), rtol=1e-13, atol=0.0)

    # Deplete driver0 energy depletion for ramps and stage to be comparable to driver
    driver0.set_Es(driver0.Es()*(1-stage.depletion_efficiency))
    driver0.set_Es(driver0.Es()*(1-stage.depletion_efficiency))
    driver0.set_Es(driver0.Es()*(1-stage.depletion_efficiency))

    assert np.allclose(driver.uzs(), driver0.uzs(), rtol=1e-6, atol=0.0)


    # ========== No jitter, no angular offset ==========
    driver_source = setup_basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False)
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, return_tracked_driver=True, store_beams_for_tests=True)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)

    stage.nom_energy = 36.9e9                                                     # [eV]
    _, driver = stage.track(main_source.track())
    driver0 = stage.driver_incoming

    # Cannot compare the whole phase space due to driver evolution
    assert np.isclose(driver.x_offset(), driver0.x_offset(), rtol=0.0, atol=1e-20)
    assert np.isclose(driver.y_offset(), driver0.y_offset(), rtol=0.0, atol=1e-20)
    assert np.isclose(driver.z_offset(), driver0.z_offset(), rtol=1e-15, atol=0.0)
    assert np.isclose(driver.x_angle(), driver0.x_angle(), rtol=0.0, atol=1e-20)
    assert np.isclose(driver.y_angle(), driver0.y_angle(), rtol=0.0, atol=1e-20)
    assert np.isclose(driver.norm_emittance_x(), driver0.norm_emittance_x(), rtol=1e-13, atol=0.0)
    assert np.isclose(driver.norm_emittance_y(), driver0.norm_emittance_y(), rtol=1e-13, atol=0.0)
    assert np.isclose(driver.bunch_length(), driver0.bunch_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(driver.peak_current(), driver0.peak_current(), rtol=1e-15, atol=0.0)
    assert np.isclose(driver.particle_mass, driver0.particle_mass, rtol=1e-13, atol=0.0)

    assert np.allclose(driver.qs(), driver0.qs(), rtol=1e-15, atol=0.0)
    assert np.allclose(driver.weightings(), driver0.weightings(), rtol=1e-13, atol=0.0)


    # ========== No jitter, large angular offset ==========
    x_angle = 5e-6                                                                # [rad]
    y_angle = 2e-5                                                                # [rad]
    driver_source2 = setup_basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=x_angle, y_angle=y_angle)
    stage2 = setup_StageBasic(driver_source=driver_source2, use_ramps=True, return_tracked_driver=True, store_beams_for_tests=True)
    main_source2 = setup_basic_main_source(ramp_beta_mag=stage2.ramp_beta_mag)

    stage2.nom_energy = 36.9e9                                                     # [eV]
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

    # Deplete driver0 energy depletion for ramps and stage to be comparable to driver
    driver0.set_Es(driver0.Es()*(1-stage.depletion_efficiency))
    driver0.set_Es(driver0.Es()*(1-stage.depletion_efficiency))
    driver0.set_Es(driver0.Es()*(1-stage.depletion_efficiency))
    assert np.isclose(driver.norm_emittance_x(), driver0.norm_emittance_x(), rtol=1e-6, atol=0.0)
    assert np.isclose(driver.norm_emittance_y(), driver0.norm_emittance_y(), rtol=1e-5, atol=0.0)
    assert np.isclose(driver.bunch_length(), driver0.bunch_length(), rtol=1e-9, atol=0.0)
    assert np.isclose(driver.peak_current(), driver0.peak_current(), rtol=1e-2, atol=0.0)
    assert np.isclose(driver.particle_mass, driver0.particle_mass, rtol=1e-13, atol=0.0)

    assert np.allclose(driver.qs(), driver0.qs(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.weightings(), driver0.weightings(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.uzs(), driver0.uzs(), rtol=1e-5, atol=0.0)


@pytest.mark.StageBasic
def test_optimize_plasma_density():
    """
    Tests for ``StageBasic.optimize_plasma_density()``.
    """

    np.random.seed(42)

    driver_source = setup_basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False)
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=False, return_tracked_driver=True, store_beams_for_tests=True)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)

    stage.optimize_plasma_density(main_source)
    assert np.isclose(stage.plasma_density, 6.592382486466319e+21, rtol=1e-5, atol=0.0)


@pytest.mark.StageBasic
def test_copy_config2blank_stage():
    """
    Tests for ``StageBasic.copy_config2blank_stage()``.
    """

    np.random.seed(42)

    stage = setup_StageBasic(driver_source=None, use_ramps=False, return_tracked_driver=True, store_beams_for_tests=True)

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
    assert stage_copy.has_ramp() is False
    assert stage_copy.is_upramp() is False
    assert stage_copy.is_downramp() is False
    
    assert np.isclose(stage_copy.transformer_ratio, stage.transformer_ratio, rtol=1e-5, atol=0.0)
    assert np.isclose(stage_copy.depletion_efficiency, stage.depletion_efficiency, rtol=1e-5, atol=0.0)
    assert np.isclose(stage_copy.probe_evolution, stage.probe_evolution, rtol=1e-5, atol=0.0)


    stage = setup_StageBasic(driver_source=None, use_ramps=False, return_tracked_driver=True, store_beams_for_tests=True)
    stage.transformer_ratio = 1.74
    stage.depletion_efficiency = 0.5354
    stage.probe_evolution = True
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
    assert stage_copy.has_ramp() is False
    assert stage_copy.is_upramp() is False
    assert stage_copy.is_downramp() is False
    
    assert np.isclose(stage_copy.transformer_ratio, stage.transformer_ratio, rtol=1e-5, atol=0.0)
    assert np.isclose(stage_copy.depletion_efficiency, stage.depletion_efficiency, rtol=1e-5, atol=0.0)
    assert np.isclose(stage_copy.probe_evolution, stage.probe_evolution, rtol=1e-5, atol=0.0)


    