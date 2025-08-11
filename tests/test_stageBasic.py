# -*- coding: utf-8 -*-
"""
ABEL : StageBasic tests
=======================================

This file is a part of ABEL.
Copyright 2022– C.A.Lindstrøm, B.Chen, O.G. Finnerud,
D. Kallvik, E. Hørlyk, K.N. Sjobak, E.Adli, University of Oslo

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


def setup_Basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=0.0, y_angle=0.0):
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


def setup_basic_main_source(plasma_density=7.0e21, ramp_beta_mag=10.0):
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


def setup_StageBasic(driver_source=None, nom_accel_gradient=6.4e9, nom_energy_gain=31.9e9, plasma_density=7.0e21, ramp_beta_mag=10.0, use_ramps=False, transformer_ratio=1, depletion_efficiency=0.75, calc_evolution=False, return_tracked_driver=False, test_beam_between_ramps=False):
    
    stage = StageBasic()
    stage.nom_accel_gradient = nom_accel_gradient                                 # [V/m]
    stage.nom_energy_gain = nom_energy_gain                                       # [eV]
    stage.plasma_density = plasma_density                                         # [m^-3]
    stage.driver_source = driver_source
    stage.ramp_beta_mag = ramp_beta_mag
    stage.test_beam_between_ramps = test_beam_between_ramps
    stage.transformer_ratio = transformer_ratio
    stage.depletion_efficiency = depletion_efficiency
    stage.calc_evolution = calc_evolution
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
    driver_source = setup_Basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False)
    main_source = setup_basic_main_source()


    # ========== Set flattop length and nominal energy gain ==========
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, test_beam_between_ramps=False)

    stage.length_flattop = 4.82                                                   # [m]
    stage.nom_energy_gain = 31.9e9                                                # [eV]

    linac = PlasmaLinac(source=main_source, stage=stage, num_stages=1)
    linac.run('test_stage_length_gradient_energyGain', overwrite=True, verbose=False)

    assert np.allclose(stage.nom_energy_gain, 31.9e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length_flattop, 4.82, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient, stage.nom_energy_gain/stage.length, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient_flattop, stage.nom_energy_gain_flattop/stage.length_flattop, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length, stage.length_flattop + stage.upramp.length_flattop + stage.downramp.length_flattop, rtol=1e-15, atol=0.0)


    # ========== Set flattop nominal accelertaion gradient and nominal energy gain ==========
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, test_beam_between_ramps=False)

    stage.nom_energy_gain = 7.8e9                                                 # [eV]
    stage.nom_accel_gradient = None                                               # [V/m]
    stage.nom_accel_gradient_flattop = 1.0e9                                      # [V/m]
    
    linac = PlasmaLinac(source=main_source, stage=stage, num_stages=1)
    linac.run('test_stage_length_gradient_energyGain', overwrite=True, verbose=False)

    assert np.allclose(stage.nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient_flattop, 1.0e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient, stage.nom_energy_gain/stage.length, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient_flattop, stage.nom_energy_gain_flattop/stage.length_flattop, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length, stage.length_flattop + stage.upramp.length_flattop + stage.downramp.length_flattop, rtol=1e-15, atol=0.0)

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageBasic
def test_beam_between_ramps():
    """
    Tests for ensuring that the beams are correctly transferred between ramps 
    and stage.
    """

    np.random.seed(42)

    # ========== Driver jitter, no angular offset ==========
    driver_source = setup_Basic_driver_source(enable_xy_jitter=True, enable_xpyp_jitter=True)
    main_source = setup_basic_main_source()
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, test_beam_between_ramps=True)

    linac = PlasmaLinac(source=main_source, stage=stage, num_stages=1)
    linac.run('test_beam_between_ramps', overwrite=True, verbose=False)

    # Assert that there has been no significant changes in the beams between parents and its upramp
    # Between a upramp and main stage
    Beam.comp_beams(stage.upramp.driver_out, stage.driver_in, comp_location=True)
    Beam.comp_beams(stage.upramp.beam_out, stage.beam_in, comp_location=True)

    # Between a main stage and downramp
    Beam.comp_beams(stage.driver_out, stage.downramp.driver_in, comp_location=True, rtol=1e-11, atol=0.0)
    Beam.comp_beams(stage.beam_out, stage.downramp.beam_in, comp_location=True, rtol=1e-11, atol=0.0)

    # Assert that the output beam matches the out beam for the downramp
    final_beam = linac[0].get_beam(-1)
    downramp_beam_out = stage.downramp.beam_out
    downramp_beam_out.stage_number += 1  # There is a mismatch in stage number, as it is not added until after the stage.
    _, beam_outgoing = stage.undo_beam_coordinate_systems_rotation(stage.driver_incoming, stage.downramp.driver_out, downramp_beam_out)
    Beam.comp_beams(final_beam, beam_outgoing, comp_location=True, rtol=1e-11, atol=0.0)

    # Assert that the propagation length of the output beam matches the total length of the stage
    assert np.allclose(final_beam.location - stage.upramp.beam_in.location, stage.length, rtol=1e-15, atol=0.0)


    # ========== No jitter, no angular offset ==========
    driver_source2 = setup_Basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False)
    main_source2 = setup_basic_main_source()
    stage2 = setup_StageBasic(driver_source=driver_source2, use_ramps=True, test_beam_between_ramps=True)

    linac = PlasmaLinac(source=main_source2, stage=stage2, num_stages=1)
    linac.run('test_beam_between_ramps', overwrite=True, verbose=False)

    # Assert that there has been no significant changes in the beams between parents and its upramp
    # Between a upramp and main stage
    Beam.comp_beams(stage2.upramp.driver_out, stage2.driver_in, comp_location=True)
    Beam.comp_beams(stage2.upramp.beam_out, stage2.beam_in, comp_location=True)

    # Between a main stage and downramp
    Beam.comp_beams(stage2.driver_out, stage2.downramp.driver_in, comp_location=True, rtol=1e-13, atol=0.0)
    Beam.comp_beams(stage2.beam_out, stage2.downramp.beam_in, comp_location=True, rtol=1e-11, atol=0.0)

    # Assert that the output beam matches the out beam for the downramp
    final_beam = linac[0].get_beam(-1)
    downramp_beam_out = stage2.downramp.beam_out
    downramp_beam_out.stage_number += 1  # There is a mismatch in stage number, as it is not added until after the stage.
    _, beam_outgoing = stage2.undo_beam_coordinate_systems_rotation(stage2.driver_incoming, stage2.downramp.driver_out, downramp_beam_out)
    Beam.comp_beams(final_beam, beam_outgoing, comp_location=True, rtol=1e-11, atol=0.0)

    # Assert that the propagation length of the output beam matches the total length of the stage
    assert np.allclose(final_beam.location - stage2.upramp.beam_in.location, stage2.length, rtol=1e-15, atol=0.0)


    # ========== No jitter, large angular offset ==========
    driver_source3 = setup_Basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=1e-6, y_angle=1e-5)
    main_source3 = setup_basic_main_source()
    stage3 = setup_StageBasic(driver_source=driver_source3, use_ramps=True, test_beam_between_ramps=True)

    linac = PlasmaLinac(source=main_source3, stage=stage3, num_stages=1)
    linac.run('test_beam_between_ramps', overwrite=True, verbose=False)

    # Assert that there has been no significant changes in the beams between parents and its upramp
    # Between a upramp and main stage
    Beam.comp_beams(stage3.upramp.driver_out, stage3.driver_in, comp_location=True)
    Beam.comp_beams(stage3.upramp.beam_out, stage3.beam_in, comp_location=True)

    # Between a main stage and downramp
    Beam.comp_beams(stage3.driver_out, stage3.downramp.driver_in, comp_location=True, rtol=1e-13, atol=0.0)
    Beam.comp_beams(stage3.beam_out, stage3.downramp.beam_in, comp_location=True, rtol=1e-11, atol=0.0)

    # Assert that the output beam matches the out beam for the downramp
    final_beam = linac[0].get_beam(-1)
    downramp_beam_out = stage3.downramp.beam_out
    downramp_beam_out.stage_number += 1  # There is a mismatch in stage number, as it is not added until after the stage.
    _, beam_outgoing = stage3.undo_beam_coordinate_systems_rotation(stage3.driver_incoming, stage3.downramp.driver_out, downramp_beam_out)
    Beam.comp_beams(final_beam, beam_outgoing, comp_location=True, rtol=1e-11, atol=0.0)

    # Assert that the propagation length of the output beam matches the total length of the stage
    assert np.allclose(final_beam.location - stage3.upramp.beam_in.location, stage3.length, rtol=1e-15, atol=0.0)

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
    driver_source = setup_Basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=x_angle, y_angle=y_angle)
    main_source = setup_basic_main_source()
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=False, return_tracked_driver=True, test_beam_between_ramps=True)

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
    driver_source = setup_Basic_driver_source(enable_xy_jitter=True, enable_xpyp_jitter=True)
    main_source = setup_basic_main_source()
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, return_tracked_driver=True, test_beam_between_ramps=True)

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
    assert np.isclose(driver.norm_emittance_x(), driver0.norm_emittance_x(), rtol=1e-7, atol=0.0)
    assert np.isclose(driver.norm_emittance_y(), driver0.norm_emittance_y(), rtol=1e-7, atol=0.0)
    assert np.isclose(driver.bunch_length(), driver0.bunch_length(), rtol=1e-11, atol=0.0)
    assert np.isclose(driver.peak_current(), driver0.peak_current(), rtol=1e-3, atol=0.0)
    assert np.isclose(driver.particle_mass, driver0.particle_mass, rtol=1e-13, atol=0.0)

    assert np.allclose(driver.qs(), driver0.qs(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.weightings(), driver0.weightings(), rtol=1e-13, atol=0.0)
    assert np.allclose(driver.xs(), driver0.xs(), rtol=0.0, atol=1e-8)
    assert np.allclose(driver.ys(), driver0.ys(), rtol=0.0, atol=1e-8)
    assert np.allclose(driver.zs(), driver0.zs(), rtol=0.0, atol=1e-8)

    # Deplete driver0 energy depletion for ramps and stage to be comparable to driver
    driver0.set_Es(driver0.Es()*(1-stage.depletion_efficiency))
    driver0.set_Es(driver0.Es()*(1-stage.depletion_efficiency))
    driver0.set_Es(driver0.Es()*(1-stage.depletion_efficiency))

    assert np.allclose(driver.uzs(), driver0.uzs(), rtol=1e-6, atol=0.0)


    # ========== No jitter, no angular offset ==========
    driver_source = setup_Basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False)
    main_source = setup_basic_main_source()
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, return_tracked_driver=True, test_beam_between_ramps=True)

    stage.nom_energy = 36.9e9                                                     # [eV], HALHF v2 last stage nominal input energy 
    _, driver = stage.track(main_source.track())
    driver0 = stage.driver_incoming

    # Cannot compare the whole phase space due to driver evolution
    assert np.isclose(driver.x_offset(), driver0.x_offset(), rtol=0.0, atol=1e-20)
    assert np.isclose(driver.y_offset(), driver0.y_offset(), rtol=0.0, atol=1e-20)
    assert np.isclose(driver.z_offset(), driver0.z_offset(), rtol=1e-15, atol=0.0)
    assert np.isclose(driver.x_angle(), driver0.x_angle(), rtol=0.0, atol=1e-20)
    assert np.isclose(driver.y_angle(), driver0.y_angle(), rtol=0.0, atol=1e-20)
    assert np.isclose(driver.norm_emittance_x(), driver0.norm_emittance_x(), rtol=1e-13, atol=0.0)
    assert np.isclose(driver.norm_emittance_y(), driver0.norm_emittance_y(), rtol=1e-15, atol=0.0)
    assert np.isclose(driver.bunch_length(), driver0.bunch_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(driver.peak_current(), driver0.peak_current(), rtol=1e-15, atol=0.0)
    assert np.isclose(driver.particle_mass, driver0.particle_mass, rtol=1e-13, atol=0.0)

    assert np.allclose(driver.qs(), driver0.qs(), rtol=1e-15, atol=0.0)
    assert np.allclose(driver.weightings(), driver0.weightings(), rtol=1e-13, atol=0.0)


    # ========== No jitter, large angular offset ==========
    x_angle = 5e-6                                                                # [rad]
    y_angle = 2e-5                                                                # [rad]
    driver_source2 = setup_Basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=x_angle, y_angle=y_angle)
    main_source2 = setup_basic_main_source()
    stage2 = setup_StageBasic(driver_source=driver_source2, use_ramps=True, return_tracked_driver=True, test_beam_between_ramps=True)

    stage2.nom_energy = 36.9e9                                                     # [eV], HALHF v2 last stage nominal input energy 
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
    assert np.isclose(driver.x_angle(), driver0.x_angle(), rtol=1e-5, atol=0.0)
    assert np.isclose(driver.y_angle(), driver0.y_angle(), rtol=1e-4, atol=0.0)

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
    assert np.allclose(driver.uzs(), driver0.uzs(), rtol=1e-6, atol=0.0)
    