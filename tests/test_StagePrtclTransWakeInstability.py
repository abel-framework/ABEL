# -*- coding: utf-8 -*-
"""
ABEL : StagePrtclTransWakeInstability tests
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

def setup_trapezoid_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False):
    driver = SourceTrapezoid()
    driver.current_head = 0.1e3                                                   # [A]
    driver.bunch_length = 1050e-6                                                 # [m] This value is for trapezoid.
    driver.z_offset = 1615e-6                                                     # [m]

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


def setup_basic_main_source(plasma_density, ramp_beta_mag):
    from abel.utilities.plasma_physics import beta_matched

    main = SourceBasic()
    main.bunch_length = 40.0e-06                                                  # [m], rms. Standard value
    main.num_particles = 10000                                               
    main.charge = -SI.e * 1.0e10                                                     # [C]

    # Energy parameters
    main.energy = 369.6e9                                                         # [eV], HALHF v2 last stage nominal input energy
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


def setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability=True, enable_radiation_reaction=True, enable_ion_motion=False, use_ramps=False, drive_beam_update_period=0, return_tracked_driver=False, run_tests=False):

    stage = StagePrtclTransWakeInstability()
    stage.time_step_mod = 0.03                                                    # In units of betatron wavelengths/c.
    stage.nom_energy_gain = 7.8e9                                                 # [eV]
    stage.length_flattop = 7.8                                                    # [m]
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

    stage._return_tracked_driver = return_tracked_driver
    stage.run_tests = run_tests

    # Set up ramps after the stage is fully configured
    if use_ramps:
        stage.upramp = stage.stage2ramp() 
        stage.downramp = stage.stage2ramp()
    
    return stage


@pytest.mark.StagePrtclTransWakeInstability
def test_beam_between_ramps():
    "Tests for ensuring that the beams are correctly transferred between ramps and stage."

    np.random.seed(42)

    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = True
    enable_xpyp_jitter = True
    enable_tr_instability = True
    enable_radiation_reaction = True
    enable_ion_motion = False
    use_ramps = True
    drive_beam_update_period = 1

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period, run_tests=True)

    linac = PlasmaLinac(source=main_source, stage=stage, num_stages=1)
    linac.run('test_beam_between_ramps', overwrite=True, verbose=False)

    # Assert that there has been no significant changes in the beams between parents and its upramp
    # Between a upramp and main stage
    Beam.comp_beams(stage.upramp.driver_out, stage.driver_in, comp_location=True)
    Beam.comp_beams(stage.upramp.beam_out, stage.beam_in, comp_location=True)

    # Between a main stage and downramp
    Beam.comp_beams(stage.driver_out, stage.downramp.driver_in, comp_location=True, rtol=1e-13, atol=0.0)
    Beam.comp_beams(stage.beam_out, stage.downramp.beam_in, comp_location=True, rtol=1e-11, atol=0.0)

    # Assert that the output beam matches the out beam for the downramp
    final_beam = linac[0].get_beam(-1)
    Beam.comp_beams(final_beam, stage.downramp.beam_out, comp_location=True)

    # Assert that the propagation length of the output beam matches the total length of the stage
    assert np.allclose(final_beam.location - stage.upramp.beam_in.location, stage.length, rtol=1e-15, atol=0.0)

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StagePrtclTransWakeInstability
def test_stage_length_gradient_energyGain():
    "Tests ensuring that the flattop length and total length of the stage as well as nominal gradient and nominal energy gain are set correctly."

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
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period, run_tests=False)

    stage.length_flattop = 7.8                                                    # [m]
    stage.nom_energy_gain = 7.8e9                                                 # [eV]

    linac = PlasmaLinac(source=main_source, stage=stage, num_stages=1)
    linac.run('test_stage_length_gradient_energyGain', overwrite=True, verbose=False)
    assert np.allclose(stage.nom_energy_gain_flattop, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length_flattop, 7.8, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient_flattop, 1.0e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length, stage.length_flattop + stage.upramp.length_flattop + stage.downramp.length_flattop, rtol=1e-15, atol=0.0)

    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period, run_tests=False)

    stage.nom_energy_gain = 7.8e9                                                 # [eV]
    stage.nom_accel_gradient_flattop = 1.0e9                                      # [V/m]
    linac = PlasmaLinac(source=main_source, stage=stage, num_stages=1)
    linac.run('test_stage_length_gradient_energyGain', overwrite=True, verbose=False)
    assert np.allclose(stage.nom_energy_gain_flattop, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length_flattop, 7.8, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.nom_accel_gradient_flattop, 1.0e9, rtol=1e-15, atol=0.0)
    assert np.allclose(stage.length, stage.length_flattop + stage.upramp.length_flattop + stage.downramp.length_flattop, rtol=1e-15, atol=0.0)

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StagePrtclTransWakeInstability
def test_driver_unrotation():
    "Tests for checking the driver being correctly un-rotated back to its original coordinate system."

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

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period, return_tracked_driver, run_tests=True)

    stage.nom_energy = 369.6e9                                                    # [eV], HALHF v2 last stage nominal input energy
    _, driver = stage.track(main_source.track())
    driver0 = stage.driver_incoming

    assert np.allclose(driver0.x_angle(), driver.x_angle(), rtol=1e-15, atol=0.0)
    assert np.allclose(driver0.y_angle(), driver.y_angle(), rtol=1e-13, atol=0.0)
   

# TODO: Test on bubble radius tracing

