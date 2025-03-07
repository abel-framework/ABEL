# -*- coding: utf-8 -*-
"""
ABEL : StagePrtclTransWakeInstability beamline calculation tests
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
    main = SourceBasic()
    main.bunch_length = 40.0e-06                                                  # [m], rms. Standard value
    main.num_particles = 10000                                               
    main.charge = -e * 1.0e10                                                     # [C]

    # Energy parameters
    main.energy = 3e9                                                             # [eV]
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


def setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability=True, enable_radiation_reaction=True, enable_ion_motion=False, use_ramps=False, drive_beam_update_period=0):
    stage = StagePrtclTransWakeInstability()
    stage.time_step_mod = 0.03                                                    # In units of betatron wavelengths/c.
    stage.nom_energy_gain = 7.8e9                                                 # [eV]
    stage.length = 7.8                                                            # [m]
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

    # Set up ramps after the stage is fully configured
    if use_ramps:
        stage.upramp = stage.stage2ramp() 
        stage.downramp = stage.stage2ramp()
    
    return stage


def setup_InterstageElegant(stage):
    interstage = InterstageElegant()
    interstage.save_evolution = False
    interstage.save_apl_field_map = False
    interstage.enable_isr = True
    interstage.enable_csr = True
    interstage.beta0 = lambda energy: stage.matched_beta_function(energy)
    interstage.dipole_length = lambda energy: 1 * np.sqrt(energy/10e9)           # [m(eV)], energy-dependent length
    interstage.dipole_field = lambda energy: np.min([0.52, 40e9/energy])         # [T]

    return interstage


@pytest.mark.transverse_wake_instability_linac
def test_baseline_linac():
    "All ``StagePrtclTransWakeInstability`` physics effects disabled, no driver evolution, no driver jitter, no ramps."

    num_stages = 2
    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = False
    enable_xpyp_jitter = False
    enable_tr_instability = False
    enable_radiation_reaction = False
    enable_ion_motion = False
    use_ramps = False
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_baseline_linac', overwrite=True, verbose=False)



@pytest.mark.transverse_wake_instability_linac
def test_ramped_linac():
    "All ``StagePrtclTransWakeInstability`` physics effects disabled, no driver evolution, no driver jitter, with ramps."

    num_stages = 2
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
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_ramped_linac', overwrite=True, verbose=False)



###################################################
# Drive beam evolution tests

@pytest.mark.transverse_wake_instability_linac
def test_driverEvol_linac():
    "All ``StagePrtclTransWakeInstability`` physics effects disabled, driver evolution, driver angular jitter, no ramps."

    num_stages = 2
    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = False
    enable_xpyp_jitter = False
    enable_tr_instability = False
    enable_radiation_reaction = False
    enable_ion_motion = False
    use_ramps = False
    drive_beam_update_period = 1

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_driverEvol_linac', overwrite=True, verbose=False)


@pytest.mark.transverse_wake_instability_linac
def test_driverEvol_ramped_linac():
    "All ``StagePrtclTransWakeInstability`` physics effects disabled, driver evolution, driver angular jitter, with ramps."

    num_stages = 2
    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = False
    enable_xpyp_jitter = False
    enable_tr_instability = False
    enable_radiation_reaction = False
    enable_ion_motion = False
    use_ramps = True
    drive_beam_update_period = 1

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_driverEvol_ramped_linac', overwrite=True, verbose=False)



###################################################
# Angular jitter tests

@pytest.mark.transverse_wake_instability_linac
def test_angular_jitter_linac():
    "All ``StagePrtclTransWakeInstability`` physics effects disabled, no driver evolution, driver angular jitter, no ramps."

    num_stages = 2
    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = False
    enable_xpyp_jitter = True
    enable_tr_instability = False
    enable_radiation_reaction = False
    enable_ion_motion = False
    use_ramps = False
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_angular_jitter_linac', overwrite=True, verbose=False)


@pytest.mark.transverse_wake_instability_linac
def test_angular_jitter_ramped_linac():
    "All ``StagePrtclTransWakeInstability`` physics effects disabled, no driver evolution, driver angular jitter, with ramps."

    num_stages = 2
    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = False
    enable_xpyp_jitter = True
    enable_tr_instability = False
    enable_radiation_reaction = False
    enable_ion_motion = False
    use_ramps = True
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_angular_jitter_ramped_linac', overwrite=True, verbose=False)



###################################################
# Physics effects tests

@pytest.mark.transverse_wake_instability_linac
def test_trInstability_linac():
    "``StagePrtclTransWakeInstability`` transverse instability enabled, radiation reaction enabled, no driver evolution, no driver jitter, no ramps."

    num_stages = 2
    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = False
    enable_xpyp_jitter = False
    enable_tr_instability = True
    enable_radiation_reaction = True
    enable_ion_motion = False
    use_ramps = False
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_trInstability_linac', overwrite=True, verbose=False)


@pytest.mark.transverse_wake_instability_linac
def test_jitter_trInstability_ramped_linac():
    "``StagePrtclTransWakeInstability`` transverse instability enabled, radiation reaction enabled, no driver evolution, xy driver jitter, with ramps."

    num_stages = 2
    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = True
    enable_xpyp_jitter = False
    enable_tr_instability = True
    enable_radiation_reaction = True
    enable_ion_motion = False
    use_ramps = True
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_jitter_trInstability_ramped_linac', overwrite=True, verbose=False)


@pytest.mark.transverse_wake_instability_linac
def test_driverEvol_jitter_trInstability_ramped_linac():
    "``StagePrtclTransWakeInstability`` transverse instability enabled, radiation reaction enabled, with driver evolution, xy driver jitter, with ramps."

    num_stages = 2
    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = True
    enable_xpyp_jitter = False
    enable_tr_instability = True
    enable_radiation_reaction = True
    enable_ion_motion = False
    use_ramps = True
    drive_beam_update_period = 1

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_driverEvol_jitter_trInstability_ramped_linac', overwrite=True, verbose=False)



@pytest.mark.transverse_wake_instability_linac
def test_ionMotion_linac():
    "``StagePrtclTransWakeInstability`` ion motion enabled, no driver evolution, no driver jitter, no ramps."

    num_stages = 2
    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = False
    enable_xpyp_jitter = False
    enable_tr_instability = False
    enable_radiation_reaction = True
    enable_ion_motion = False
    use_ramps = False
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_ionMotion_linac', overwrite=True, verbose=False)


@pytest.mark.transverse_wake_instability_linac
def test_jitter_trInstability_ionMotion_linac():
    "``StagePrtclTransWakeInstability`` transverse instability enabled, radiation reaction enabled, ion motion enabled, driver xy jitter, no driver evolution, no ramps."

    num_stages = 2
    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = True
    enable_xpyp_jitter = False
    enable_tr_instability = True
    enable_radiation_reaction = True
    enable_ion_motion = True
    use_ramps = False
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_trInstability_ionMotion_linac', overwrite=True, verbose=False)


@pytest.mark.transverse_wake_instability_linac
def test_driverEvol_jitter_trInstability_ionMotion_ramped_linac():
    "``StagePrtclTransWakeInstability`` transverse instability enabled, radiation reaction enabled, ion motion enabled, driver xy jitter, no driver evolution, no ramps."

    num_stages = 2
    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = True
    enable_xpyp_jitter = False
    enable_tr_instability = True
    enable_radiation_reaction = True
    enable_ion_motion = True
    use_ramps = True
    drive_beam_update_period = 1

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StagePrtclTransWakeInstability(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_trInstability_ionMotion_linac', overwrite=True, verbose=False)