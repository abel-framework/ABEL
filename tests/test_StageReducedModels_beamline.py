# -*- coding: utf-8 -*-
"""
ABEL : StageReducedModels beamline integration tests
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
import matplotlib
import os, shutil
matplotlib.use('Agg')  # Use a backend that does not display figure to suppress plots.

def setup_trapezoid_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False):
    driver = SourceTrapezoid()
    driver.current_head = 0.1e3                                                     # [A]
    driver.bunch_length = 1050e-6                                                   # [m] This value is for trapezoid.
    driver.z_offset = 1615e-6                                                       # [m]

    driver.num_particles = 30000                                                 
    driver.charge = 5.0e10 * -SI.e                                                  # [C]
    driver.energy = 4.9e9                                                           # [eV] 
    driver.gaussian_blur = 50e-6                                                    # [m]
    driver.rel_energy_spread = 0.01                                              

    driver.emit_nx, driver.emit_ny = 50e-6, 100e-6                                  # [m rad]
    driver.beta_x, driver.beta_y = 0.5, 0.5                                         # [m]

    if enable_xy_jitter:
        driver.jitter.x = 100e-9                                                    # [m], std
        driver.jitter.y = 100e-9                                                    # [m], std

    if enable_xpyp_jitter:
        driver.jitter.xp = 1.0e-6                                                   # [rad], std
        driver.jitter.yp = 1.0e-6                                                   # [rad], std

    driver.symmetrize = True

    return driver


def setup_basic_main_source(plasma_density, ramp_beta_mag=1.0, energy=361.8e9): # Large default energy for faster tracking
    from abel.utilities.plasma_physics import beta_matched

    main = SourceBasic()
    main.bunch_length = 40.0e-06                                                    # [m], rms. Standard value
    main.num_particles = 10000                                               
    main.charge = -SI.e * 1.0e10                                                    # [C]

    # Energy parameters
    main.energy = energy                                                            # [eV], Default set to HALHF v2 second to last stage nominal input energy
    main.rel_energy_spread = 0.02                                                   # Relative rms energy spread

    # Emittances
    #main.emit_nx, main.emit_ny = 90.0e-6, 0.32e-6                                   # [m rad], budget value
    main.emit_nx, main.emit_ny = 15e-6, 0.1e-6                                      # [m rad]

    # Beta functions
    main.beta_x = beta_matched(plasma_density, main.energy) * ramp_beta_mag         # [m]
    main.beta_y = main.beta_x                                                       # [m]

    # Offsets
    main.z_offset = 0.00e-6                                                         # [m] # Standard value

    # Other
    main.symmetrize_6d = True

    return main


def setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability=True, enable_radiation_reaction=True, enable_ion_motion=False, use_ramps=False, drive_beam_update_period=0, save_final_step=False):
    
    stage = StageReducedModels()
    stage.time_step_mod = 0.03                                                      # In units of betatron wavelengths/c.
    stage.nom_energy_gain = 7.8e9                                                   # [eV]
    stage.length_flattop = 7.8                                                      # [m]
    stage.plasma_density = plasma_density                                           # [m^-3]
    stage.driver_source = driver_source
    stage.main_source = main_source
    stage.ramp_beta_mag = ramp_beta_mag
    stage.enable_tr_instability = enable_tr_instability 
    stage.enable_radiation_reaction = enable_radiation_reaction

    stage.enable_ion_motion = enable_ion_motion
    stage.ion_charge_num = 1.0
    ion_mass = 4.002602 * SI.physical_constants['atomic mass constant'][0]          # [kg], He mass
    stage.ion_mass = ion_mass
    stage.num_z_cells_main = 51
    stage.num_y_cells_rft = 50
    stage.num_x_cells_rft = 50
    stage.num_xy_cells_probe = 41
    stage.ion_wkfld_update_period = 1  # Updates the ion wakefield perturbation every nth time step.
    stage.drive_beam_update_period = drive_beam_update_period  # Updates the drive beam every nth time step.

    stage.probe_evol_period = 3
    stage.save_final_step = save_final_step
    stage.make_animations = False

    # Set up ramps after the stage is fully configured
    if use_ramps:
        stage.upramp = PlasmaRamp()
        stage.downramp = PlasmaRamp()
    
    return stage


def setup_InterstageElegant(stage):
    interstage = InterstageElegant()
    interstage.save_evolution = False
    interstage.save_apl_field_map = False
    interstage.enable_isr = True
    interstage.enable_csr = True
    interstage.beta0 = lambda energy: stage.matched_beta_function(energy)
    interstage.dipole_length = lambda energy: 1 * np.sqrt(energy/10e9)              # [m(eV)], energy-dependent length
    interstage.dipole_field = lambda energy: np.min([0.52, 40e9/energy])            # [T]

    return interstage


@pytest.mark.StageReducedModels_linac
def test_baseline_linac():
    """
    All ``StageReducedModels`` physics effects disabled, no driver 
    evolution, no driver jitter, no ramps. Also tests some plotting functions.
    """

    np.random.seed(42)

    num_stages = 2
    plasma_density = 6.0e+20                                                        # [m^-3]
    #ramp_beta_mag = 5.0
    enable_xy_jitter = False
    enable_xpyp_jitter = False
    enable_tr_instability = False
    enable_radiation_reaction = False
    enable_ion_motion = False
    use_ramps = False
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag=1.0)
    stage = setup_StageReducedModels(plasma_density, driver_source, main_source, 1.0, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_baseline_linac', overwrite=True, verbose=False)
    
    # Check the machine parameters
    stages = linac.stages
    assert len(stages) == num_stages
    assert np.isclose(linac.nom_energy, 377400000000.0, rtol=1e-5, atol=0.0)
    assert np.isclose(linac.nom_energy, stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=1e-5, atol=0.0)
    assert np.isclose(linac.get_length(), 48.559, rtol=1e-4, atol=0.0)

    assert np.isclose(stages[0].nom_energy, 361.8e9)
    assert np.isclose(stages[0].nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[1].nom_energy, stages[0].nom_energy + stages[0].nom_energy_gain)
    assert np.isclose(stages[1].nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)

    interstage = linac.interstages[0]
    interstage_nom_energy = interstage.nom_energy
    assert np.isclose(interstage.nom_energy, stages[1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstage.beta0, stage.matched_beta_function(interstage_nom_energy), rtol=1e-15, atol=0.0)
    assert np.isclose(interstage.dipole_length, np.sqrt(interstage_nom_energy/10e9), rtol=1e-15, atol=0.0)
    assert np.isclose(interstage.dipole_field, np.min([0.52, 40e9/interstage_nom_energy]), rtol=1e-15, atol=0.0)

    # Check the initial and final beams
    initial_beam = linac.get_beam(0)
    assert np.isclose(initial_beam.energy(), stages[0].nom_energy, rtol=1e-3, atol=0.0)
    assert np.isclose(initial_beam.beta_x(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)
    assert np.isclose(initial_beam.beta_y(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)

    final_beam = linac.get_beam(-1)
    assert final_beam.stage_number == 2
    assert np.isclose(final_beam.location, linac.get_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.energy(), stages[1].nom_energy + stages[1].nom_energy_gain, rtol=1e-2, atol=0.0)
    assert np.isclose(final_beam.bunch_length(), main_source.bunch_length, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.charge(), main_source.charge, rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.rel_energy_spread(), 0.0192, rtol=1e-2, atol=0.0)

    nom_beam_size_x = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_x()
    nom_beam_size_y = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_y()
    assert np.isclose(final_beam.beam_size_x(), nom_beam_size_x, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.beam_size_y(), nom_beam_size_y, rtol=1e-1, atol=0.0)
    nom_beta_x = np.sqrt(final_beam.energy()/main_source.energy) * initial_beam.beta_x()
    nom_beta_y = np.sqrt(final_beam.energy()/main_source.energy) * initial_beam.beta_y()
    assert np.isclose(final_beam.beta_x(), nom_beta_x, rtol=0.2, atol=0.0)
    assert np.isclose(final_beam.beta_y(), nom_beta_y, rtol=0.2, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_x(), main_source.emit_nx, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_y(), main_source.emit_ny, rtol=0.2, atol=0.0)

    # Test plotting functions
    linac.stages[-1].plot_Ez_rb_cut()
    linac.stages[-1].plot_wake()

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageReducedModels_linac
def test_ramped_linac():
    """
    All ``StageReducedModels`` physics effects disabled, no driver 
    evolution, no driver jitter, with ramps. Also tests some plotting functions.
    """

    np.random.seed(42)

    num_stages = 2
    plasma_density = 6.0e+20                                                        # [m^-3]
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
    stage = setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_ramped_linac', overwrite=True, verbose=False)

    # Check the machine parameters
    stages = linac.stages
    assert len(stages) == num_stages
    assert np.isclose(linac.nom_energy, 377400000000.0, rtol=1e-5, atol=0.0)
    assert np.isclose(linac.nom_energy, stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=1e-5, atol=0.0)
    assert np.isclose(linac.get_length(), 52.224, rtol=1e-5, atol=0.0)
    
    assert np.isclose(stages[0].nom_energy, 361.8e9)
    assert np.isclose(stages[0].nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[1].nom_energy, stages[0].nom_energy + stages[0].nom_energy_gain)
    assert np.isclose(stages[1].nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.length, 0.9164935, rtol=1e-5, atol=0.0)
    assert np.isclose(stages[-1].upramp.length_flattop, stages[-1].upramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy, stages[-1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_flattop, stages[-1].upramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_gain_flattop, stages[-1].upramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_accel_gradient_flattop, stages[-1].upramp.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.length, 0.9261138, rtol=1e-5, atol=0.0)
    assert np.isclose(stages[-1].downramp.length_flattop, stages[-1].downramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy, stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_flattop, stages[-1].downramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_gain_flattop, stages[-1].downramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_accel_gradient_flattop, stages[-1].downramp.nom_accel_gradient, rtol=1e-15, atol=0.0)

    interstage = linac.interstages[0]
    assert len(linac.interstages) == num_stages - 1 
    interstage_nom_energy = interstage.nom_energy
    assert np.isclose(interstage.nom_energy, stages[1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstage.beta0, stage.matched_beta_function(interstage_nom_energy), rtol=1e-15, atol=0.0)
    assert np.isclose(interstage.dipole_length, np.sqrt(interstage_nom_energy/10e9), rtol=1e-15, atol=0.0)
    assert np.isclose(interstage.dipole_field, np.min([0.52, 40e9/interstage_nom_energy]), rtol=1e-15, atol=0.0)

    # Check the initial and final beams
    initial_beam = linac.get_beam(0)
    assert np.isclose(initial_beam.energy(), stages[0].nom_energy, rtol=1e-3, atol=0.0)
    assert np.isclose(initial_beam.beta_x(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)
    assert np.isclose(initial_beam.beta_y(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)

    final_beam = linac.get_beam(-1)
    assert final_beam.stage_number == 2
    assert np.isclose(final_beam.location, linac.get_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.energy(), stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=1e-2, atol=0.0)
    assert np.isclose(final_beam.bunch_length(), main_source.bunch_length, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.charge(), main_source.charge, rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.rel_energy_spread(), 0.0192, rtol=1e-2, atol=0.0)

    nom_beam_size_x = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_x()
    nom_beam_size_y = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_y()
    assert np.isclose(final_beam.beam_size_x(), nom_beam_size_x, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.beam_size_y(), nom_beam_size_y, rtol=1e-1, atol=0.0)
    nom_beta_x = np.sqrt(final_beam.energy()/main_source.energy) * initial_beam.beta_x()
    nom_beta_y = np.sqrt(final_beam.energy()/main_source.energy) * initial_beam.beta_y()
    assert np.isclose(final_beam.beta_x(), nom_beta_x, rtol=0.1, atol=0.0)
    assert np.isclose(final_beam.beta_y(), nom_beta_y, rtol=0.1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_x(), main_source.emit_nx, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_y(), main_source.emit_ny, rtol=0.1, atol=0.0)

    # Test plotting
    linac.stages[-1].plot_evolution(bunch='beam')
    linac.plot_survey()
    linac.plot_evolution()
    linac.plot_waterfalls()

    # Remove output directory
    shutil.rmtree(linac.run_path())



###################################################
# Drive beam evolution tests

# @pytest.mark.transverse_wake_instability_linac
# def test_driverEvol_linac():
#     """
#     All ``StageReducedModels`` physics effects disabled, driver 
#     evolution, driver angular jitter, no ramps.
#     """

#     np.random.seed(42)

#     num_stages = 2
#     plasma_density = 6.0e+20                                                        # [m^-3]
#     ramp_beta_mag = 5.0
#     enable_xy_jitter = False
#     enable_xpyp_jitter = False
#     enable_tr_instability = False
#     enable_radiation_reaction = False
#     enable_ion_motion = False
#     use_ramps = False
#     drive_beam_update_period = 1

#     driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
#     main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
#     stage = setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
#     interstage = setup_InterstageElegant(stage)

#     linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

#     # Perform tracking
#     linac.run('test_driverEvol_linac', overwrite=True, verbose=False)

#     # Remove output directory
#     shutil.rmtree(linac.run_path())


# @pytest.mark.transverse_wake_instability_linac
# def test_driverEvol_ramped_linac():
#     """
#     All ``StageReducedModels`` physics effects disabled, driver 
#     evolution, driver angular jitter, with ramps.
#     """

#     np.random.seed(42)

#     num_stages = 2
#     plasma_density = 6.0e+20                                                        # [m^-3]
#     ramp_beta_mag = 5.0
#     enable_xy_jitter = False
#     enable_xpyp_jitter = False
#     enable_tr_instability = False
#     enable_radiation_reaction = False
#     enable_ion_motion = False
#     use_ramps = True
#     drive_beam_update_period = 1

#     driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
#     main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
#     stage = setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
#     interstage = setup_InterstageElegant(stage)

#     linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

#     # Perform tracking
#     linac.run('test_driverEvol_ramped_linac', overwrite=True, verbose=False)

#     # Remove output directory
#     shutil.rmtree(linac.run_path())



###################################################
# Angular jitter tests

@pytest.mark.StageReducedModels_linac
def test_angular_jitter_linac():
    """
    All ``StageReducedModels`` physics effects disabled, no driver 
    evolution, driver angular jitter, no ramps.
    """

    np.random.seed(42)

    num_stages = 2
    plasma_density = 6.0e+20                                                        # [m^-3]
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
    stage = setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_angular_jitter_linac', overwrite=True, verbose=False)

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageReducedModels_linac
def test_angular_jitter_ramped_linac():
    """
    All ``StageReducedModels`` physics effects disabled, no driver 
    evolution, driver angular jitter, with ramps.
    """

    np.random.seed(42)

    num_stages = 2
    plasma_density = 6.0e+20                                                        # [m^-3]
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
    stage = setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_angular_jitter_ramped_linac', overwrite=True, verbose=False)

    # Remove output directory
    shutil.rmtree(linac.run_path())



###################################################
# Physics effects tests

@pytest.mark.StageReducedModels_linac
def test_trInstability_linac():
    """
    ``StageReducedModels`` transverse instability enabled, radiation 
    reaction enabled, no driver evolution, no driver jitter, no ramps.
    """

    np.random.seed(42)

    num_stages = 2
    plasma_density = 6.0e+20                                                        # [m^-3]
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
    main_source.energy = 3.0e9                                                      # [eV], HALHF v2 start energy
    stage = setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_trInstability_linac', overwrite=True, verbose=False)

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageReducedModels_linac
def test_jitter_trInstability_ramped_linac():
    """
    ``StageReducedModels`` transverse instability enabled, radiation 
    reaction enabled, no driver evolution, xy driver jitter, with ramps.
    """

    np.random.seed(42)

    num_stages = 2
    plasma_density = 6.0e+20                                                        # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = True
    enable_xpyp_jitter = False
    enable_tr_instability = True
    enable_radiation_reaction = True
    enable_ion_motion = False
    use_ramps = True
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag, energy=3.0e9)
    stage = setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_jitter_trInstability_ramped_linac', overwrite=True, verbose=False)

    # Check the machine parameters
    stages = linac.stages
    assert len(stages) == num_stages
    assert np.isclose(linac.nom_energy, 18.6e9, rtol=1e-5, atol=0.0)
    assert np.isclose(linac.nom_energy, stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=1e-5, atol=0.0)
    assert np.isclose(linac.get_length(), 21.835477, rtol=1e-5, atol=0.0)
    
    assert np.isclose(stages[0].nom_energy, 3.0e9)
    assert np.isclose(stages[0].nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[1].nom_energy, stages[0].nom_energy + stages[0].nom_energy_gain)
    assert np.isclose(stages[1].nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.length, 0.15666612, rtol=1e-5, atol=0.0)
    assert np.isclose(stages[-1].upramp.length_flattop, stages[-1].upramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy, stages[-1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_flattop, stages[-1].upramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_gain_flattop, stages[-1].upramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_accel_gradient_flattop, stages[-1].upramp.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.length, 0.2055985, rtol=1e-5, atol=0.0)
    assert np.isclose(stages[-1].downramp.length_flattop, stages[-1].downramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy, stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_flattop, stages[-1].downramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_gain_flattop, stages[-1].downramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_accel_gradient_flattop, stages[-1].downramp.nom_accel_gradient, rtol=1e-15, atol=0.0)

    interstage = linac.interstages[-1]
    assert len(linac.interstages) == num_stages - 1 
    interstage_nom_energy = interstage.nom_energy
    assert np.isclose(interstage.nom_energy, stages[-1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstage.beta0, stage.matched_beta_function(interstage_nom_energy), rtol=1e-15, atol=0.0)
    assert np.isclose(interstage.dipole_length, np.sqrt(interstage_nom_energy/10e9), rtol=1e-15, atol=0.0)
    assert np.isclose(interstage.dipole_field, np.min([0.52, 40e9/interstage_nom_energy]), rtol=1e-15, atol=0.0)

    # Check the initial and final beams
    initial_beam = linac.get_beam(0)
    assert np.isclose(initial_beam.energy(), stages[0].nom_energy, rtol=1e-3, atol=0.0)
    assert np.isclose(initial_beam.beta_x(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)
    assert np.isclose(initial_beam.beta_y(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)

    final_beam = linac.get_beam(-1)
    assert final_beam.stage_number == 2
    assert np.isclose(final_beam.location, linac.get_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.energy(), stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=5e-2, atol=0.0)
    assert np.isclose(final_beam.bunch_length(), main_source.bunch_length, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.charge(), main_source.charge, rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.rel_energy_spread(), 0.02165, rtol=1e-2, atol=0.0)

    nom_beam_size_x = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_x()
    nom_beam_size_y = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_y()
    assert np.isclose(final_beam.beam_size_x(), nom_beam_size_x, rtol=0.3, atol=0.0)
    assert np.isclose(final_beam.beam_size_y(), nom_beam_size_y, rtol=1e-1, atol=0.0)
    nom_beta_x = np.sqrt(final_beam.energy()/main_source.energy) * initial_beam.beta_x()
    nom_beta_y = np.sqrt(final_beam.energy()/main_source.energy) * initial_beam.beta_y()
    assert np.isclose(final_beam.beta_x(), nom_beta_x, rtol=0.1, atol=0.0)
    assert np.isclose(final_beam.beta_y(), nom_beta_y, rtol=0.1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_x(), main_source.emit_nx, rtol=0.1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_y(), 1.5467557e-07, rtol=1e-3, atol=0.0)  # Expect emittance growth.

    # Remove output directory
    shutil.rmtree(linac.run_path())


# @pytest.mark.transverse_wake_instability_linac
# def test_driverEvol_jitter_trInstability_ramped_linac():
#     """
#     ``StageReducedModels`` transverse instability enabled, radiation 
#     reaction enabled, with driver evolution, xy driver jitter, with ramps.
#     """

#     np.random.seed(42)

#     num_stages = 2
#     plasma_density = 6.0e+20                                                        # [m^-3]
#     ramp_beta_mag = 5.0
#     enable_xy_jitter = True
#     enable_xpyp_jitter = False
#     enable_tr_instability = True
#     enable_radiation_reaction = True
#     enable_ion_motion = False
#     use_ramps = True
#     drive_beam_update_period = 1

#     driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
#     main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
#     stage = setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
#     interstage = setup_InterstageElegant(stage)

#     linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

#     # Perform tracking
#     linac.run('test_driverEvol_jitter_trInstability_ramped_linac', overwrite=True, verbose=False)

#     # Remove output directory
#     shutil.rmtree(linac.run_path())


@pytest.mark.StageReducedModels_linac
def test_ionMotion_linac():
    """
    ``StageReducedModels`` ion motion enabled, no driver evolution, 
    no driver jitter, no ramps.
    """

    np.random.seed(42)

    num_stages = 2
    plasma_density = 6.0e+20                                                        # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = False
    enable_xpyp_jitter = False
    enable_tr_instability = False
    enable_radiation_reaction = True
    enable_ion_motion = True
    use_ramps = False
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag, energy=361.8e9)
    stage = setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_ionMotion_linac', overwrite=True, verbose=False)

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageReducedModels_linac
def test_jitter_trInstability_ionMotion_linac():
    """
    ``StageReducedModels`` transverse instability enabled, radiation 
    reaction enabled, ion motion enabled, driver xy jitter, no driver evolution, 
    no ramps.
    """

    np.random.seed(42)

    num_stages = 2
    plasma_density = 6.0e+20                                                        # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = True
    enable_xpyp_jitter = False
    enable_tr_instability = True
    enable_radiation_reaction = True
    enable_ion_motion = True
    use_ramps = False
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag, energy=361.8e9)
    stage = setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_trInstability_ionMotion_linac', overwrite=True, verbose=False)

    # Remove output directory
    shutil.rmtree(linac.run_path())


@pytest.mark.StageReducedModels_linac
def test_jitter_trInstability_ionMotion_ramped_linac():
    """
    ``StageReducedModels`` transverse instability enabled, radiation 
    reaction enabled, ion motion enabled, driver xy jitter, no driver evolution, 
    with ramps.
    """
    
    np.random.seed(42)

    num_stages = 2
    plasma_density = 6.0e+20                                                        # [m^-3]
    ramp_beta_mag = 5.0
    enable_xy_jitter = True
    enable_xpyp_jitter = False
    enable_tr_instability = True
    enable_radiation_reaction = True
    enable_ion_motion = True
    use_ramps = True
    drive_beam_update_period = 0

    driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag, energy=81.0e9)  # Choosing an energy that gives a sensible number of time steps.
    stage = setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period, save_final_step=True)
    interstage = setup_InterstageElegant(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_jitter_trInstability_ionMotion_ramped_linac', overwrite=True, verbose=False)

    # Check the machine parameters
    stages = linac.stages
    assert len(stages) == num_stages
    assert np.isclose(linac.nom_energy, 96.6e9, rtol=1e-5, atol=0.0)
    assert np.isclose(linac.nom_energy, stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=1e-5, atol=0.0)
    assert np.isclose(linac.get_length(), 33.5511544, rtol=1e-5, atol=0.0)
    
    assert np.isclose(stages[0].nom_energy, 81.0e9)
    assert np.isclose(stages[0].nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[1].nom_energy, stages[0].nom_energy + stages[0].nom_energy_gain)
    assert np.isclose(stages[1].nom_energy_gain, 7.8e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.length, 0.4492312, rtol=1e-5, atol=0.0)
    assert np.isclose(stages[-1].upramp.length_flattop, stages[-1].upramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy, stages[-1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_flattop, stages[-1].upramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_gain_flattop, stages[-1].upramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_accel_gradient_flattop, stages[-1].upramp.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.length, 0.4685457, rtol=1e-5, atol=0.0)
    assert np.isclose(stages[-1].downramp.length_flattop, stages[-1].downramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy, stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_flattop, stages[-1].downramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_gain, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_gain_flattop, stages[-1].downramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_accel_gradient, 0.0, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_accel_gradient_flattop, stages[-1].downramp.nom_accel_gradient, rtol=1e-15, atol=0.0)

    interstage = linac.interstages[-1]
    assert len(linac.interstages) == num_stages - 1 
    interstage_nom_energy = interstage.nom_energy
    assert np.isclose(interstage.nom_energy, stages[-1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstage.beta0, stage.matched_beta_function(interstage_nom_energy), rtol=1e-15, atol=0.0)
    assert np.isclose(interstage.dipole_length, np.sqrt(interstage_nom_energy/10e9), rtol=1e-15, atol=0.0)
    assert np.isclose(interstage.dipole_field, np.min([0.52, 40e9/interstage_nom_energy]), rtol=1e-15, atol=0.0)

    # Check the initial and final beams
    initial_beam = linac.get_beam(0)
    assert np.isclose(initial_beam.energy(), stages[0].nom_energy, rtol=1e-3, atol=0.0)
    assert np.isclose(initial_beam.beta_x(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)
    assert np.isclose(initial_beam.beta_y(), stages[0].matched_beta_function(stages[0].nom_energy), rtol=1e-1, atol=0.0)

    final_beam = linac.get_beam(-1)
    assert final_beam.stage_number == 2
    assert np.isclose(final_beam.location, linac.get_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.energy(), stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=1e-2, atol=0.0)
    assert np.isclose(final_beam.bunch_length(), main_source.bunch_length, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.charge(), main_source.charge, rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.rel_energy_spread(), 0.01739, rtol=1e-2, atol=0.0)

    nom_beam_size_x = (stages[0].nom_energy/stages[-1].nom_energy)**(1/4)*initial_beam.beam_size_x()
    assert np.isclose(final_beam.beam_size_x(), nom_beam_size_x, rtol=0.01, atol=0.0)
    assert np.isclose(final_beam.beam_size_y(), 8.2e-07, rtol=1e-2, atol=0.0)  # Expect deviation from nominal
    nom_beta_x = np.sqrt(final_beam.energy()/main_source.energy) * initial_beam.beta_x()
    nom_beta_y = np.sqrt(final_beam.energy()/main_source.energy) * initial_beam.beta_y()
    assert np.isclose(final_beam.beta_x(), nom_beta_x, rtol=0.1, atol=0.0)
    assert np.isclose(final_beam.beta_y(), nom_beta_y, rtol=0.1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_x(), main_source.emit_nx, rtol=0.1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_y(), 1.8e-07, rtol=5e-2, atol=0.0)  # Expect emittance growth.

    # Remove output directory
    shutil.rmtree(linac.run_path())


# @pytest.mark.transverse_wake_instability_linac
# def test_driverEvol_jitter_trInstability_ionMotion_ramped_linac():
#     """
#     ``StageReducedModels`` transverse instability enabled, radiation 
#     reaction enabled, ion motion enabled, driver xy jitter, with driver 
#     evolution, with ramps.
#     """

#     np.random.seed(42)

#     num_stages = 2
#     plasma_density = 6.0e+20                                                        # [m^-3]
#     ramp_beta_mag = 5.0
#     enable_xy_jitter = True
#     enable_xpyp_jitter = False
#     enable_tr_instability = True
#     enable_radiation_reaction = True
#     enable_ion_motion = True
#     use_ramps = True
#     drive_beam_update_period = 1

#     driver_source = setup_trapezoid_driver_source(enable_xy_jitter, enable_xpyp_jitter)
#     main_source = setup_basic_main_source(plasma_density, ramp_beta_mag, energy=3.0e9)
#     stage = setup_StageReducedModels(plasma_density, driver_source, main_source, ramp_beta_mag, enable_tr_instability, enable_radiation_reaction, enable_ion_motion, use_ramps, drive_beam_update_period)
#     interstage = setup_InterstageElegant(stage)

#     linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

#     # Perform tracking
#     linac.run('test_driverEvol_jitter_trInstability_ionMotion_ramped_linac', overwrite=True, verbose=False)

#     # Remove output directory
#     shutil.rmtree(linac.run_path())


def test_remove_run_data_dir():
    """
    Not an actual test. Just removes the run_data directory.
    """
    import os
    dir = 'run_data'
    if os.path.isdir(dir):
        # Remove output directory
        shutil.rmtree(dir)