# -*- coding: utf-8 -*-
"""
ABEL : StageBasic beamline calculation tests
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


def setup_Basic_driver_source(enable_xt_jitter=False, enable_xpyp_jitter=False, x_angle=0.0, y_angle=0.0):
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

    driver.symmetrize = True

    return driver


def setup_basic_main_source(plasma_density=7.0e21, ramp_beta_mag=10.0):
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


def setup_StageBasic(driver_source=None, nom_accel_gradient=6.4e9, nom_energy_gain=31.9e9, plasma_density=7.0e21, ramp_beta_mag=10.0, use_ramps=False, transformer_ratio=1, depletion_efficiency=0.75, calc_evolution=False, return_tracked_driver=False, test_beam_between_ramps=False):
    
    stage = StageBasic()
    stage.nom_accel_gradient = nom_accel_gradient                                   # [GV/m]
    stage.nom_energy_gain = nom_energy_gain                                         # [eV]
    stage.plasma_density = plasma_density                                           # [m^-3]
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


def setup_InterstageBasic(stage):
    interstage = InterstageBasic()
    interstage.beta0 = lambda E: stage.matched_beta_function(E)
    interstage.dipole_length = lambda E: 1 * np.sqrt(E/10e9)                        # [m(eV)]
    interstage.dipole_field = lambda E: np.min([1.0, 100e9/E])                      # [T]

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
    
    driver_source = setup_Basic_driver_source(enable_xt_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source()
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=False, calc_evolution=False)
    interstage = setup_InterstageBasic(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_baseline_linac', overwrite=True, verbose=False)
    
    # Check the outputs
    stages = linac.stages
    interstages = linac.interstages
    assert len(stages) == num_stages
    assert len(interstages) == num_stages - 1
    assert np.isclose(linac.get_length(), 79.30149471158896, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].nom_energy, 5.0e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[1].nom_energy, 36.9e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[2].nom_energy, 68.8e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[3].nom_energy, 100.7e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[4].nom_energy, 132.6e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[4].nom_energy_gain, 31.9e9, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].nom_energy, stages[1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[2].nom_energy, stages[3].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].nom_energy, stages[4].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].beta0, 0.24137835827389, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].dipole_length, 1.9209372712298547, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].dipole_field, 1.0, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].beta0, 0.45756933131960525, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].dipole_length, 3.641428291206625, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].dipole_field, 0.7541478129713424, rtol=1e-15, atol=0.0)

    final_beam = linac.get_beam(-1)
    final_beam.beam_name = 'Test beam'

    assert final_beam.stage_number == 5
    assert np.isclose(linac.get_beam(0).energy(), stages[0].nom_energy, rtol=1e-3, atol=0.0)
    assert np.isclose(final_beam.location, linac.get_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.energy(), stages[4].nom_energy + stages[4].nom_energy_gain, rtol=1e-4, atol=0.0)
    assert np.isclose(final_beam.bunch_length(), main_source.bunch_length, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.charge(), main_source.charge, rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.rel_energy_spread(), 0.00030271, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.beam_size_x(), 1.580286797571071e-05, rtol=1e-2, atol=0.0)
    assert np.isclose(final_beam.beam_size_y(), 9.331889547894206e-07, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_x(), main_source.emit_nx, rtol=1e-2, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_y(), main_source.emit_ny, rtol=1e-1, atol=0.0)

    
    #ref_beam = Beam.load('./tests/data/test_StagePrtclTransWakeInstability_beamline/test_baseline_linac/shot_000/beam_003_00048.558626.h5')
    #ref_beam.beam_name = 'Reference beam'

    final_beam.print_summary()
    #ref_beam.print_summary()
    #Beam.comp_beam_params(final_beam, ref_beam, comp_location=True)  # Compare output beam with reference beam file.

    # # plot linac survey
    # linac.plot_survey()

    # # plot beam evolution
    # linac.plot_evolution()
    # linac.plot_waterfalls()

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
    
    driver_source = setup_Basic_driver_source(enable_xt_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source()
    stage = setup_StageBasic(driver_source=driver_source, use_ramps=True, calc_evolution=False)
    interstage = setup_InterstageBasic(stage)

    linac = PlasmaLinac(source=main_source, stage=stage, interstage=interstage, num_stages=num_stages)

    # Perform tracking
    linac.run('test_ramped_linac', overwrite=True, verbose=False)
    
    # Check the outputs
    stages = linac.stages
    interstages = linac.interstages
    assert len(stages) == num_stages
    assert len(interstages) == num_stages - 1
    assert np.isclose(linac.get_length(), 79.30149471158896, rtol=1e-15, atol=0.0)
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
    assert np.isclose(stages[0].upramp.nom_energy_gain, 4618203.641879139, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].upramp.nom_energy_gain_flattop, stages[0].upramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].upramp.nom_accel_gradient, 104636463.79451889, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].upramp.nom_accel_gradient_flattop, stages[0].upramp.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.length, 0.11989973028624575, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.length_flattop, stages[0].downramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.nom_energy, stages[1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.nom_energy_flattop, stages[0].downramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.nom_energy_gain, 12544067.50348952, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.nom_energy_gain_flattop, stages[0].downramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.nom_accel_gradient, 104621315.4403443, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[0].downramp.nom_accel_gradient_flattop, stages[0].downramp.nom_accel_gradient, rtol=1e-15, atol=0.0)

    assert np.isclose(stages[-1].upramp.length, 0.22728814548579593, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.length_flattop, stages[-1].upramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy, stages[-1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_flattop, stages[-1].upramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_gain, 25453396.56685818, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_energy_gain_flattop, stages[-1].upramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_accel_gradient, 111987347.65711243, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].upramp.nom_accel_gradient_flattop, stages[-1].upramp.nom_accel_gradient, rtol=1e-15, atol=0.0)

    assert np.isclose(stages[-1].downramp.length, 0.2531558538336775, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.length_flattop, stages[-1].downramp.length, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy, stages[-1].nom_energy + stages[-1].nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_flattop, stages[-1].downramp.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_gain, 28327631.60459319, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_energy_gain_flattop, stages[-1].downramp.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_accel_gradient, 111897991.59534484, rtol=1e-15, atol=0.0)
    assert np.isclose(stages[-1].downramp.nom_accel_gradient_flattop, stages[-1].downramp.nom_accel_gradient, rtol=1e-15, atol=0.0)

    assert np.isclose(interstages[0].nom_energy, stages[1].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[2].nom_energy, stages[3].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].nom_energy, stages[4].nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].beta0, 0.24137835827389, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].dipole_length, 1.9209372712298547, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[0].dipole_field, 1.0, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].beta0, 0.45756933131960525, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].dipole_length, 3.641428291206625, rtol=1e-15, atol=0.0)
    assert np.isclose(interstages[3].dipole_field, 0.7541478129713424, rtol=1e-15, atol=0.0)

    final_beam = linac.get_beam(-1)
    final_beam.beam_name = 'Test beam'

    assert final_beam.stage_number == 5
    assert np.isclose(linac.get_beam(0).energy(), stages[0].nom_energy, rtol=1e-3, atol=0.0)
    assert np.isclose(final_beam.location, linac.get_length(), rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.energy(), stages[-1].nom_energy + stages[-1].nom_energy_gain + stages[-1].upramp.nom_energy_gain + stages[-1].downramp.nom_energy_gain, rtol=1e-3, atol=0.0)
    assert np.isclose(final_beam.bunch_length(), main_source.bunch_length, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.charge(), main_source.charge, rtol=1e-15, atol=0.0)
    assert np.isclose(final_beam.rel_energy_spread(), 0.00030198, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.beam_size_x(), 1.559091953307779e-05, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.beam_size_y(), 9.405771185853757e-07, rtol=1e-1, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_x(), main_source.emit_nx, rtol=1e-2, atol=0.0)
    assert np.isclose(final_beam.norm_emittance_y(), main_source.emit_ny, rtol=1e-1, atol=0.0)

    
    #ref_beam = Beam.load('./tests/data/test_StagePrtclTransWakeInstability_beamline/test_baseline_linac/shot_000/beam_003_00048.558626.h5')
    #ref_beam.beam_name = 'Reference beam'

    final_beam.print_summary()
    #ref_beam.print_summary()
    #Beam.comp_beam_params(final_beam, ref_beam, comp_location=True)  # Compare output beam with reference beam file.

    # Remove output directory
    shutil.rmtree(linac.run_path())

    