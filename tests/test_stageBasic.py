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


def setup_Basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False):
    driver = SourceBasic()
    driver.bunch_length = 42e-6                                                   # [m] This value is for trapezoid.
    driver.z_offset = 300e-6                                                      # [m]

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

    #driver.symmetrize = True

    return driver


def setup_basic_main_source(plasma_density, ramp_beta_mag):
    from abel.utilities.plasma_physics import beta_matched

    main = SourceBasic()
    main.bunch_length = 18e-6                                                     # [m], rms. Standard value
    main.num_particles = 10000                                               
    main.charge = -SI.e * 1.0e10                                                  # [C]

    # Energy parameters
    main.energy = 5e9                                                             # [eV]
    main.rel_energy_spread = 0.01                                                 # Relative rms energy spread

    # Emittances
    main.emit_nx, main.emit_ny = 160e-6, 0.56e-6                                  # [m rad]

    # Beta functions
    main.beta_x = beta_matched(plasma_density, main.energy) * ramp_beta_mag       # [m]
    main.beta_y = main.beta_x                                                     # [m]

    # Offsets
    main.z_offset = -36e-6                                                        # [m] # Standard value

    # Other
    #main.symmetrize_6d = True

    return main


def setup_StageBasic(nom_accel_gradient=6.4e9, nom_energy_gain=31.9e9, plasma_density=7e21, driver_source=None, ramp_beta_mag=10, use_ramps=False, test_beam_between_ramps=False, transformer_ratio=1, depletion_efficiency=0.75, calc_evolution=False):
    
    stage = StageBasic()
    stage.nom_accel_gradient = nom_accel_gradient                                 # [GV/m]
    stage.nom_energy_gain = nom_energy_gain                                       # [eV]
    stage.plasma_density = plasma_density                                         # [m^-3]
    stage.driver_source = driver_source
    stage.ramp_beta_mag = ramp_beta_mag
    stage.test_beam_between_ramps = test_beam_between_ramps
    stage.transformer_ratio = transformer_ratio
    stage.depletion_efficiency = depletion_efficiency
    stage.calc_evolution = calc_evolution

    # Set up ramps after the stage is fully configured
    if use_ramps:
        stage.upramp = stage.stage2ramp() 
        stage.downramp = stage.stage2ramp()
    
    return stage


@pytest.mark.StageBasic
def test_beam_between_ramps():
    "Tests for ensuring that the beams are correctly transferred between ramps and stage."

    np.random.seed(42)

    nom_accel_gradient = 6.4e9
    nom_energy_gain = 31.9e9
    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 10.0
    enable_xy_jitter = True
    enable_xpyp_jitter = True
    use_ramps = True

    driver_source = setup_Basic_driver_source(enable_xy_jitter, enable_xpyp_jitter)
    main_source = setup_basic_main_source(plasma_density, ramp_beta_mag)
    stage = setup_StageBasic(nom_accel_gradient, nom_energy_gain, plasma_density, driver_source, ramp_beta_mag, use_ramps=use_ramps, test_beam_between_ramps=True)

    linac = PlasmaLinac(source=main_source, stage=stage, num_stages=1)
    linac.run('test_beam_between_ramps', overwrite=True, verbose=False)

    # Assert that there has been no significant changes in the beams between parents and its upramp
    # Between a upramp and main stage
    Beam.comp_beams(stage.upramp.driver_out, stage.driver_in, comp_location=True)
    Beam.comp_beams(stage.upramp.beam_out, stage.beam_in, comp_location=True)

    # Between a main stage and downramp
    Beam.comp_beams(stage.driver_out, stage.downramp.driver_in, comp_location=True, rtol=1e-14, atol=0.0)
    Beam.comp_beams(stage.beam_out, stage.downramp.beam_in, comp_location=True, rtol=1e-11, atol=0.0)

    # Assert that the output beam matches the out beam for the downramp
    final_beam = linac[0].get_beam(-1)
    Beam.comp_beams(final_beam, stage.downramp.beam_out, comp_location=True)

    # Assert that the propagation length of the output beam matches the total length of the stage
    assert np.allclose(final_beam.location - stage.upramp.beam_in.location, stage.length, rtol=1e-15, atol=0.0)

    # Remove output directory
    shutil.rmtree(linac.run_path())
