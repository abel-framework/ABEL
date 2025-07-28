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
        stage.nom_accel_gradient = nom_accel_gradient                               # [GV/m]
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