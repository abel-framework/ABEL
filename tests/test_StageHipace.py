# -*- coding: utf-8 -*-
# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later


"""
ABEL : StageHipace unit tests
"""

import pytest
from abel import *
#import shutil
import numpy as np


def setup_trapezoid_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=0.0, y_angle=0.0):
    driver = SourceTrapezoid()
    driver.current_head = 0.1e3                                                   # [A]
    driver.bunch_length = 1050e-6                                                 # [m]
    driver.z_offset = 1602e-6                                                     # [m]
    driver.x_angle = x_angle                                                      # [rad]
    driver.y_angle = y_angle                                                      # [rad]

    driver.num_particles = 1000000                                                 
    driver.charge = 5.0e10 * -SI.e                                                # [C]
    driver.energy = 50e9                                                          # [eV] 
    driver.gaussian_blur = 50e-6                                                  # [m]
    driver.rel_energy_spread = 0.01                                              

    driver.emit_nx = 50e-6 * driver.energy/4.5e9                                  # [m rad]
    driver.emit_ny = 100e-6 * driver.energy/4.5e9                                 # [m rad]
    driver.beta_x, driver.beta_y = 0.5, 0.5                                       # [m]

    if enable_xy_jitter:
        driver.jitter.x = 100e-9                                                  # [m], std
        driver.jitter.y = 100e-9                                                  # [m], std

    if enable_xpyp_jitter:
        driver.jitter.xp = 1.0e-6                                                 # [rad], std
        driver.jitter.yp = 1.0e-6                                                 # [rad], std

    driver.symmetrize = True

    return driver


def setup_minimal_StageHipace(nom_energy=100e9, plasma_density=6e20, external_focusing=False, nom_accel_gradient_flattop=1e9):
    stage = StageHipace()
    stage.nom_energy = nom_energy  # [eV]
    stage.plasma_density = plasma_density  # [m^-3]
    stage.driver_source = setup_trapezoid_driver_source()
    stage.external_focusing = external_focusing
    stage.nom_accel_gradient_flattop = nom_accel_gradient_flattop  # [V/m]

    return stage


@pytest.mark.StageHipace
def test_external_focusing():
    """
    Tests for ``StageHipace.calc_length_num_beta_osc()`` for accurately matching the 
    stage length and external driver guiding field gradient to a desired number 
    of drive beam and main beam betatron oscillations.
    """

    num_beta_osc = 4.0  # The number of betatron oscilations that the main beam is expected to perform.

    stage = setup_minimal_StageHipace()

    # ========== Tests without any external fields ==========
    assert stage.external_focusing is False
    assert stage._external_focusing_gradient is None

    stage.length_flattop = stage.calc_length_num_beta_osc(num_beta_osc, initial_energy=stage.nom_energy, driver_half_oscillations=1.0)
    
    assert np.isclose(num_beta_osc, stage.length_flattop2num_beta_osc(), rtol=1e-5, atol=0.0)
    assert stage._external_focusing_gradient is None
    assert np.isclose(3.440220555221998, stage.length_flattop, rtol=1e-5, atol=0.0)


    stage.length_flattop = stage.calc_length_num_beta_osc(num_beta_osc, driver_half_oscillations=2.0)
    
    assert np.isclose(num_beta_osc, stage.length_flattop2num_beta_osc(), rtol=1e-5, atol=0.0)
    assert stage._external_focusing_gradient is None
    assert np.isclose(3.440220555221998, stage.length_flattop, rtol=1e-5, atol=0.0)


    # ========== Tests with external fields ==========
    stage = setup_minimal_StageHipace(external_focusing=True)
    stage.length_flattop = stage.calc_length_num_beta_osc(num_beta_osc, driver_half_oscillations=1.0)
    assert stage._external_focusing_gradient is None
    stage._external_focusing_gradient = 140.1695315279373
    
    assert np.isclose(num_beta_osc, stage.length_flattop2num_beta_osc(), rtol=1e-10, atol=0.0)
    assert np.isclose(3.4268706541720553, stage.length_flattop, rtol=1e-5, atol=0.0)


    stage = setup_minimal_StageHipace(external_focusing=True)
    stage.length_flattop = stage.calc_length_num_beta_osc(num_beta_osc, driver_half_oscillations=2.0)
    assert stage._external_focusing_gradient is None
    stage._external_focusing_gradient = 574.1247168375805
    
    assert np.isclose(num_beta_osc, stage.length_flattop2num_beta_osc(), rtol=1e-10, atol=0.0)
    assert np.isclose(3.3865024719148242, stage.length_flattop, rtol=1e-5, atol=0.0)


    # ========== With external fields, lower stage nominal energy ==========
    num_beta_osc2 = 8.0
    stage = setup_minimal_StageHipace(nom_energy=3e9, external_focusing=True)
    stage.length_flattop = stage.calc_length_num_beta_osc(num_beta_osc2, driver_half_oscillations=1.0)
    assert stage._external_focusing_gradient is None
    stage._external_focusing_gradient = 1038.1202586404845
    
    assert np.isclose(num_beta_osc2, stage.length_flattop2num_beta_osc(), rtol=1e-10, atol=0.0)
    assert np.isclose(1.259217324849329, stage.length_flattop, rtol=1e-5, atol=0.0)


    stage = setup_minimal_StageHipace(nom_energy=3e9, external_focusing=True)
    stage.length_flattop = stage.calc_length_num_beta_osc(num_beta_osc2, driver_half_oscillations=2.0)
    assert stage._external_focusing_gradient is None
    stage._external_focusing_gradient = 5119.8513420067175
    
    assert np.isclose(num_beta_osc2, stage.length_flattop2num_beta_osc(), rtol=1e-10, atol=0.0)
    assert np.isclose(1.13403339406399, stage.length_flattop, rtol=1e-5, atol=0.0)


    # ========== With external fields, lower stage nominal energy, lower driver energy ==========
    stage = setup_minimal_StageHipace(nom_energy=3e9, external_focusing=True)
    stage.driver_source.energy = 4.5e9  # [eV]
    stage.length_flattop = stage.calc_length_num_beta_osc(num_beta_osc2, driver_half_oscillations=1.0)
    assert stage._external_focusing_gradient is None
    stage._external_focusing_gradient = 88.3976616856249
    
    assert np.isclose(num_beta_osc2, stage.length_flattop2num_beta_osc(), rtol=1e-10, atol=0.0)
    assert np.isclose(1.2945695557961696, stage.length_flattop, rtol=1e-5, atol=0.0)


    stage = setup_minimal_StageHipace(nom_energy=3e9, external_focusing=True)
    stage.driver_source.energy = 4.5e9  # [eV]
    stage.length_flattop = stage.calc_length_num_beta_osc(num_beta_osc2, driver_half_oscillations=2.0)
    assert stage._external_focusing_gradient is None
    stage._external_focusing_gradient = 359.3285677857948
    
    assert np.isclose(num_beta_osc2, stage.length_flattop2num_beta_osc(), rtol=1e-10, atol=0.0)
    assert np.isclose(1.2841918239865389, stage.length_flattop, rtol=1e-5, atol=0.0)



@pytest.mark.StageHipace
def test_match_length_guiding_2_num_beta_osc():
    """
    Tests for ``StageHipace.match_length_guiding_2_num_beta_osc()`` for 
    accurately matching the stage length and external driver guiding field 
    gradient to a desired number of drive beam and main beam betatron 
    oscillations.
    """

    num_beta_osc = 4.0  # The number of betatron oscilations that the main beam is expected to perform.
    stage = setup_minimal_StageHipace()

    # ========== Tests without any external fields ==========
    assert stage.external_focusing is False
    assert stage._external_focusing_gradient is None
    assert np.isclose(1.0, stage.driver_half_oscillations, rtol=1e-15, atol=0.0)

    stage.match_length_guiding_2_num_beta_osc(num_beta_osc=num_beta_osc, driver_half_oscillations=1.0, set_consistent_params=True)
    assert np.isclose(1.0, stage.driver_half_oscillations, rtol=1e-15, atol=0.0)
    assert stage._external_focusing_gradient is None
    assert np.isclose(num_beta_osc, stage.length_flattop2num_beta_osc(), rtol=1e-5, atol=0.0)
    assert np.isclose(3.440220555221998, stage.length_flattop, rtol=1e-5, atol=0.0)


    stage.match_length_guiding_2_num_beta_osc(num_beta_osc=num_beta_osc, driver_half_oscillations=2.0, set_consistent_params=True)
    assert np.isclose(1.0, stage.driver_half_oscillations, rtol=1e-15, atol=0.0) # Should still be the default value 1.0, since stage.external_focusing is False.
    assert np.isclose(num_beta_osc, stage.length_flattop2num_beta_osc(), rtol=1e-5, atol=0.0)
    assert stage._external_focusing_gradient is None
    assert np.isclose(3.440220555221998, stage.length_flattop, rtol=1e-5, atol=0.0)


    # ========== Tests with external fields ==========
    stage = setup_minimal_StageHipace(external_focusing=True)
    stage.match_length_guiding_2_num_beta_osc(num_beta_osc=num_beta_osc)
    
    assert np.isclose(num_beta_osc, stage.length_flattop2num_beta_osc(), rtol=1e-10, atol=0.0)
    assert np.isclose(140.1695315279373, stage._external_focusing_gradient, rtol=1e-5, atol=0.0)
    assert np.isclose(3.4268706541720553, stage.length_flattop, rtol=1e-5, atol=0.0)


    stage = setup_minimal_StageHipace(external_focusing=True)
    stage.match_length_guiding_2_num_beta_osc(num_beta_osc=num_beta_osc, driver_half_oscillations=2.0)

    assert np.isclose(2.0, stage.driver_half_oscillations, rtol=1e-15, atol=0.0)
    assert np.isclose(num_beta_osc, stage.length_flattop2num_beta_osc(), rtol=1e-10, atol=0.0)
    assert np.isclose(574.1247168375805, stage._external_focusing_gradient, rtol=1e-5, atol=0.0)
    assert np.isclose(3.3865024719148242, stage.length_flattop, rtol=1e-5, atol=0.0)


    # ========== With external fields, lower stage nominal energy ==========
    num_beta_osc2 = 8.0
    stage = setup_minimal_StageHipace(nom_energy=3e9, external_focusing=True)
    stage.match_length_guiding_2_num_beta_osc(num_beta_osc=num_beta_osc2)
    
    assert np.isclose(num_beta_osc2, stage.length_flattop2num_beta_osc(), rtol=1e-10, atol=0.0)
    assert np.isclose(1038.1202586404845, stage._external_focusing_gradient, rtol=1e-5, atol=0.0)
    assert np.isclose(1.259217324849329, stage.length_flattop, rtol=1e-5, atol=0.0)


    stage = setup_minimal_StageHipace(nom_energy=3e9, external_focusing=True)
    stage.match_length_guiding_2_num_beta_osc(num_beta_osc=num_beta_osc2, driver_half_oscillations=2.0)
    
    assert np.isclose(num_beta_osc2, stage.length_flattop2num_beta_osc(), rtol=1e-10, atol=0.0)
    assert np.isclose(5119.8513420067175, stage._external_focusing_gradient, rtol=1e-5, atol=0.0)
    assert np.isclose(1.13403339406399, stage.length_flattop, rtol=1e-5, atol=0.0)


    # ========== With external fields, lower stage nominal energy, lower driver energy ==========
    stage = setup_minimal_StageHipace(nom_energy=3e9, external_focusing=True)
    stage.driver_source.energy = 4.5e9  # [eV]
    stage.match_length_guiding_2_num_beta_osc(num_beta_osc=num_beta_osc2)
    
    assert np.isclose(num_beta_osc2, stage.length_flattop2num_beta_osc(), rtol=1e-10, atol=0.0)
    assert np.isclose(88.3976616856249, stage._external_focusing_gradient, rtol=1e-5, atol=0.0)
    assert np.isclose(1.2945695557961696, stage.length_flattop, rtol=1e-5, atol=0.0)


    stage = setup_minimal_StageHipace(nom_energy=3e9, external_focusing=True)
    stage.driver_source.energy = 4.5e9  # [eV]
    stage.match_length_guiding_2_num_beta_osc(num_beta_osc=num_beta_osc2, driver_half_oscillations=2.0)
    
    assert np.isclose(num_beta_osc2, stage.length_flattop2num_beta_osc(), rtol=1e-10, atol=0.0)
    assert np.isclose(359.3285677857948, stage._external_focusing_gradient, rtol=1e-5, atol=0.0)
    assert np.isclose(1.2841918239865389, stage.length_flattop, rtol=1e-5, atol=0.0)


    # ========== Tests with external fields, with ramps ==========
    num_beta_osc3 = 5.5
    stage = setup_minimal_StageHipace(nom_energy=10e9, external_focusing=True)
    stage.driver_source.energy = 20e9  # [eV]
    stage.ramp_beta_mag = 2.0
    stage.upramp = PlasmaRamp()
    stage.downramp = PlasmaRamp()

    stage.match_length_guiding_2_num_beta_osc(num_beta_osc=num_beta_osc3, driver_half_oscillations=1.0)

    assert np.isclose(263.5820977342094, stage._external_focusing_gradient, rtol=1e-5, atol=0.0)
    assert np.isclose(1.3834379291754226, stage.length_flattop, rtol=1e-5, atol=0.0)
    assert np.isclose(num_beta_osc3 - 0.5, stage.length_flattop2num_beta_osc(), rtol=1e-4, atol=0.0)
    

    stage = setup_minimal_StageHipace(nom_energy=10e9, external_focusing=True)
    stage.driver_source.energy = 20e9  # [eV]
    stage.ramp_beta_mag = 2.0
    stage.upramp = PlasmaRamp()
    stage.downramp = PlasmaRamp()

    stage.match_length_guiding_2_num_beta_osc(num_beta_osc=num_beta_osc3, driver_half_oscillations=2.0)

    assert np.isclose(2.0, stage.driver_half_oscillations, rtol=1e-10, atol=0.0)
    assert np.isclose(1097.699359490811, stage._external_focusing_gradient, rtol=1e-5, atol=0.0)
    assert np.isclose(1.3520401015419283, stage.length_flattop, rtol=1e-5, atol=0.0)
    assert np.isclose(num_beta_osc3 - 0.5, stage.length_flattop2num_beta_osc(), rtol=1e-4, atol=0.0)
    
    


