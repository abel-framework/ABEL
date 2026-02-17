# -*- coding: utf-8 -*-
# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later


"""
ABEL : StageWakeT unit tests
"""

import pytest
from abel import *
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a backend that does not display figure to suppress plots.


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


def setup_StageWakeT(driver_source=None, nom_accel_gradient_flattop=6.4e9, nom_energy_gain_flattop=31.9e9, plasma_density=7.0e21, use_ramps=False):

    stage = StageWakeT()
    stage.nom_accel_gradient_flattop = nom_accel_gradient_flattop
    stage.nom_energy_gain_flattop = nom_energy_gain_flattop
    stage.plasma_density = plasma_density                                         # [m^-3]
    stage.driver_source = driver_source
    if use_ramps:
        stage.ramp_beta_mag = 10.0
    else:
        stage.ramp_beta_mag = 1.0
    
    # Set up ramps after the stage is fully configured
    if use_ramps:
        stage.upramp = PlasmaRamp() 
        stage.downramp = PlasmaRamp()
    
    return stage


@pytest.mark.StageWakeT
def test_driver_source_setter():
    """
    Tests ensuring that the driver source setter does not set invalid classes 
    and that the driver source has valid energy.
    """

    driver_source = setup_basic_driver_source()
    driver_complex = DriverComplex()
    driver_complex.source = driver_source
    stage = StageWakeT()

    # Valid options
    stage.driver_source = driver_source
    assert isinstance(stage.driver_source, Source)
    stage.driver_source = None
    assert stage.driver_source is None
    stage.driver_source = driver_complex
    assert isinstance(stage.driver_source, DriverComplex)

    # Invalid instances
    with pytest.raises(TypeError):
        stage.driver_source = 42
    with pytest.raises(TypeError):
        stage.driver_source = 4.2
    with pytest.raises(TypeError):
        stage.driver_source = 'lorem'

    # Invalid driver source energy
    stage2 = StageWakeT()
    driver_source2 = setup_basic_driver_source()
    driver_source2.energy = None
    with pytest.raises(ValueError):
        stage2.driver_source = driver_source2 # driver source energy must be set before being added to a stage.
    
    driver_complex2 = DriverComplex()
    driver_complex2.source = driver_source2
    with pytest.raises(ValueError):
        stage2.driver_source = driver_complex2 # driver source energy must be set before being added to a stage.  


@pytest.mark.StageWakeT
def test_tracking():
    """
    Tests for tracking ``StageWakeT`` without ramps.

    Examine stage configurations and the output main beam parameters. Also run 
    some diagnostics.
    """

    import matplotlib.pyplot as plt

    # reset seed (for reproducibility)
    np.random.seed(42)

    nom_energy_gain=0.319e9
    
    # setup beams and stage
    driver_source = setup_basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=0.0, y_angle=0.0)
    stage = setup_StageWakeT(driver_source=driver_source, nom_energy_gain_flattop=nom_energy_gain, use_ramps=False)
    main_source = setup_basic_main_source()
    stage.nom_energy = main_source.energy

    # perform simulation
    beam = stage.track(main_source.track())

    # Inspect stage configurations
    assert stage.driver_source.align_beam_axis is True
    assert stage.has_ramp() is False
    assert np.isclose(stage.plasma_density, 7.0e21, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy, 5.0e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_flattop, stage.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_gain, nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_gain_flattop, stage.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_accel_gradient, 6.4e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_accel_gradient_flattop, 6.4e9, rtol=1e-2, atol=0.0)
    assert np.isclose(stage.length, stage.nom_energy_gain/stage.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.length_flattop, stage.nom_energy_gain_flattop/stage.nom_accel_gradient_flattop, rtol=1e-15, atol=0.0)

    # examine the plasma density profile
    assert np.isclose(stage.get_plasma_profile(), stage.plasma_density, rtol=1e-15, atol=0.0)
    
    # Examine output beam
    assert np.isclose(beam.energy(), main_source.energy + stage.nom_energy_gain, rtol=3e-2, atol=0.0)
    assert np.isclose(beam.charge(), main_source.charge, rtol=1e-3, atol=0.0)
    assert len(beam) == main_source.num_particles
    assert beam.stage_number == 1
    assert np.isclose(beam.location, stage.length, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.particle_mass, SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.x_offset(), main_source.x_offset, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.y_offset(), main_source.y_offset, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.z_offset(), main_source.z_offset, rtol=0.0, atol=1.0e-6)
    assert np.isclose(beam.x_angle(), main_source.x_angle, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.y_angle(), main_source.y_angle, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.norm_emittance_x(), main_source.emit_nx, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), main_source.emit_ny, rtol=7e-2, atol=0.0)
    nom_beta_x = np.sqrt(beam.energy()/main_source.energy) * main_source.beta_x
    nom_beta_y = np.sqrt(beam.energy()/main_source.energy) * main_source.beta_y
    assert np.isclose(beam.beta_x(), nom_beta_x, rtol=1e-1, atol=0.0)
    assert np.isclose(beam.beta_y(), nom_beta_y, rtol=1e-1, atol=0.0)
    assert np.isclose(beam.bunch_length(), main_source.bunch_length, rtol=5e-2, atol=0.0)
    assert np.isclose(beam.rel_energy_spread(), 0.0097, rtol=1e-1, atol=0.0)

    # Test that the diagnostics can be executed
    stage.plot_wakefield()
    stage.plot_wake()
    stage.plot_driver_evolution()
    stage.plot_evolution()

    # Close all plots
    plt.close('all')


@pytest.mark.StageWakeT
def test_ramped_tracking():
    """
    Tests for tracking ``StageWakeT`` with ramps.

    Examine stage configurations and the output main beam parameters. Also run 
    some diagnostics.
    """

    import matplotlib.pyplot as plt

    np.random.seed(42)

    nom_energy_gain=0.319e9
    
    driver_source = setup_basic_driver_source(enable_xy_jitter=False, enable_xpyp_jitter=False, x_angle=0.0, y_angle=0.0)
    stage = setup_StageWakeT(driver_source=driver_source, nom_energy_gain_flattop=nom_energy_gain, use_ramps=True)
    main_source = setup_basic_main_source(ramp_beta_mag=stage.ramp_beta_mag)
    stage.nom_energy = main_source.energy

    # perform tracking
    beam = stage.track(main_source.track())

    # Inspect stage configurations
    assert stage.driver_source.align_beam_axis is True
    assert stage.has_ramp() is True
    assert np.isclose(stage.plasma_density, 7.0e21, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy, 5.0e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_flattop, stage.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_gain, nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_gain_flattop, stage.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_accel_gradient, 2.287e9, rtol=1e-2, atol=0.0)
    assert np.isclose(stage.nom_accel_gradient_flattop, 6.4e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.length, stage.nom_energy_gain/stage.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.length_flattop, stage.nom_energy_gain_flattop/stage.nom_accel_gradient_flattop, rtol=1e-15, atol=0.0)

    # examine the plasma density profile
    plasma_profile = stage.get_plasma_profile()
    assert callable(plasma_profile)
    assert np.isclose(plasma_profile(0.0), stage.plasma_density/stage.ramp_beta_mag, rtol=1e-3, atol=0.0)
    assert np.isclose(plasma_profile(stage.length/2), stage.plasma_density, rtol=1e-3, atol=0.0)
    assert np.isclose(plasma_profile(stage.length), stage.plasma_density/stage.ramp_beta_mag, rtol=1e-3, atol=0.0)
    
    # Examine output beam
    assert np.isclose(beam.energy(), main_source.energy + stage.nom_energy_gain, rtol=3e-2, atol=0.0)
    assert np.isclose(beam.charge(), main_source.charge, rtol=1e-3, atol=0.0)
    assert len(beam) == main_source.num_particles
    assert beam.stage_number == 1
    assert np.isclose(beam.location, stage.length, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.particle_mass, SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.x_offset(), main_source.x_offset, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.y_offset(), main_source.y_offset, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.z_offset(), main_source.z_offset, rtol=0.0, atol=1.0e-6)
    assert np.isclose(beam.x_angle(), main_source.x_angle, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.y_angle(), main_source.y_angle, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.norm_emittance_x(), main_source.emit_nx, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), main_source.emit_ny, rtol=7e-2, atol=0.0)
    nom_beta_x = np.sqrt(beam.energy()/main_source.energy) * main_source.beta_x
    nom_beta_y = np.sqrt(beam.energy()/main_source.energy) * main_source.beta_y
    assert np.isclose(beam.beta_x(), nom_beta_x, rtol=1e-1, atol=0.0)
    assert np.isclose(beam.beta_y(), nom_beta_y, rtol=1e-1, atol=0.0)
    assert np.isclose(beam.bunch_length(), main_source.bunch_length, rtol=5e-2, atol=0.0)
    assert np.isclose(beam.rel_energy_spread(), 0.0097, rtol=1e-1, atol=0.0)

    # Test that the diagnostics can be executed
    stage.plot_wakefield()
    stage.plot_wake()
    stage.plot_driver_evolution()
    stage.plot_evolution()

    # Close all plots
    plt.close('all')

    
@pytest.mark.StageWakeT
def test_tracking_only_main_beam():
    """
    Tests for tracking ``StageWakeT`` without ramps tracking only a single main 
    beam.

    Examine stage configurations and the output main beam parameters. Also run 
    some diagnostics.
    """

    import matplotlib.pyplot as plt

    # reset seed (for reproducibility)
    np.random.seed(42)

    nom_energy_gain=0.319e9
    
    # setup beams and stage
    stage = setup_StageWakeT(driver_source=None, nom_energy_gain_flattop=nom_energy_gain, use_ramps=False)
    stage.use_single_beam = True
    main_source = setup_basic_main_source()
    stage.nom_energy = main_source.energy

    # perform simulation
    beam = stage.track(main_source.track())

    # Inspect stage configurations
    assert stage.has_ramp() is False
    assert np.isclose(stage.plasma_density, 7.0e21, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy, 5.0e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_flattop, stage.nom_energy, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_gain, nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_energy_gain_flattop, stage.nom_energy_gain, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_accel_gradient, 6.4e9, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.nom_accel_gradient_flattop, 6.4e9, rtol=1e-2, atol=0.0)
    assert np.isclose(stage.length, stage.nom_energy_gain/stage.nom_accel_gradient, rtol=1e-15, atol=0.0)
    assert np.isclose(stage.length_flattop, stage.nom_energy_gain_flattop/stage.nom_accel_gradient_flattop, rtol=1e-15, atol=0.0)

    # examine the plasma density profile
    assert np.isclose(stage.get_plasma_profile(), stage.plasma_density, rtol=1e-15, atol=0.0)
    
    # Examine output beam
    assert np.isclose(beam.energy(), 4821973619.821054, rtol=3e-2, atol=0.0)
    assert np.isclose(beam.charge(), main_source.charge, rtol=1e-3, atol=0.0)
    assert len(beam) == main_source.num_particles
    assert beam.stage_number == 1
    assert np.isclose(beam.location, stage.length, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.particle_mass, SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.x_offset(), main_source.x_offset, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.y_offset(), main_source.y_offset, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.z_offset(), main_source.z_offset, rtol=0.0, atol=1.0e-6)
    assert np.isclose(beam.x_angle(), main_source.x_angle, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.y_angle(), main_source.y_angle, rtol=0.0, atol=1.0e-8)
    assert np.isclose(beam.norm_emittance_x(), 0.0003031640487476247, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), 1.019663344335304e-06, rtol=7e-2, atol=0.0)
    assert np.isclose(beam.beta_x(), 0.023933341262344372, rtol=1e-1, atol=0.0)
    assert np.isclose(beam.beta_y(), 0.01761011889735352, rtol=1e-1, atol=0.0)
    assert np.isclose(beam.bunch_length(), main_source.bunch_length, rtol=5e-2, atol=0.0)
    assert np.isclose(beam.rel_energy_spread(), 0.02030454802939791, rtol=1e-1, atol=0.0)

    # Test that the diagnostics can be executed
    stage.plot_wakefield()
    stage.plot_wake()
    stage.plot_evolution()

    # Close all plots
    plt.close('all')

    





