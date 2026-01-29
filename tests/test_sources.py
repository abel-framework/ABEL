# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

"""
ABEL : Source class tests
"""

import pytest
from abel import *
from abel.utilities.plasma_physics import beta_matched
import os, shutil
import scipy.constants as SI
import numpy as np


def setup_basic_source(plasma_density=6.0e20):

    source = SourceBasic()
    source.bunch_length = 40.0e-06                                                  # [m], rms.
    source.num_particles = 10000                                               
    source.charge = -SI.e * 1.0e10                                                  # [C]

    # Energy parameters
    source.energy = 3.0e9                                                           # [eV]
    source.rel_energy_spread = 0.02                                                 # Relative rms energy spread

    # Emittances
    source.emit_nx, source.emit_ny = 15e-6, 0.1e-6                                  # [m rad]

    # Beta functions
    source.beta_x = beta_matched(plasma_density, source.energy)                     # [m]
    source.beta_y = source.beta_x                                                   # [m]

    # Offsets
    source.z_offset = 0.0                                                           # [m]
    source.x_offset = 0.0                                                           # [m]
    source.y_offset = 0.0                                                           # [m]

    # Other
    source.symmetrize_6d = True

    return source


def setup_trapezoid_source(current_head=0.1e3, bunch_length=1050e-6, z_offset=1620e-6, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0, enable_xy_jitter=False, enable_xpyp_jitter=False):
    source = SourceTrapezoid()
    source.current_head = current_head                                              # [A]
    source.bunch_length = bunch_length                                              # [m]

    source.num_particles = 10000                                                 
    source.charge = 5.0e10 * -SI.e                                                  # [C]
    source.energy = 4.0e9                                                           # [eV] 
    source.gaussian_blur = 50e-6                                                    # [m]
    source.rel_energy_spread = 0.01                                              

    source.emit_nx, source.emit_ny = 40e-6, 80e-6                                   # [m rad]
    source.beta_x, source.beta_y = 0.5, 0.5                                         # [m]

     # Offsets
    source.z_offset = z_offset                                                      # [m]
    source.x_offset = x_offset                                                      # [m]
    source.y_offset = y_offset                                                      # [m]
    source.x_angle = x_angle                                                        # [rad]
    source.y_angle = y_angle                                                        # [rad]

    if enable_xy_jitter:
        source.jitter.x = 100e-9                                                    # [m], std
        source.jitter.y = 100e-9                                                    # [m], std

    if enable_xpyp_jitter:
        source.jitter.xp = 1.0e-6                                                   # [rad], std
        source.jitter.yp = 1.0e-6                                                   # [rad], std

    source.symmetrize = True

    return source


def check_beam_source_parameters(beam, source):
    assert np.allclose(beam.particle_mass, SI.m_e, rtol=1e-05, atol=1e-08)
    assert np.allclose(len(beam), source.num_particles, rtol=1e-05, atol=1e-08)
    assert np.allclose(beam.charge(), source.charge, rtol=1e-05, atol=1e-08)
    assert np.allclose(beam.energy(), source.energy, rtol=0.0, atol=0.1e9)
    assert np.allclose(beam.rel_energy_spread(), source.rel_energy_spread, rtol=0.0, atol=0.005)
    assert np.allclose(beam.norm_emittance_x(), source.emit_nx, rtol=0.01, atol=1e-6)
    assert np.allclose(beam.norm_emittance_y(), source.emit_ny, rtol=0.01, atol=0.05e-6)
    assert np.allclose(beam.beta_x(), source.beta_x, rtol=0.01, atol=5e-3)
    assert np.allclose(beam.beta_y(), source.beta_y, rtol=0.01, atol=5e-3)

    if isinstance(source, SourceTrapezoid) or isinstance(source, SourceFlatTop):
        assert np.allclose(beam.zs().max(), source.z_offset, rtol=0.01, atol=2e-6)
    else:
        assert np.allclose(beam.z_offset(), source.z_offset, rtol=0.0, atol=1e-8)
    assert np.allclose(beam.x_offset(), source.x_offset, rtol=0.0, atol=1e-8)
    assert np.allclose(beam.y_offset(), source.y_offset, rtol=0.0, atol=1e-8)
    assert np.allclose(beam.x_angle(), source.x_angle, rtol=0.0, atol=1e-8)
    assert np.allclose(beam.y_angle(), source.y_angle, rtol=0.0, atol=1e-8)

    if isinstance(source, SourceTrapezoid) or isinstance(source, SourceFlatTop):
        assert np.allclose(beam.zs().max()-beam.zs().min(), source.bunch_length, rtol=0.01, atol=0.0)
    else:
        assert np.allclose(beam.bunch_length(), source.bunch_length, rtol=0.01, atol=2e-6)


@pytest.mark.sources
def test_SourceBasic2Beam():
    """
    Check that the generated ``Beam`` from a ``SourceBasic`` has the desired 
    properties.
    """

    np.random.seed(42)

    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0

    source = SourceBasic()
    source.bunch_length = 40.0e-06                                                # [m], rms.
    source.num_particles = 10000                                               
    source.charge = -SI.e * 1.0e10                                                   # [C]

    # Energy parameters
    source.energy = 3e9                                                           # [eV]
    source.rel_energy_spread = 0.02                                               # Relative rms energy spread

    # Emittances
    source.emit_nx, source.emit_ny = 90.0e-6, 0.32e-6                             # [m rad], budget value

    # Beta functions
    source.beta_x = beta_matched(plasma_density, source.energy) * ramp_beta_mag   # [m]
    source.beta_y = source.beta_x                                                 # [m]

    # Offsets
    source.z_offset = 0.00e-6                                                     # [m]
    source.x_offset = 1.50e-6                                                     # [m]
    source.y_offset = -1.0e-6                                                     # [m]
    source.x_angle = -0.1e-6                                                      # [rad]
    source.y_angle = 0.7e-6                                                       # [rad]

    # Other
    source.symmetrize_6d = True

    # Track and check
    beam = source.track()
    beam.print_summary()
    check_beam_source_parameters(beam, source)


@pytest.mark.sources
def test_SourceBasic_beam_alignment():
    """
    Check that the generated ``Beam`` from a ``SourceBasic`` is aligned to its 
    propgation direction.
    """

    np.random.seed(42)

    source0 = setup_basic_source()
    #source0.align_beam_axis = False                                                # False by default
    source0.x_angle = 1.3e-6                                                        # [rad]
    source0.y_angle = 2e-6                                                          # [rad]
    beam0 = source0.track()
    assert np.isclose(beam0.x_angle(), source0.x_angle, rtol=1e-5, atol=0.0)
    assert np.isclose(beam0.y_angle(), source0.y_angle, rtol=1e-5, atol=0.0)
    assert np.isclose(0.0, beam0.x_tilt_angle(), rtol=0.0, atol=1e-15)
    assert np.isclose(0.0, beam0.y_tilt_angle(), rtol=0.0, atol=1e-15)

    source = setup_basic_source()
    source.align_beam_axis = True
    source.x_angle = 1.3e-6                                                         # [rad]
    source.y_angle = 2e-6                                                           # [rad]
    beam = source.track()
    assert np.isclose(beam.x_angle(), beam.x_tilt_angle(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam.y_angle(), beam.y_tilt_angle(), rtol=1e-2, atol=0.0)

    source2 = setup_basic_source()
    source2.align_beam_axis = True
    source2.x_angle = -1.3e-6                                                       # [rad]
    source2.y_angle = -2e-6                                                         # [rad]
    beam2 = source2.track()
    assert np.isclose(beam2.x_angle(), beam2.x_tilt_angle(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam2.y_angle(), beam2.y_tilt_angle(), rtol=1e-2, atol=0.0)

    source3 = setup_basic_source()
    source3.align_beam_axis = True
    source3.x_angle = 1.3e-5                                                        # [rad]
    source3.y_angle = -2e-6                                                         # [rad]
    beam3 = source3.track()
    assert np.isclose(beam3.x_angle(), beam3.x_tilt_angle(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam3.y_angle(), beam3.y_tilt_angle(), rtol=1e-2, atol=0.0)

    source4 = setup_basic_source()
    source4.align_beam_axis = True
    source4.x_angle = -1.3e-6                                                       # [rad]
    source4.y_angle = 2.3e-5                                                        # [rad]
    source4.x_offset = 5.1e-6                                                       # [m]
    source4.y_offset = -4.3e-6                                                      # [m]
    beam4 = source4.track()
    assert np.isclose(beam4.x_angle(), beam4.x_tilt_angle(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam4.y_angle(), beam4.y_tilt_angle(), rtol=1e-2, atol=0.0)

    source5 = setup_basic_source()
    source5.align_beam_axis = True
    source5.x_angle = -1.3e-8                                                       # [rad]
    source5.y_angle = 2.3e-8                                                        # [rad]
    source5.x_offset = 5.1e-6                                                       # [m]
    source5.y_offset = -4.3e-6                                                      # [m]
    beam5 = source5.track()
    assert np.isclose(beam5.x_angle(), beam5.x_tilt_angle(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam5.y_angle(), beam5.y_tilt_angle(), rtol=1e-2, atol=0.0)

    source6 = setup_basic_source()
    source6.align_beam_axis = True
    source6.x_angle = 5.0e-5                                                        # [rad]
    source6.y_angle = -5.0e-5                                                       # [rad]
    beam6 = source6.track()
    assert np.isclose(beam6.x_angle(), beam6.x_tilt_angle(), rtol=1e-1, atol=0.0)
    assert np.isclose(beam6.y_angle(), beam6.y_tilt_angle(), rtol=1e-1, atol=0.0)



@pytest.mark.sources
def test_SourceTrapezoid2Beam():
    """
    Check that the generated ``Beam`` from a ``SourceTrapezoid`` has the desired 
    properties.
    """

    np.random.seed(42)

    source = SourceTrapezoid()
    source.bunch_length = 1050e-6                                                 # [m], rms.
    source.num_particles = 30000                                               
    source.charge = -SI.e * 5.0e10                                                   # [C]
    source.current_head = 0.1e3                                                   # [A]

    # Energy parameters
    source.energy = 4.0e9                                                         # [eV]
    source.rel_energy_spread = 0.01                                               # Relative rms energy spread

    # Emittances
    source.emit_nx, source.emit_ny = 50e-6, 100e-6                                # [m rad], budget value

    # Beta functions
    source.beta_x, source.beta_y = 0.5, 0.5                                       # [m]

    # Offsets
    source.z_offset = 1615e-6                                                     # [m]
    source.x_offset = 1.50e-6                                                     # [m]
    source.y_offset = -1.0e-6                                                     # [m]
    source.x_angle = -0.1e-6                                                      # [rad]
    source.y_angle = 0.7e-6                                                       # [rad]

    # Other
    source.symmetrize = True
    #source.gaussian_blur = 50e-6                                                  # [m], excluding blur makes it easier to control z_offset.

    # Track and check
    beam = source.track()
    beam.print_summary()
    check_beam_source_parameters(beam, source)


@pytest.mark.sources
def test_SourceTrapezoid_beam_alignment():
    """
    Check that the generated ``Beam`` from a ``SourceTrapezoid`` is aligned to its 
    propgation direction.
    """

    np.random.seed(42)

    source0 = setup_trapezoid_source()
    #source0.align_beam_axis = False                                                # False by default
    source0.x_angle = 1.3e-6                                                        # [rad]
    source0.y_angle = 2e-6                                                          # [rad]
    beam0 = source0.track()
    assert np.isclose(beam0.x_angle(), source0.x_angle, rtol=1e-5, atol=0.0)
    assert np.isclose(beam0.y_angle(), source0.y_angle, rtol=1e-5, atol=0.0)
    assert np.isclose(0.0, beam0.x_tilt_angle(), rtol=0.0, atol=1e-15)
    assert np.isclose(0.0, beam0.y_tilt_angle(), rtol=0.0, atol=1e-15)

    source = setup_trapezoid_source()
    source.align_beam_axis = True
    source.x_angle = 1.3e-6                                                         # [rad]
    source.y_angle = 2e-6                                                           # [rad]
    beam = source.track()
    assert np.isclose(beam.x_angle(), beam.x_tilt_angle(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam.y_angle(), beam.y_tilt_angle(), rtol=1e-2, atol=0.0)

    source2 = setup_trapezoid_source()
    source2.align_beam_axis = True
    source2.x_angle = -1.3e-6                                                       # [rad]
    source2.y_angle = -2e-6                                                         # [rad]
    beam2 = source2.track()
    assert np.isclose(beam2.x_angle(), beam2.x_tilt_angle(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam2.y_angle(), beam2.y_tilt_angle(), rtol=1e-2, atol=0.0)

    source3 = setup_trapezoid_source()
    source3.align_beam_axis = True
    source3.x_angle = 1.3e-5                                                        # [rad]
    source3.y_angle = -2e-6                                                         # [rad]
    beam3 = source3.track()
    assert np.isclose(beam3.x_angle(), beam3.x_tilt_angle(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam3.y_angle(), beam3.y_tilt_angle(), rtol=1e-2, atol=0.0)

    source4 = setup_trapezoid_source()
    source4.align_beam_axis = True
    source4.x_angle = -1.3e-6                                                       # [rad]
    source4.y_angle = 2e-6                                                          # [rad]
    source4.x_offset = 5.1e-6                                                       # [m]
    source4.y_offset = -4.3e-6                                                      # [m]
    beam4 = source4.track()
    assert np.isclose(beam4.x_angle(), beam4.x_tilt_angle(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam4.y_angle(), beam4.y_tilt_angle(), rtol=1e-2, atol=0.0)

    source5 = setup_trapezoid_source()
    source5.align_beam_axis = True
    source5.x_angle = -1.3e-8                                                       # [rad]
    source5.y_angle = 2.3e-8                                                        # [rad]
    source5.x_offset = 5.1e-6                                                       # [m]
    source5.y_offset = -4.3e-6                                                      # [m]
    beam5 = source5.track()
    assert np.isclose(beam5.x_angle(), beam5.x_tilt_angle(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam5.y_angle(), beam5.y_tilt_angle(), rtol=1e-2, atol=0.0)

    source6 = setup_trapezoid_source()
    source6.align_beam_axis = True
    source6.x_angle = 5.0e-4                                                        # [rad]
    source6.y_angle = -5.0e-4                                                       # [rad]
    beam6 = source6.track()
    assert np.isclose(beam6.x_angle(), beam6.x_tilt_angle(), rtol=1e-1, atol=0.0)
    assert np.isclose(beam6.y_angle(), beam6.y_tilt_angle(), rtol=1e-1, atol=0.0)


@pytest.mark.sources
def test_SourceFlatopBeam():
    """
    Check that the generated ``Beam`` from a ``SourceFlatTop`` has the desired 
    properties.
    """

    np.random.seed(42)

    source = SourceFlatTop()
    source.bunch_length = 1050e-6                                                 # [m], rms.
    source.num_particles = 30000                                               
    source.charge = -SI.e * 5.0e10                                                   # [C]

    # Energy parameters
    source.energy = 4.0e9                                                         # [eV]
    source.rel_energy_spread = 0.01                                               # Relative rms energy spread

    # Emittances
    source.emit_nx, source.emit_ny = 50e-6, 100e-6                                # [m rad], budget value

    # Beta functions
    source.beta_x, source.beta_y = 0.5, 0.5                                       # [m]

    # Offsets
    source.z_offset = 1615e-6                                                     # [m]
    source.x_offset = 1.50e-6                                                     # [m]
    source.y_offset = -1.0e-6                                                     # [m]
    source.x_angle = -0.1e-6                                                      # [rad]
    source.y_angle = 0.7e-6                                                       # [rad]

    # Other
    source.symmetrize = True

    # Track and check
    beam = source.track()
    beam.print_summary()
    check_beam_source_parameters(beam, source)


@pytest.mark.sources
def test_SourceCapsule2Beam():
    """
    Check that ``SourceCapsule`` retuns returns the same ``Beam`` as the input 
    ``Beam``.
    """

    # Generate a beam for comparison
    source_ref = setup_basic_source()
    ref_beam = source_ref.track()

    # Compare the beams
    source = SourceCapsule()
    source.beam = ref_beam
    beam = source.track()
    Beam.comp_beams(beam, ref_beam)


@pytest.mark.sources
def test_SourceFromFile2Beam():
    """
    Check that ``SourceFromFile`` retuns returns the same ``Beam`` as the input 
    ``Beam``.
    """

    # Make a beam file for comparison
    save_dir = 'tests' + os.sep + 'data' + os.sep + 'test_beam'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    beam_file = save_dir + os.sep + 'beam_test_SourceCapsule2Beam.h5'
    source = setup_basic_source()
    ref_beam = source.track()
    ref_beam.save(filename=beam_file)

    # Use SourceFromFile to extract from beam file and compare beams
    source = SourceFromFile()
    source.file = beam_file
    beam = source.track()
    Beam.comp_beams(beam, ref_beam)
    shutil.rmtree(save_dir)

    # Trigger exception for when a file does not exist and check that these are handled correctly
    with pytest.raises(FileNotFoundError):
        file = 'tests' + os.sep + 'data' + os.sep + 'test_StageReducedModels_beamline' + os.sep + 'test_baseline_linac' + os.sep + 'blabla.h5'
        source = SourceFromFile(file=file)
    with pytest.raises(FileNotFoundError):
        file = 'tests' + os.sep + 'data' + os.sep + 'test_StageReducedModels_beamline' + os.sep + 'test_baseline_linac' + os.sep + 'shot_000' + os.sep + 'blabla.h5'
        source = SourceFromFile()
        source.file = file

    


