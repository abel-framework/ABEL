"""
ABEL : Source class tests
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
from abel.utilities.plasma_physics import beta_matched
import os
import scipy.constants as SI
import numpy as np

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
    "Check that the generated ``Beam`` from a ``SourceBasic`` has the desired properties."

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
def test_SourceTrapezoid2Beam():
    "Check that the generated ``Beam`` from a ``SourceTrapezoid`` has the desired properties."

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
def test_SourceFlatopBeam():
    "Check that the generated ``Beam`` from a ``SourceFlatTop`` has the desired properties."

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
    "Check that ``SourceCapsule`` retuns returns the same ``Beam`` as the input ``Beam``."

    beam_file = 'tests/data/test_StagePrtclTransWakeInstability_beamline/test_baseline_linac/shot_000/beam_003_00048.558626.h5'

    source = SourceCapsule()
    ref_beam = Beam.load(beam_file)
    source.beam = ref_beam
    beam = source.track()
    Beam.comp_beams(beam, ref_beam)


@pytest.mark.sources
def test_SourceFromFile2Beam():
    "Check that ``SourceFromFile`` retuns returns the same ``Beam`` as the input ``Beam``."

    beam_file = 'tests' + os.sep + 'data' + os.sep + 'test_StagePrtclTransWakeInstability_beamline' + os.sep + 'test_baseline_linac' + os.sep + 'shot_000' + os.sep + 'beam_003_00048.558626.h5'

    source = SourceFromFile()
    source.file = beam_file
    beam = source.track()
    ref_beam = Beam.load(beam_file)
    Beam.comp_beams(beam, ref_beam)

    # Trigger exception for when a file does not exist
    with pytest.raises(FileNotFoundError):
        file = 'tests' + os.sep + 'data' + os.sep + 'test_StagePrtclTransWakeInstability_beamline' + os.sep + 'test_baseline_linac' + os.sep + 'blabla.h5'
        source = SourceFromFile(file=file)
    with pytest.raises(FileNotFoundError):
        file = 'tests' + os.sep + 'data' + os.sep + 'test_StagePrtclTransWakeInstability_beamline' + os.sep + 'test_baseline_linac' + os.sep + 'shot_000' + os.sep + 'blabla.h5'
        source = SourceFromFile()
        source.file = file


