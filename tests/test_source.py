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

def test_SourceBasic2Beam():
    "Check that the generated ``Beam`` from a ``SourceBasic`` has the desired properties."

    np.random.seed(42)

    plasma_density = 6.0e+20                                                      # [m^-3]
    ramp_beta_mag = 5.0

    main = SourceBasic()
    main.bunch_length = 40.0e-06                                                  # [m], rms.
    main.num_particles = 10000                                               
    main.charge = -e * 1.0e10                                                     # [C]

    # Energy parameters
    main.energy = 3e9                                                             # [eV]
    main.rel_energy_spread = 0.02                                                 # Relative rms energy spread

    # Emittances
    main.emit_nx, main.emit_ny = 90.0e-6, 0.32e-6                                 # [m rad], budget value

    # Beta functions
    main.beta_x = beta_matched(plasma_density, main.energy) * ramp_beta_mag       # [m]
    main.beta_y = main.beta_x                                                     # [m]

    # Offsets
    main.z_offset = 0.00e-6                                                       # [m]
    main.x_offset = 1.50e-6                                                       # [m]
    main.y_offset = -1.0e-6                                                       # [m]
    main.x_angle = -0.1e-6                                                         # [rad]
    main.y_angle = 0.7e-6                                                         # [rad]

    # Other
    main.symmetrize_6d = True

    # Track and check
    beam = main.track()

    beam.print_summary()

    assert np.allclose(beam.particle_mass, SI.m_e, rtol=1e-05, atol=1e-08)
    assert np.allclose(len(beam), main.num_particles, rtol=1e-05, atol=1e-08)
    assert np.allclose(beam.charge(), main.charge, rtol=1e-05, atol=1e-08)
    assert np.allclose(beam.energy(), main.energy, rtol=0.0, atol=0.1e9)
    assert np.allclose(beam.rel_energy_spread(), main.rel_energy_spread, rtol=0.0, atol=0.005)
    assert np.allclose(beam.bunch_length(), main.bunch_length, rtol=0.0, atol=2e-6)
    assert np.allclose(beam.norm_emittance_x(), main.emit_nx, rtol=0.0, atol=1e-6)
    assert np.allclose(beam.norm_emittance_y(), main.emit_ny, rtol=0.0, atol=0.05e-6)
    assert np.allclose(beam.beta_x(), main.beta_x, rtol=0.0, atol=5e-3)
    assert np.allclose(beam.beta_y(), main.beta_y, rtol=0.0, atol=5e-3)     
    assert np.allclose(beam.z_offset(), main.z_offset, rtol=0.0, atol=1e-8)
    assert np.allclose(beam.x_offset(), main.x_offset, rtol=0.0, atol=1e-8)
    assert np.allclose(beam.y_offset(), main.y_offset, rtol=0.0, atol=1e-8)
    assert np.allclose(beam.x_angle(), main.x_angle, rtol=0.0, atol=1e-8)
    assert np.allclose(beam.y_angle(), main.y_angle, rtol=0.0, atol=1e-8)