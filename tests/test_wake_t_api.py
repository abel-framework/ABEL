# -*- coding: utf-8 -*-
"""
ABEL : unit tests for the Wake-T API
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

import pytest, os
import numpy as np
import scipy.constants as SI
from abel.classes.source.impl.source_basic import SourceBasic
from abel.apis.wake_t.wake_t_api import *


def setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0):
    from abel.utilities.plasma_physics import beta_matched

    source = SourceBasic()
    source.bunch_length = 40.0e-06                                                  # [m], rms.
    source.num_particles = 10000                                               
    source.charge = -SI.e * 1.0e10                                                  # [C]

    # Energy parameters
    source.energy = 3.0e9                                                           # [eV]
    source.rel_energy_spread = 0.02                                                 # Relative rms energy spread

    # Emittances
    source.emit_nx, source.emit_ny = 15e-6, 0.1e-6                                    # [m rad]

    # Beta functions
    source.beta_x = beta_matched(plasma_density, source.energy) * ramp_beta_mag       # [m]
    source.beta_y = source.beta_x                                                     # [m]

    # Offsets
    source.z_offset = 0.00e-6                                                       # [m]

    # Other
    source.symmetrize_6d = True

    return source


@pytest.mark.wake_t_api_unit_test
def test_beam2wake_t_bunch():
    """
    Test for ``beam2wake_t_bunch()``.
    """

    source = setup_basic_source()
    beam = source.track()

    # Convert to a Wake-T particle bunch with beam2wake_t_bunch() 
    wake_t_bunch = beam2wake_t_bunch(beam)

    # Extract the phase space
    phasespace = wake_t_bunch.get_6D_matrix_with_charge()

    # Compare the ABEL phase space and Wake-T phase space
    assert np.allclose(len(beam.qs()), len(phasespace[6]), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.qs(), phasespace[6], rtol=1e-15, atol=0.0)
    assert np.allclose(beam.xs(), phasespace[0], rtol=1e-15, atol=0.0)
    assert np.allclose(beam.ys(), phasespace[2], rtol=1e-15, atol=0.0)
    assert np.allclose(beam.zs(), phasespace[4], rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pxs(), phasespace[1]*SI.c*SI.m_e, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pys(), phasespace[3]*SI.c*SI.m_e, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pzs(), phasespace[5]*SI.c*SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.location, wake_t_bunch.prop_distance, rtol=1e-15, atol=0.0)


