"""
ABEL : plasma lens tests
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
import os, copy
import scipy.constants as SI
import numpy as np

@pytest.mark.plasma_lens
def test_PlasmaLensNonlinearThin():
    """
    Check that the thin nonlinear plasma lens works as intended.
    """

    np.random.seed(42)

    # set up beam
    source = SourceBasic()
    source.bunch_length = 100e-6 # [m]
    source.num_particles = 50000
    source.charge = -SI.e * 1.0e10 # [C]
    source.energy = 1e9 # [eV]
    source.rel_energy_spread = 1e-5
    source.emit_nx, source.emit_ny = 1e-6, 1e-6 # [m rad]
    source.beta_x = 0.01 # [m]
    source.beta_y = source.beta_x

    # drift distance
    L_drift = 1.0 # [m]

    # lens length and radius
    L_pl = 0.001
    R_pl = 500e-6

    # calculate strength required to refocus in distance L
    f = (L_drift+L_pl/2)/2
    k = 1/(L_pl*f)
    g = k*source.energy/SI.c
    I = g*(2*np.pi*R_pl**2)/SI.mu_0
    
    # define plasma lens
    plasma_lens = PlasmaLensNonlinearThin()
    plasma_lens.length = L_pl
    plasma_lens.rel_nonlinearity = 0.0
    plasma_lens.radius = R_pl
    plasma_lens.current = I
    plasma_lens.offset_x = -R_pl/4
    
    # make beam
    beam0 = source.track()
    beam = copy.deepcopy(beam0)

    # transport to lens position
    beam.transport(L_pl)
    
    # track lens
    beam = plasma_lens.track(beam)
    
    # transport to focal location
    beam.transport(L_pl)

    assert np.isclose(beam.charge(), beam0.charge(), rtol=1e-15)
    assert np.isclose(beam.beam_size_x(), beam0.beam_size_x(), rtol=1e-1)
    assert np.isclose(beam.beam_size_y(), beam0.beam_size_y(), rtol=1e-1)
    assert np.isclose(beam.norm_emittance_x(), beam0.norm_emittance_x(), rtol=1e-1)
    assert np.isclose(beam.norm_emittance_y(), beam0.norm_emittance_y(), rtol=1e-1)
    assert np.isclose(beam.x_angle(), -plasma_lens.offset_x/f, atol=1e-4)
    assert np.isclose(beam.y_angle(), -plasma_lens.offset_y/f, atol=1e-4)


@pytest.mark.plasma_lens
def test_PlasmaLensNonlinearThick():
    """
    Check that the thick nonlinear plasma lens works as intended.
    """

    np.random.seed(42)

    # set up beam
    source = SourceBasic()
    source.bunch_length = 100e-6 # [m]
    source.num_particles = 50000
    source.charge = -SI.e * 1.0e10 # [C]
    source.energy = 1e9 # [eV]
    source.rel_energy_spread = 1e-5
    source.emit_nx, source.emit_ny = 1e-6, 1e-6 # [m rad]
    source.beta_x = 0.01 # [m]
    source.beta_y = source.beta_x

    # drift distance
    L_drift = 1.0 # [m]

    # lens length and radius
    L_pl = 0.01
    R_pl = 500e-6

    # calculate strength required to refocus in distance L
    f = (L_drift+L_pl/2)/2
    k = 1/(L_pl*f)
    g = k*source.energy/SI.c
    I = g*(2*np.pi*R_pl**2)/SI.mu_0
    
    # define plasma lens
    plasma_lens = PlasmaLensNonlinearThick()
    plasma_lens.length = L_pl
    plasma_lens.rel_nonlinearity = 0.0
    plasma_lens.radius = R_pl
    plasma_lens.current = I
    plasma_lens.offset_x = -R_pl/4
    
    # make beam
    beam0 = source.track()
    beam = copy.deepcopy(beam0)

    # transport to lens position
    beam.transport(L_pl)
    
    # track lens
    beam = plasma_lens.track(beam)
    
    # transport to focal location
    beam.transport(L_pl)

    assert np.isclose(beam.charge(), beam0.charge(), rtol=1e-15)
    assert np.isclose(beam.beam_size_x(), 7.25564785610576e-06, rtol=1e-1)
    assert np.isclose(beam.beam_size_y(), 7.25564785610576e-06, rtol=1e-1)
    assert np.isclose(beam.norm_emittance_x(), beam0.norm_emittance_x(), rtol=1e-1)
    assert np.isclose(beam.norm_emittance_y(), beam0.norm_emittance_y(), rtol=1e-1)
    assert np.isclose(beam.x_angle(), -plasma_lens.offset_x/f, atol=1e-4)
    assert np.isclose(beam.y_angle(), -plasma_lens.offset_y/f, atol=1e-4)



