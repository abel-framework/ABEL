# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

"""
ABEL : plasma lens tests
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



