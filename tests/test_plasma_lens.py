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



@pytest.mark.plasma_lens
@pytest.mark.impactx
def test_PlasmaLensImpactX():
    """
    Check that the ImpactX plasma lens agrees with the thick nonlinear lens.

    Both model the same drift-kick sequence, so they should agree to well
    within the difference between their slicing schemes.
    """

    np.random.seed(42)

    # set up beam
    source = SourceBasic()
    source.bunch_length = 100e-6 # [m]
    source.num_particles = 20000
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

    beam0 = source.track()

    def track(plasma_lens):
        plasma_lens.length = L_pl
        plasma_lens.radius = R_pl
        plasma_lens.current = I
        plasma_lens.rel_nonlinearity = 0.5
        plasma_lens.offset_x = -R_pl/4
        plasma_lens.offset_y = R_pl/8
        beam = copy.deepcopy(beam0)
        beam.transport(L_pl)
        beam = plasma_lens.track(beam)
        beam.transport(L_pl)
        return beam

    beam_thick = track(PlasmaLensNonlinearThick(num_slice=30))
    beam_impactx = track(PlasmaLensImpactX(num_slices=30))

    # charge is conserved (no apertures by default)
    assert np.isclose(beam_impactx.charge(), beam0.charge(), rtol=1e-15)
    assert np.isclose(beam_impactx.energy(), beam0.energy(), rtol=1e-6)

    # the two implementations agree
    assert np.isclose(beam_impactx.beam_size_x(), beam_thick.beam_size_x(), rtol=1e-3)
    assert np.isclose(beam_impactx.beam_size_y(), beam_thick.beam_size_y(), rtol=1e-3)
    assert np.isclose(beam_impactx.norm_emittance_x(), beam_thick.norm_emittance_x(), rtol=1e-3)
    assert np.isclose(beam_impactx.norm_emittance_y(), beam_thick.norm_emittance_y(), rtol=1e-3)

    # the transverse offsets deflect the beam by the expected amount
    assert np.isclose(beam_impactx.x_angle(), beam_thick.x_angle(), rtol=5e-3)
    assert np.isclose(beam_impactx.y_angle(), beam_thick.y_angle(), rtol=5e-3)


@pytest.mark.plasma_lens
@pytest.mark.impactx
def test_PlasmaLensImpactX_apertures():
    """
    Check that the ImpactX plasma lens clips charge outside the capillary
    when apertures are enabled.
    """

    np.random.seed(42)

    source = SourceBasic()
    source.bunch_length = 100e-6 # [m]
    source.num_particles = 10000
    source.charge = -SI.e * 1.0e10 # [C]
    source.energy = 1e9 # [eV]
    source.rel_energy_spread = 1e-5
    source.emit_nx, source.emit_ny = 1e-6, 1e-6 # [m rad]
    source.beta_x = 120.0 # [m] large beta => beam wider than the capillary
    source.beta_y = source.beta_x

    R_pl = 500e-6 # [m] roughly two beam sigmas

    plasma_lens = PlasmaLensImpactX(length=0.01, radius=R_pl, current=1e3, use_apertures=True)

    beam0 = source.track()
    beam = plasma_lens.track(copy.deepcopy(beam0))

    # charge outside the capillary has been removed
    assert beam.abs_charge() < beam0.abs_charge()
    assert np.all(np.sqrt(beam.xs()**2 + beam.ys()**2) <= R_pl*(1 + 1e-9))
