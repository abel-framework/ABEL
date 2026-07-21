# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

"""
ABEL : BDS class tests
"""

import pytest
from abel import *

def setup_source():
    """
    Set up a basic source
    """
    source = SourceTrapezoid()
    source.charge = -100e-12 # [C]
    source.energy = 650e9 # [eV]
    source.rel_energy_spread = 0.001
    source.bunch_length = 25e-6 # [m]
    source.z_offset = 0e-6 # [m]
    source.current_head = 1.5e3
    source.gaussian_blur = 3e-6 # [m]
    source.emit_nx, source.emit_ny = 10e-6, 10e-6 # [m rad]
    source.beta_x = 5.0
    source.beta_y = source.beta_x
    source.num_particles = 50000
    return source


@pytest.mark.bds
def test_bds_plasmalens_impactx():
    """
    Test the plasma-lens-based BDS (with ImpactX)
    """

    # get a source
    source = setup_source()

    # define the BDS
    bds = BeamDeliverySystemPlasmaLensImpactX()
    bds.nom_energy = source.energy
    bds.beta0 = source.beta_x
    bds.cancel_chromaticity = True
    bds.L_star = 2.0 # [m]
    bds.length_scale = 2.5 # [m]
    bds.beta_star = 1.5e-3 # [m]
    bds.Dpx_star = 0.033 # [m]

    # make the beam
    beam0 = source.track()

    # track the beam through the BDS
    beam = bds.track(beam0)

    assert np.isclose(beam0.abs_charge(), beam.abs_charge(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam.beam_size_x(), 170e-9, rtol=1e-1, atol=0.0)
    assert np.isclose(beam.beam_size_y(), 134e-9, rtol=1e-1, atol=0.0)
    assert np.isclose(beam0.bunch_length(), beam.bunch_length(), rtol=1e-2, atol=0.0)

@pytest.mark.bds
def test_bds_plasmalens_basic():
    """
    Test the plasma-lens-based BDS (basic)
    """

    # get a source
    source = setup_source()

    # define the BDS
    bds = BeamDeliverySystemPlasmaLensBasic()
    bds.nom_energy = source.energy
    bds.beta0 = source.beta_x
    bds.cancel_chromaticity = True
    bds.L_star = 2.0 # [m]
    bds.length_scale = 2.5 # [m]
    bds.beta_star = 1.5e-3 # [m]
    bds.Dpx_star = 0.033 # [m]

    # make the beam
    beam0 = source.track()

    # track the beam through the BDS
    beam = bds.track(beam0)

    assert np.isclose(beam0.abs_charge(), beam.abs_charge(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam0.norm_emittance_x(), beam.norm_emittance_x(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam0.norm_emittance_y(), beam.norm_emittance_y(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam.beta_x(), bds.beta_star, rtol=1e-1, atol=0.0)
    assert np.isclose(beam.beta_y(), bds.beta_star, rtol=1e-1, atol=0.0)
    assert np.isclose(beam0.bunch_length(), beam.bunch_length(), rtol=1e-2, atol=0.0)

@pytest.mark.bds
def test_bds_basic():
    """
    Test the basic BDS
    """

    # get a source
    source = setup_source()

    # define the BDS
    bds = BeamDeliverySystemBasic()
    bds.nom_energy = source.energy
    bds.beta_x = 10e-3
    bds.beta_y = 0.1e-3
    bds.length = 500.0
    bds.bunch_length = 50e-6

    # make the beam
    beam0 = source.track()

    # track the beam through the BDS
    beam = bds.track(beam0)

    assert np.isclose(bds.get_length(), bds.length, rtol=1e-10, atol=0.0)
    assert np.isclose(beam0.abs_charge(), beam.abs_charge(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam0.norm_emittance_x(), beam.norm_emittance_x(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam0.norm_emittance_y(), beam.norm_emittance_y(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam.beta_x(), bds.beta_x, rtol=1e-1, atol=0.0)
    assert np.isclose(beam.beta_y(), bds.beta_y, rtol=1e-1, atol=0.0)
    assert np.isclose(beam0.bunch_length(), bds.bunch_length, rtol=1e-2, atol=0.0)
    
    