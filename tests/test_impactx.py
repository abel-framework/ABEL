# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import pytest
from abel import *

def make_source(energy=10e9):
    """Helper function making a beam source."""
    
    source = SourceBasic()
    source.energy = energy
    source.charge = -0.05e-9
    source.emit_nx = 10e-6
    source.emit_ny = 0.1e-6
    source.beta_x = 0.015*np.sqrt(source.energy/10e9)
    source.beta_y = source.beta_x
    source.bunch_length = 10e-15*SI.c
    source.rel_energy_spread = 1e-2
    source.num_particles = 10000
    source.spin_polarization = 0.9
    source.spin_polarization_direction = 'y'
    
    return source


def make_interstage(source, use_quads=False):
    """Helper function making an interstage."""
    
    # define interstage (plasma-lens-based)
    if use_quads:
        interstage = InterstageQuadsImpactX()
    else:
        interstage = InterstagePlasmaLensImpactX()
    interstage.nom_energy = source.energy
    interstage.R56 = 0.0
    interstage.beta0 = source.beta_x
    interstage.length_dipole = 1.0*np.sqrt(source.energy/10e9)
    interstage.field_dipole = 1.0
    interstage.cancel_chromaticity = True
    interstage.cancel_sec_order_dispersion = True
    interstage.use_apertures = False
    interstage.enable_csr = True
    interstage.enable_isr = True
    
    return interstage


@pytest.mark.impactx
def test_interstages_quads_impactx():
    """Test of the quad-based interstage using ImpactX."""
    
    # define electron source
    source = make_source()
    
    # define interstage (quadrupole-based)
    interstage_quads = make_interstage(source, use_quads=True)

    # track the particles
    beam0 = source.track()
    beam_quads = interstage_quads.track(beam0)


@pytest.mark.impactx
def test_interstage_plasma_lens_impactx():
    """Test of the plasma-lens-based interstage using ImpactX."""
    
    # define electron source
    source = make_source()
    
    # define interstage (plasma-lens-based)
    interstage_pl = make_interstage(source, use_quads=False)

    # track the particles
    beam0 = source.track()
    beam_pl = interstage_pl.track(beam0)

@pytest.mark.impactx
def test_interstage_plasma_lens_impactx_quality_preservation():
    """Test of the plasma-lens-based interstage using ImpactX."""
    
    # define electron source
    source = make_source()
    
    # define interstage (plasma-lens-based)
    interstage_pl = make_interstage(source, use_quads=False)

    # track the particles
    beam0 = source.track()
    beam_pl = interstage_pl.track(beam0)

    rtol0 = 0.05
    assert np.isclose(beam_pl.charge(), beam0.charge(), rtol=0.001)
    assert np.isclose(beam_pl.energy(), beam0.energy(), rtol=0.01)
    assert np.isclose(beam_pl.bunch_length(), beam0.bunch_length(), rtol=rtol0)
    assert np.isclose(beam_pl.norm_emittance_x(), beam0.norm_emittance_x(), rtol=rtol0)
    assert np.isclose(beam_pl.norm_emittance_y(), beam0.norm_emittance_y(), rtol=rtol0)
    assert np.isclose(beam_pl.rel_energy_spread(), beam0.rel_energy_spread(), rtol=0.01)
    assert np.isclose(beam_pl.beta_x(), beam0.beta_x(), rtol=rtol0)
    assert np.isclose(beam_pl.beta_y(), beam0.beta_y(), rtol=rtol0)
    assert np.isclose(beam_pl.spin_polarization_y(), beam0.spin_polarization_y(), rtol=rtol0)

@pytest.mark.impactx
def test_interstage_plasma_lens_impactx_spin_tracking():
    """Test of spin tracking in plasma-lens-based interstage using ImpactX."""
    
    # define electron source
    source = make_source()
    
    # define interstage (plasma-lens-based)
    interstage_pl = make_interstage(source, use_quads=False)
    
    # track the particles (x polarized; should be rotated)
    source.spin_polarization_direction = 'x'
    beam0_x = source.track()
    beam_pl_x = interstage_pl.track(beam0_x)
    assert np.isclose(beam_pl_x.spin_polarization(), beam0_x.spin_polarization(), atol=0.02)
    assert np.isclose(beam_pl_x.spin_polarization_x(), 0.42164940, atol=0.02)
    assert np.isclose(beam_pl_x.spin_polarization_y(), 0, atol=0.02)
    assert np.isclose(beam_pl_x.spin_polarization_z(), -0.79613911, atol=0.02)

    # track the particles (y polarized; should be preserved)
    source.spin_polarization_direction = 'y'
    beam0_y = source.track()
    beam_pl_y = interstage_pl.track(beam0_y)
    assert np.isclose(beam_pl_y.spin_polarization(), beam0_y.spin_polarization(), atol=0.02)
    assert np.isclose(beam_pl_y.spin_polarization_x(), 0, atol=0.02)
    assert np.isclose(beam_pl_y.spin_polarization_y(), beam_pl_y.spin_polarization_y(), atol=0.02)
    assert np.isclose(beam_pl_y.spin_polarization_z(), 0, atol=0.02)
    
    # track the particles (z polarized; should be rotated)
    source.spin_polarization_direction = 'z'
    beam0_z = source.track()
    beam_pl_z = interstage_pl.track(beam0_z)
    assert np.isclose(beam_pl_z.spin_polarization(), beam0_z.spin_polarization(), atol=0.02)
    assert np.isclose(beam_pl_z.spin_polarization_x(), 0.79808656, atol=0.02)
    assert np.isclose(beam_pl_z.spin_polarization_y(), 0, atol=0.02)
    assert np.isclose(beam_pl_z.spin_polarization_z(), 0.42291824, atol=0.02)
    


@pytest.mark.impactx
def test_interstage_plasma_lens_prealign():
    """Test the pre-alignment procedure of the plasma-lens-based interstage."""

    # reset the seed for reproducibility
    np.random.seed(42)
    
    # define two energies to test (low for CSR, high for ISR)
    Es = [2e9, 2000e9]
    for E in Es:
        print(E)
        # define electron source
        source = make_source(energy=E)
    
        # define interstage
        interstage_pl = make_interstage(source, use_quads=False)
        
        # make a beam
        beam0 = source.track()
        
        # track through the non-aligned interstage
        beam_before = interstage_pl.track(beam0, verbose=False)
        if E < 10e9:
            assert not np.isclose(abs(beam_before.x_offset()), 0.0, atol=1e-6)
        else:
            assert not np.isclose(abs(beam_before.x_offset()), 0.0, atol=1e-7)
        
        # re-do the pre-alignment (in parallel)
        interstage_pl.pre_align(source, parallel=True)
    
        # track through the pre-aligned interstage
        beam_after = interstage_pl.track(beam0, verbose=False)
        if E < 10e9:
            assert np.isclose(abs(beam_after.x_offset()), 0.0, atol=1e-6)
        else:
            assert np.isclose(abs(beam_after.x_offset()), 0.0, atol=1e-7)

        assert beam_after.norm_amplitude_x(beta0=source.beta_x) < beam_before.norm_amplitude_x(beta0=source.beta_x)

