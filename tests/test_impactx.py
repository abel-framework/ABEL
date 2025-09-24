import pytest
from abel import *

def make_source():
    """Helper function making a beam source."""
    
    source = SourceBasic()
    source.energy = 10e9
    source.charge = -0.05e-9
    source.emit_nx = 10e-6
    source.emit_ny = 0.1e-6
    source.beta_x = 0.015
    source.beta_y = source.beta_x
    source.bunch_length = 10e-15*SI.c
    source.rel_energy_spread = 1e-2
    source.num_particles = 10000
    return source


@pytest.mark.impactx
def test_interstages_quads_impactx():
    """Test of the quad-based interstage using ImpactX."""
    
    # define electron source
    source = make_source()
    
    # define interstage (quadrupole-based)
    interstage_quads = InterstageQuadsImpactX()
    interstage_quads.nom_energy = source.energy
    interstage_quads.R56 = 0.0
    interstage_quads.beta0 = source.beta_x
    interstage_quads.length_dipole = 1.0
    interstage_quads.field_dipole = 1.0
    interstage_quads.cancel_chromaticity = True
    interstage_quads.cancel_sec_order_dispersion = True

    # track the particles
    beam0 = source.track()
    beam_quads = interstage_quads.track(beam0)


@pytest.mark.impactx
def test_interstage_plasma_lens_impactx():
    """Test of the plasma-lens-based interstage using ImpactX."""
    
    # define electron source
    source = make_source()
    
    # define interstage (plasma-lens-based)
    interstage_pl = InterstagePlasmaLensImpactX()
    interstage_pl.nom_energy = source.energy
    interstage_pl.R56 = 0.0
    interstage_pl.beta0 = source.beta_x
    interstage_pl.length_dipole = 1.0
    interstage_pl.field_dipole = 1.0
    interstage_pl.cancel_chromaticity = True
    interstage_pl.cancel_sec_order_dispersion = True

    # track the particles
    beam0 = source.track()
    beam_pl = interstage_pl.track(beam0)

    
    