# -*- coding: utf-8 -*-
# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

"""
ABEL : Beam polarization unit tests
"""

#import os
import pytest

from abel import *

@pytest.fixture
def beam():
    source = SourceBasic()
    source.charge = -1e10 * SI.e # [C]
    source.energy = 5e9 # [eV]
    source.rel_energy_spread = 0.01
    source.bunch_length = 18e-6 # [m]
    source.z_offset = -36e-6 # [m]
    source.emit_nx, source.emit_ny = 160e-6, 0.56e-6 # [m rad]
    source.beta_x = 1
    source.beta_y = source.beta_x
    source.num_particles = 10000
    beam = source.track()
    return beam

@pytest.mark.spin
def test_set_spin_unpolarized(beam):
    """
    Test for unpolarized spins to ensure it generates spins that are uniformly 
    distributed on the unit sphere with no preferential polarization along any 
    axis.
    """
    beam.set_spin_unpolarized()

    s_x, s_y, s_z = beam.spxs(), beam.spys(), beam.spzs()
    norms = np.sqrt(s_x**2 + s_y**2 + s_z**2)
    
    assert np.allclose(norms, 1, atol=1e-6)
    assert np.abs(np.mean(s_z)) < 0.1
    assert np.abs(np.mean(s_x)) < 0.1
    assert np.abs(np.mean(s_y)) < 0.1
    
@pytest.mark.spin
def test_spin_polarized_x(beam): 
    "Test setting x-polarized spin." 
    beam.set_spin_polarized_x() 
    assert np.all(beam.spxs() == 1)
    assert np.all(beam.spys() == 0) 
    assert np.all(beam.spzs() == 0)
    
@pytest.mark.spin
def test_spin_polarized_y(beam): 
    "Test setting y-polarized spin." 
    beam.set_spin_polarized_y() 
    assert np.all(beam.spxs() == 0)
    assert np.all(beam.spys() == 1) 
    assert np.all(beam.spzs() == 0)
    
@pytest.mark.spin
def test_spin_polarized_z(beam): 
    "Test setting z-polarized spin." 
    beam.set_spin_polarized_z() 
    assert np.all(beam.spxs() == 0)
    assert np.all(beam.spys() == 0) 
    assert np.all(beam.spzs() == 1)

@pytest.mark.spin
def test_spin_check_x(beam):
    "Test if the beam detects zero-lenght spin."
    beam.set_spin_polarized_x()
    assert beam.spin_check()

@pytest.mark.spin
def test_spin_check_y(beam):
    "Test if the beam detects zero-lenght spin."
    beam.set_spin_polarized_y()
    assert beam.spin_check()
    
@pytest.mark.spin
def test_spin_check_z(beam):
    "Test if the beam detects zero-lenght spin."
    beam.set_spin_polarized_z()
    assert beam.spin_check()

@pytest.mark.spin 
def test_set_arbitrary_spin_polarization_z(beam): 
    "Test if make_random_spins generates correctly polarized spins."
    s_m = 0.85
    beam.set_arbitrary_spin_polarization(s_m, 'z')

    s_x, s_y, s_z = beam.spxs(), beam.spys(), beam.spzs()
    
    assert np.isclose(np.mean(s_x), 0, atol= 0.05)
    assert np.isclose(np.mean(s_y), 0, atol= 0.05)
    assert np.isclose(np.mean(s_z), s_m, atol= 0.05)

    norms = np.sqrt(s_x**2 + s_y**2 + s_z**2)
    assert np.allclose(norms, 1, atol=1e-6)

@pytest.mark.spin 
def test_set_arbitrary_spin_polarization_y(beam): 
    "Test if make_random_spins generates correctly polarized spins."
    s_m = 0.85
    beam.set_arbitrary_spin_polarization(s_m, 'y')

    s_x, s_y, s_z = beam.spxs(), beam.spys(), beam.spzs()
    
    assert np.isclose(np.mean(s_x), 0, atol= 0.05)
    assert np.isclose(np.mean(s_y), s_m, atol= 0.05)
    assert np.isclose(np.mean(s_z), 0, atol= 0.05)

    norms = np.sqrt(s_x**2 + s_y**2 + s_z**2)
    assert np.allclose(norms, 1, atol=1e-6)
    
@pytest.mark.spin 
def test_set_arbitrary_spin_polarization_x(beam): 
    "Test if make_random_spins generates correctly polarized spins."
    s_m = 0.85
    beam.set_arbitrary_spin_polarization(s_m, 'x')

    s_x, s_y, s_z = beam.spxs(), beam.spys(), beam.spzs()
    
    assert np.isclose(np.mean(s_x), s_m, atol= 0.05)
    assert np.isclose(np.mean(s_y), 0, atol= 0.05)
    assert np.isclose(np.mean(s_z), 0, atol= 0.05)

    norms = np.sqrt(s_x**2 + s_y**2 + s_z**2)
    assert np.allclose(norms, 1, atol=1e-6)


@pytest.mark.spin
def test_set_arbitrary_spin_polarization_edge_cases(beam):
    "Test edge cases where polarization degree is -1, 0 or 1."
    for pol in [-1, 0, 1]:
        beam.set_arbitrary_spin_polarization(pol, 'z')
        assert np.isclose(np.mean(beam.spzs()), pol, atol = 0.1)

    
@pytest.mark.spin
def test_renormalize_spin(beam):
    "Test renormalizing spin vectors"
    beam.set_spxs(np.ones(len(beam)) * 0.5)
    beam.set_spys(np.ones(len(beam)) * 0.5)
    beam.set_spzs(np.ones(len(beam)) * 0.5)
    beam.renormalize_spin()

    norms = np.sqrt(beam.spxs()**2 + beam.spys()**2 + beam.spzs()**2)
    assert np.allclose(norms, 1)
    
@pytest.mark.spin
def test_spin_polarization(beam): 
    "Test the overall spin polarization."
    beam.set_spin_polarized_x()
    assert np.isclose(beam.spin_polarization(),1)
