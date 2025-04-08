# -*- coding: utf-8 -*-
"""
ABEL : Beam polarization tests
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

import os
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
    source.num_particles = 100
    beam = source.track()
    return beam

@pytest.mark.spin
def test_set_spin_unpolarized(beam):
    """Test for unpolarized spins to ensure it generates spins that are uniformly distributed on the unit sphere with
    no preferential polarization along any axis."""
    beam.set_spin_unpolarized()

    s_x, s_y, s_z = beam.sxs(), beam.sys(), beam.szs()
    norms = np.sqrt(s_x**2 + s_y**2 + s_z**2)
    
    assert np.allclose(norms, 1, atol=1e-6)
    assert np.abs(np.mean(s_z)) < 0.1
    assert np.abs(np.mean(s_x)) < 0.1
    assert np.abs(np.mean(s_y)) < 0.1
    
@pytest.mark.spin
def test_spin_polarized_x(beam): 
    "Test setting x-polarized spin." 
    beam.set_spin_polarized_x() 
    assert np.all(beam.sxs() == 1)
    assert np.all(beam.sys() == 0) 
    assert np.all(beam.szs() == 0)
    
@pytest.mark.spin
def test_spin_polarized_y(beam): 
    "Test setting y-polarized spin." 
    beam.set_spin_polarized_y() 
    assert np.all(beam.sxs() == 0)
    assert np.all(beam.sys() == 1) 
    assert np.all(beam.szs() == 0)
    
@pytest.mark.spin
def test_spin_polarized_z(beam): 
    "Test setting z-polarized spin." 
    beam.set_spin_polarized_z() 
    assert np.all(beam.sxs() == 0)
    assert np.all(beam.sys() == 0) 
    assert np.all(beam.szs() == 1)

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
def test_make_random_spins(beam): 
    "Test if make_random_spins generates correctly polarized spins."
    s_m = 0.85
    beam.make_random_spins(s_m)

    s_x, s_y, s_z = beam.sxs(), beam.sys(), beam.szs()
    
    assert np.isclose(np.mean(s_x), 0, atol= 0.05)
    assert np.isclose(np.mean(s_y), 0, atol= 0.05)
    assert np.isclose(np.mean(s_z), s_m, atol= 0.05)

    norms = np.sqrt(s_x**2 + s_y**2 + s_z**2)
    assert np.allclose(norms, 1, atol=1e-6)

@pytest.mark.spin
def test_make_random_spins_edge_cases(beam):
    "Test edge cases where polarization degree is -1, 0 or 1."
    for pol in [-1, 0, 1]:
        beam.make_random_spins(pol)
        assert np.isclose(np.mean(beam.szs()), pol, atol = 0.1)

@pytest.mark.spin
def test_make_random_spins_correlation(beam):
    "Test correlation between spin components by computing the correlation coefficients."
    s_m = 0.85
    beam.make_random_spins(s_m)
    
    s_x, s_y, s_z = beam.sxs(), beam.sys(), beam.szs()

    corr_xy = np.corrcoef(s_x, s_y)[0, 1]
    corr_xz = np.corrcoef(s_x, s_z)[0, 1]
    corr_yz = np.corrcoef(s_y, s_z)[0, 1]

    assert abs(corr_xy) < 0.1
    assert abs(corr_xz) < 0.15
    assert abs(corr_yz) < 0.15
    
@pytest.mark.spin
def test_renormalize_spin(beam):
    "Test renormalizing spin vectors"
    beam.set_sxs(np.ones(100) * 0.5)
    beam.set_sys(np.ones(100) * 0.5)
    beam.set_szs(np.ones(100) * 0.5)
    beam.renormalize_spin()

    norms = np.sqrt(beam.sxs()**2 + beam.sys()**2 + beam.szs()**2)
    assert np.allclose(norms, 1)
    
@pytest.mark.spin
def test_spin_polarization(beam): 
    "Test the overall spin polarization."
    beam.set_spin_polarized_x()
    assert np.isclose(beam.spin_polarization(),1)
