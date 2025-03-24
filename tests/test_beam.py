"""
ABEL : Beam class tests
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
import random


############# Basic tests #############

@pytest.mark.beam
def test_initialization():
    "Test if the ``Beam`` object initialises correctly."
    beam = Beam(num_particles=1)
    assert beam.num_bunches_in_train == 1
    assert beam.bunch_separation == 0.0
    assert beam.trackable_number == -1
    assert beam.stage_number == 0
    assert beam.location == 0


@pytest.mark.beam
def test_set_phase_space():
    "Verify that the phase space is set correctly."
    beam = Beam()
    num_particles = 10042
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.rand(num_particles)
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    assert len(beam) == num_particles
    assert np.isclose(beam.particle_mass, SI.m_e)
    assert np.allclose(beam.xs(), xs)
    assert np.allclose(beam.ys(), ys)
    assert np.allclose(beam.zs(), zs)
    assert np.allclose(beam.uxs(), uxs)
    assert np.allclose(beam.uys(), uys)
    assert np.allclose(beam.uzs(), uzs)
    assert np.allclose(beam.qs(), np.ones_like(xs)*Q/num_particles)
    assert np.isclose(beam.charge(), Q)
    assert np.allclose(beam.weightings(), np.ones_like(xs)*Q/num_particles/(-SI.e))

    assert np.allclose(beam.xps(), uxs/uzs)
    assert np.allclose(beam.yps(), uys/uzs)
    assert np.allclose(beam.pxs(), SI.m_e*uxs)
    assert np.allclose(beam.pys(), SI.m_e*uys)
    assert np.allclose(beam.pzs(), SI.m_e*uzs)
    assert np.allclose(beam.Es(), np.sqrt((SI.m_e*uzs*SI.c)**2 + (SI.m_e*SI.c**2)**2)/SI.e )


@pytest.mark.beam
def test_set_phase_space2():
    "Verify that the phase space is set correctly."
    beam = Beam()
    num_particles = 10042
    xs = np.random.normal(0.0, 2.0e-6, num_particles)
    ys = np.random.normal(0.0, 1.0e-6, num_particles)
    zs = np.random.normal(0.0, 50.0e-6, num_particles)
    xps = np.random.normal(0.0, 1.5e-4, num_particles)
    yps = np.random.normal(0.0, 1.0e-5, num_particles)
    Es = np.random.normal(500e9, 0.02*500e9, num_particles)
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs, xps=xps, yps=yps, Es=Es)

    assert len(beam) == num_particles
    assert np.isclose(beam.particle_mass, SI.m_e)
    assert np.allclose(beam.xs(), xs)
    assert np.allclose(beam.ys(), ys)
    assert np.allclose(beam.zs(), zs)
    assert np.allclose(beam.xps(), xps, rtol=1e-05, atol=1e-08)
    assert np.allclose(beam.yps(), yps, rtol=1e-05, atol=1e-08)
    assert np.allclose(beam.Es(), Es)
    assert np.allclose(beam.qs(), np.ones_like(xs)*Q/num_particles)
    assert np.isclose(beam.charge(), Q)
    assert np.allclose(beam.weightings(), np.ones_like(xs)*Q/num_particles/(-SI.e))

    assert np.allclose(beam.uzs(), np.sqrt((Es*SI.e/SI.m_e/SI.c)**2 - SI.c**2) )
    assert np.allclose(beam.uxs(), xps*beam.uzs())
    assert np.allclose(beam.uys(), yps*beam.uzs())
    assert np.allclose(beam.pxs(), SI.m_e*xps*beam.uzs(), rtol=1e-05, atol=1e-25)
    assert np.allclose(beam.pys(), SI.m_e*yps*beam.uzs(), rtol=1e-05, atol=1e-25)
    assert np.allclose(beam.pzs(), SI.m_e*beam.uzs(), rtol=1e-05, atol=1e-19)


@pytest.mark.beam
def test_set_phase_space3():
    "Verify that the phase space is set correctly."
    beam = Beam()
    num_particles = 10042
    xs = np.random.normal(1.0, 2.4e-6, num_particles)
    ys = np.random.normal(0.3, 1.1e-6, num_particles)
    zs = np.random.normal(10.0, 42.0e-6, num_particles)
    pxs = np.random.normal(0.0, 5.0e-22, num_particles)
    pys = np.random.normal(0.0, 5.0e-23, num_particles)
    pzs = np.random.normal(2.67e-16, 5.4e-18, num_particles)  # Corresponds to 500 GeV.
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs, pxs=pxs, pys=pys, pzs=pzs)

    assert len(beam) == num_particles
    assert np.isclose(beam.particle_mass, SI.m_e)
    assert np.allclose(beam.xs(), xs)
    assert np.allclose(beam.ys(), ys)
    assert np.allclose(beam.zs(), zs)
    assert np.allclose(beam.pxs(), pxs, rtol=1e-05, atol=1e-25)
    assert np.allclose(beam.pys(), pys, rtol=1e-05, atol=1e-25)
    assert np.allclose(beam.pzs(), pzs, rtol=1e-05, atol=1e-19)
    assert np.allclose(beam.qs(), np.ones_like(xs)*Q/num_particles)
    assert np.isclose(beam.charge(), Q)
    assert np.allclose(beam.weightings(), np.ones_like(xs)*Q/num_particles/(-SI.e))

    assert np.allclose(beam.xps(), pxs/pzs)
    assert np.allclose(beam.yps(), pys/pzs)
    assert np.allclose(beam.uxs(), pxs/SI.m_e)
    assert np.allclose(beam.uys(), pys/SI.m_e)
    assert np.allclose(beam.uzs(), pzs/SI.m_e)
    assert np.allclose(beam.Es(), np.sqrt((pzs*SI.c)**2 + (SI.m_e*SI.c**2)**2)/SI.e )


@pytest.mark.beam
def test_set_phase_space4():
    "Verify that the phase space is set correctly."
    beam = Beam()
    num_particles = 10042
    xs = np.random.normal(0.0, 2.0e-6, num_particles)
    ys = np.random.normal(0.0, 1.0e-6, num_particles)
    zs = np.random.normal(0.0, 50.0e-6, num_particles)
    uxs = np.random.normal(0.0, 250.0e6, num_particles)
    yps = np.random.normal(0.0, 1.0e-5, num_particles)
    pzs = np.random.normal(1.6e-18, 3.0e-20, num_particles)
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs, uxs=uxs, yps=yps, pzs=pzs)

    assert len(beam) == num_particles
    assert np.isclose(beam.particle_mass, SI.m_e)
    assert np.allclose(beam.xs(), xs)
    assert np.allclose(beam.ys(), ys)
    assert np.allclose(beam.zs(), zs)
    assert np.allclose(beam.uxs(), uxs)
    assert np.allclose(beam.yps(), yps)
    assert np.allclose(beam.pzs(), pzs, rtol=1e-05, atol=1e-25)
    assert np.allclose(beam.qs(), np.ones_like(xs)*Q/num_particles)
    assert np.isclose(beam.charge(), Q)
    assert np.allclose(beam.weightings(), np.ones_like(xs)*Q/num_particles/(-SI.e))

    assert np.allclose(beam.uzs(), pzs/SI.m_e)
    assert np.allclose(beam.xps(), uxs/beam.uzs())
    assert np.allclose(beam.uys(), yps*beam.uzs())
    assert np.allclose(beam.pxs(), SI.m_e*uxs, rtol=1e-05, atol=1e-25)
    assert np.allclose(beam.pys(), SI.m_e*yps*beam.uzs(), rtol=1e-05, atol=1e-25)    
    assert np.allclose(beam.Es(), np.sqrt((pzs*SI.c)**2 + (SI.m_e*SI.c**2)**2)/SI.e )


@pytest.mark.beam
def test_reset_phase_space():
    "Test reset_phase_space to ensure it initializes an 8xN zero matrix for the specified number of particles."
    beam = Beam()
    beam.reset_phase_space(10)
    assert beam._Beam__phasespace.shape == (8, 10)
    assert (beam._Beam__phasespace == 0).all()


@pytest.mark.beam
def test_delitem():
    "Test deleting macroparticles by indices or masks."
    beam = Beam()
    beam.reset_phase_space(5)
    beam.__delitem__([0, 2])
    assert beam._Beam__phasespace.shape[1] == 3  # Two particles removed


@pytest.mark.beam
def test_remove_nans():
    "Verify that remove_nans removes particles with any NaN value."
    beam = Beam()
    beam.reset_phase_space(10)
    beam._Beam__phasespace[0, 1] = float('nan')  # Introduce NaN in one particle
    beam._Beam__phasespace[0, 4] = float('nan')  # Introduce NaN in one particle
    beam._Beam__phasespace[1, 4] = float('nan')  # Introduce NaN in one particle
    beam._Beam__phasespace[1, 5] = float('nan')  # Introduce NaN in one particle
    beam._Beam__phasespace[5, 7] = float('nan')  # Introduce NaN in one particle
    beam.remove_nans()
    assert len(beam) == 6


@pytest.mark.beam
def test_add_beams():
    "Test the __add__ operator."
    beam1 = Beam()
    num_particles1 = 1021
    beam1.reset_phase_space(num_particles1)
    beam2 = Beam()
    num_particles2 = 42
    beam2.reset_phase_space(num_particles2)
    beam3 = beam1 + beam2
    assert len(beam3) == num_particles1 + num_particles2


@pytest.mark.beam
def test_iadd_beams():
    "Test the __iadd__ operator."
    beam1 = Beam()
    num_particles1 = 3044
    beam1.reset_phase_space(num_particles1)
    beam2 = Beam()
    num_particles2 = 4444
    beam2.reset_phase_space(num_particles2)
    beam1 += beam2
    assert len(beam1) == num_particles1 + num_particles2


@pytest.mark.beam
def test_getitem():
    "Test retrieving a single particle's data by index."

    beam = Beam()
    xs = np.random.rand(1000)
    ys = np.random.rand(1000)
    zs = np.random.rand(1000)
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs)
    random_integer = random.randint(0, 999)
    particle = beam[random_integer]
    assert particle.shape == (8,)
    assert np.allclose(particle[0], xs[random_integer])
    assert np.allclose(particle[1], ys[random_integer])
    assert np.allclose(particle[2], zs[random_integer])


@pytest.mark.beam
def test_len():
    "Verify that __len__ returns the correct number of particles."
    random_integer = random.randint(0, 30000)
    beam = Beam()
    beam.reset_phase_space(random_integer)
    assert len(beam) == random_integer


# @pytest.mark.beam
# def test_str():
#     "Test the __str__ method for correct formatting of beam properties."
#     beam = Beam()
#     beam.reset_phase_space(5)
#     string_repr = str(beam)
#     assert "Beam:" in string_repr
#     assert "5 macroparticles" in string_repr


@pytest.mark.beam
def test_copy_particle_charge():

    num_particles = 8051

    beam1 = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.rand(num_particles)
    Q = -SI.e * 1.0e10
    beam1.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    beam2 = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.rand(num_particles)
    Q = -SI.e * 3.4e10
    ws = np.random.rand(num_particles)*Q/(-SI.e)
    beam2.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs, weightings=ws)
    median_charge = np.median(beam2._Beam__phasespace[6, :])  # Extract expected median charge

    beam1.copy_particle_charge(beam2)
    assert (beam1._Beam__phasespace[6, :] == median_charge).all()  # Verify charge is copied correctly


@pytest.mark.beam
def test_scale_charge():
    "Test correct scaling of particle charges, scaling to a larger or smaller total charge. Edge case: scaling to zero charge."

    beam = Beam()
    num_particles = 4352
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs)
    original_charges = copy.deepcopy(beam.qs())

    # Scale up charge
    beam.scale_charge(-SI.e * 2.0e10)
    scaled_charges = beam.qs()
    assert np.isclose(scaled_charges.sum(), -SI.e * 2.0e10)  # Check total charge
    assert np.allclose(scaled_charges / original_charges, 2.0)  # Charges scaled proportionally

    # Scale down charge
    beam.scale_charge(-SI.e * 0.5e10)
    assert np.isclose(scaled_charges.sum(), -SI.e * 0.5e10)

    # Edge case: zero charge
    beam.scale_charge(0)
    scaled_charges = beam.qs()
    assert np.isclose(scaled_charges.sum(), 0)


@pytest.mark.beam
def test_scale_energy():
    "Test correct scaling of particle energies, scaling to a higher or lower total energy. Edge case: scaling to zero energy."

    beam = Beam()
    num_particles = 4352
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    Es = np.random.normal(1e9, 0.02*1e9, num_particles)
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs, Es=Es)
    original_energies = copy.deepcopy(beam.Es())

    # Scale up energy
    beam.scale_energy(4e9)
    scaled_energies = beam.Es()
    scaled_mean_energy = weighted_mean(scaled_energies, beam.weightings())
    assert np.isclose(scaled_mean_energy, 4e9, rtol=0.01, atol=1e8)
    assert np.allclose(scaled_energies / original_energies, 4.0, rtol=0.01, atol=0.1)  # Energies scaled proportionally

    # Scale down energy
    beam.scale_energy(1.7e8)
    scaled_energies = beam.Es()
    scaled_mean_energy = weighted_mean(scaled_energies, beam.weightings())
    assert np.isclose(scaled_mean_energy, 1.7e8, rtol=0.01, atol=1e8)

    # # Edge case: zero energy
    # try:
    #     beam.scale_energy(0)
    # except ValueError as e:
    #     assert str(e) == "Energy cannot be scaled to zero"


############# Tests on bunch pattern #############




############# Tests of rotation methods #############