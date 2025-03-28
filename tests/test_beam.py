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
from abel.utilities.beam_physics import generate_trace_space
import random


def setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, bunch_length=40.0e-06, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0):
    source = SourceBasic()
    source.bunch_length = bunch_length                                              # [m], rms.
    source.num_particles = 10000                                               
    source.charge = -e * 1.0e10                                                     # [C]

    # Energy parameters
    source.energy = energy                                                          # [eV]
    source.rel_energy_spread = 0.02                                                 # Relative rms energy spread

    # Emittances
    source.emit_nx, source.emit_ny = 15e-6, 0.1e-6                                  # [m rad]

    # Beta functions
    source.beta_x = beta_matched(plasma_density, source.energy) * ramp_beta_mag     # [m]
    source.beta_y = source.beta_x                                                   # [m]

    # Offsets
    source.z_offset = z_offset                                                      # [m]
    source.x_offset = x_offset                                                      # [m]
    source.y_offset = y_offset                                                      # [m]
    source.x_angle = x_angle                                                        # [rad]
    source.y_angle = y_angle                                                        # [rad]

    # Other
    source.symmetrize_6d = True

    return source


def setup_trapezoid_source(current_head=0.1e3, bunch_length=1050e-6, z_offset=1615e-6, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0, enable_xy_jitter=False, enable_xpyp_jitter=False):
    source = SourceTrapezoid()
    source.current_head = current_head                                              # [A]
    source.bunch_length = bunch_length                                              # [m]

    source.num_particles = 30000                                                 
    source.charge = 5.0e10 * -SI.e                                                  # [C]
    source.energy = 4.9e9                                                           # [eV] 
    source.gaussian_blur = 50e-6                                                    # [m]
    source.rel_energy_spread = 0.01                                              

    source.emit_nx, source.emit_ny = 50e-6, 100e-6                                  # [m rad]
    source.beta_x, source.beta_y = 0.5, 0.5                                         # [m]

     # Offsets
    source.z_offset = z_offset                                                      # [m]
    source.x_offset = x_offset                                                      # [m]
    source.y_offset = y_offset                                                      # [m]
    source.x_angle = x_angle                                                        # [rad]
    source.y_angle = y_angle                                                        # [rad]

    if enable_xy_jitter:
        source.jitter.x = 100e-9                                                    # [m], std
        source.jitter.y = 100e-9                                                    # [m], std

    if enable_xpyp_jitter:
        source.jitter.xp = 1.0e-6                                                   # [rad], std
        source.jitter.yp = 1.0e-6                                                   # [rad], std

    source.symmetrize = True

    return source


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
    Q = -SI.e * 1.0e10
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)

    energy_thres = 1001*SI.m_e*SI.c**2/SI.e  # [eV], 1000 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    uzs = random.uniform(uz_thres, energy2proper_velocity(10e12, unit='eV', m=SI.m_e))

    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    assert len(beam) == num_particles
    assert np.isclose(beam.particle_mass, SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge(), Q, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.abs_charge(), np.abs(Q), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.total_particles(), 1.0e10, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.weightings(), np.ones_like(xs)*Q/num_particles/(-SI.e))
    assert np.allclose(beam.xs(), xs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.ys(), ys, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.zs(), zs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uxs(), uxs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uys(), uys, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uzs(), uzs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.qs(), np.ones_like(xs)*Q/num_particles, rtol=1e-15, atol=0.0)

    assert np.allclose(beam.xps(), uxs/uzs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.yps(), uys/uzs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pxs(), SI.m_e*uxs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pys(), SI.m_e*uys, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pzs(), SI.m_e*uzs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.Es(), np.sqrt((SI.m_e*uzs*SI.c)**2 + (SI.m_e*SI.c**2)**2)/SI.e, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_set_phase_space2():
    "Verify that the phase space is set correctly."

    np.random.seed(42)

    beam = Beam()
    num_particles = 100042
    xs = np.random.normal(1.1e-6, 2.0e-6, num_particles)
    ys = np.random.normal(0.3e-6, 1.0e-6, num_particles)
    zs = np.random.normal(5.6e-6, 50.0e-6, num_particles)
    xps = np.random.normal(1.1e-5, 1.5e-7, num_particles)
    yps = np.random.normal(3.2e-6, 1.0e-8, num_particles)
    Es = np.random.normal(500e9, 0.02*500e9, num_particles)
    #weightings =  # TODO: make an array with varying weightings
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs, xps=xps, yps=yps, Es=Es)

    assert len(beam) == num_particles
    assert np.isclose(beam.particle_mass, SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge(), Q, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge_sign(), -1, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.total_energy(), SI.e * np.nansum(beam.weightings()*beam.Es()), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.weightings(), np.ones_like(xs)*Q/num_particles/(-SI.e), rtol=1e-15, atol=0.0)

    assert np.isclose(beam.energy(), 500e9, rtol=1e-4, atol=1e9)
    assert np.isclose(beam.gamma(), 500e9*SI.e/SI.m_e/SI.c**2, rtol=1e-3, atol=200)
    assert np.isclose(beam.rel_energy_spread(), 0.02, rtol=0.001, atol=0.001)
    assert np.isclose(beam.z_offset(), 5.6e-6, rtol=0.3, atol=0.0)
    assert np.isclose(beam.x_offset(), 1.1e-6, rtol=0.05, atol=0.0)
    assert np.isclose(beam.y_offset(), 0.3e-6, rtol=0.03, atol=0.0)
    assert np.isclose(beam.bunch_length(), 50.0e-6, rtol=1e-4, atol=0.00)
    assert np.isclose(beam.beam_size_x(), 2.0e-6, rtol=0.005, atol=0.03)
    assert np.isclose(beam.beam_size_y(), 1.0e-6, rtol=0.005, atol=0.03)
    assert np.isclose(beam.x_angle(), 1.1e-5, rtol=1e-5, atol=0.0)
    assert np.isclose(beam.y_angle(), 3.2e-6, rtol=1e-4, atol=0.0)
    assert np.isclose(beam.divergence_x(), 1.5e-7, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.divergence_y(), 1.0e-8, rtol=1e-2, atol=0.0)

    assert np.allclose(beam.xs(), xs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.ys(), ys, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.zs(), zs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.xps(), xps, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.yps(), yps, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.Es(), Es, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.qs(), np.ones_like(xs)*Q/num_particles, rtol=1e-15, atol=0.0)

    assert np.allclose(beam.uzs(), np.sqrt((Es*SI.e/SI.m_e/SI.c)**2 - SI.c**2), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uxs(), xps*beam.uzs(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uys(), yps*beam.uzs(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pxs(), SI.m_e*xps*beam.uzs(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pys(), SI.m_e*yps*beam.uzs(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pzs(), SI.m_e*beam.uzs(), rtol=1e-15, atol=0.0)


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
    assert np.isclose(beam.particle_mass, SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge(), Q, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.weightings(), np.ones_like(xs)*Q/num_particles/(-SI.e), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.xs(), xs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.ys(), ys, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.zs(), zs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pxs(), pxs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pys(), pys, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pzs(), pzs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.qs(), np.ones_like(xs)*Q/num_particles, rtol=1e-15, atol=0.0)

    assert np.allclose(beam.xps(), pxs/pzs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.yps(), pys/pzs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uxs(), pxs/SI.m_e, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uys(), pys/SI.m_e, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uzs(), pzs/SI.m_e, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.Es(), np.sqrt((pzs*SI.c)**2 + (SI.m_e*SI.c**2)**2)/SI.e, rtol=1e-15, atol=0.0)


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
    assert np.allclose(beam.xs(), xs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.ys(), ys, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.zs(), zs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uxs(), uxs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.yps(), yps, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pzs(), pzs, rtol=1e-05, atol=1e-25)
    assert np.allclose(beam.qs(), np.ones_like(xs)*Q/num_particles, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge(), Q, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.weightings(), np.ones_like(xs)*Q/num_particles/(-SI.e), rtol=1e-15, atol=0.0)

    assert np.allclose(beam.uzs(), pzs/SI.m_e, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.xps(), uxs/beam.uzs(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uys(), yps*beam.uzs(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pxs(), SI.m_e*uxs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pys(), SI.m_e*yps*beam.uzs(), rtol=1e-15, atol=0.0)    
    assert np.allclose(beam.Es(), np.sqrt((pzs*SI.c)**2 + (SI.m_e*SI.c**2)**2)/SI.e, rtol=1e-15, atol=0.0)


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
    num_particles = 30001
    beam = Beam()
    beam.reset_phase_space(num_particles)
    assert len(beam) == num_particles


# @pytest.mark.beam TODO
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
    energy_thres = 1001*SI.m_e*SI.c**2/SI.e  # [eV], 1000 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam1 = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = random.uniform(uz_thres, energy2proper_velocity(10e12, unit='eV', m=SI.m_e))
    Q = -SI.e * 1.0e10
    beam1.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    beam2 = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = random.uniform(uz_thres, energy2proper_velocity(10e12, unit='eV', m=SI.m_e))
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
    assert np.isclose(scaled_charges.sum(), -SI.e * 2.0e10, rtol=1e-15, atol=0.0)  # Check total charge
    assert np.allclose(scaled_charges / original_charges, 2.0, rtol=1e-15, atol=0.0)  # Charges scaled proportionally

    # Scale down charge
    beam.scale_charge(-SI.e * 0.5e10)
    assert np.isclose(scaled_charges.sum(), -SI.e * 0.5e10, rtol=1e-15, atol=0.0)

    # Edge case: zero charge
    beam.scale_charge(0)
    scaled_charges = beam.qs()
    assert np.isclose(scaled_charges.sum(), 0, rtol=1e-15, atol=0.0)


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

    # Edge case: zero energy
    try:
        beam.scale_energy(0)
    except ValueError as err:
        assert str(err) == 'Es contains values that are too small.'



############# Tests on bunch pattern #############
# TODO



############# Tests for beam statistics #############
@pytest.mark.beam
def test_param_calcs_generate_trace_space():
    "Test the beam parameter calculations using a trace space generated with generate_trace_space()."
    
    np.random.seed(42)

    alpha_x = -0.239
    alpha_y = -0.171
    beta_x = 0.120                                                                    # [m]
    beta_y = 0.120                                                                    # [m]
    geo_emitt_x = 2.552865e-09                                                        # [m rad]
    geo_emitt_y = 1.750833e-11                                                        # [m rad]

    # Generate trace spaces
    xs, xps = generate_trace_space(geo_emitt_x, beta_x, alpha_x, 10011, symmetrize=False)
    ys, yps = generate_trace_space(geo_emitt_y, beta_y, alpha_y, 10011, symmetrize=False)
    zs = np.random.normal(loc=0.0, scale=40.0e-6, size=10011)
    Es = np.random.normal(loc=3e9, scale=0.02*3e9, size=10011)

    beam = Beam()
    beam.set_phase_space(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=-e*1.0e10)

    # Examine the beam parameters
    assert np.isclose(np.std(xs), np.sqrt(geo_emitt_x*beta_x), rtol=1e-2, atol=0.0)  # Beam size
    assert np.isclose(np.std(ys), np.sqrt(geo_emitt_y*beta_y), rtol=1e-2, atol=0.0)
    assert np.isclose(np.std(Es), 0.02*3e9, rtol=0.01, atol=0.0)
    assert np.isclose(np.std(zs), 50.0e-6, rtol=0.005, atol=0.03)
    assert np.isclose(beam.beam_size_x(), np.sqrt(geo_emitt_x*beta_x), rtol=1e-2, atol=0.0)  # Beam size
    assert np.isclose(beam.beam_size_y(), np.sqrt(geo_emitt_y*beta_y), rtol=1e-2, atol=0.0)
    assert np.isclose(beam.divergence_x(), np.sqrt(geo_emitt_x/beta_x), rtol=5e-2, atol=0.0)
    assert np.isclose(beam.divergence_y(), np.sqrt(geo_emitt_y/beta_y), rtol=5e-2, atol=0.0)
    assert np.isclose(beam.energy_spread(), 0.02*3e9, rtol=0.01, atol=0.0)
    assert np.isclose(beam.bunch_length(), 50.0e-6, rtol=0.005, atol=0.03)

    assert np.isclose(beam.alpha_x(), alpha_x, rtol=6e-2, atol=0.0)
    assert np.isclose(beam.alpha_y(), alpha_y, rtol=6e-2, atol=0.0)
    assert np.isclose(beam.beta_x(), beta_x, rtol=2e-3, atol=0.0)
    assert np.isclose(beam.beta_y(), beta_y, rtol=1.5e-2, atol=0.0)
    assert np.isclose(beam.geom_emittance_x(), geo_emitt_x, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.geom_emittance_y(), geo_emitt_y, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_x(), geo_emitt_x*beam.gamma(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), geo_emitt_y*beam.gamma(), rtol=1e-2, atol=0.0)


@pytest.mark.beam
def test_param_calcs_generate_trace_space_xy():
    "Test the beam parameter calculations using a trace space generated with generate_trace_space_xy()."
    
    np.random.seed(42)

    alpha_x = -0.239
    alpha_y = -0.171
    beta_x = 0.120                                                                    # [m]
    beta_y = 0.120                                                                    # [m]
    geo_emitt_x = 2.552865e-09                                                        # [m rad]
    geo_emitt_y = 1.750833e-11                                                        # [m rad]
    num_particles = 200011

    # Generate trace space
    xs, xps, ys, yps = generate_trace_space_xy(geo_emitt_x, beta_x, alpha_x, geo_emitt_y, beta_y, alpha_y, num_particles, L=0, symmetrize=False)
    zs = np.random.normal(loc=0.0, scale=40.0e-6, size=num_particles)
    Es = np.random.normal(loc=3e9, scale=0.02*3e9, size=num_particles)

    beam = Beam()
    beam.set_phase_space(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=-e*1.0e10)

    # Examine the beam parameters
    assert np.isclose(np.std(xs), np.sqrt(geo_emitt_x*beta_x), rtol=1e-2, atol=0.0)  # Beam size
    assert np.isclose(np.std(ys), np.sqrt(geo_emitt_y*beta_y), rtol=1e-2, atol=0.0)
    assert np.isclose(np.std(Es), 0.02*3e9, rtol=0.005, atol=0.0)
    assert np.isclose(np.std(zs), 50.0e-6, rtol=0.005, atol=0.03)
    assert np.isclose(beam.beam_size_x(), np.sqrt(geo_emitt_x*beta_x), rtol=1e-3, atol=0.0)  # Beam size
    assert np.isclose(beam.beam_size_y(), np.sqrt(geo_emitt_y*beta_y), rtol=5e-3, atol=0.0)
    assert np.isclose(beam.divergence_x(), np.sqrt(geo_emitt_x/beta_x), rtol=5e-2, atol=0.0)
    assert np.isclose(beam.divergence_y(), np.sqrt(geo_emitt_y/beta_y), rtol=5e-2, atol=0.0)
    assert np.isclose(beam.energy_spread(), 0.02*3e9, rtol=0.005, atol=0.0)
    assert np.isclose(beam.bunch_length(), 50.0e-6, rtol=0.005, atol=0.03)

    assert np.isclose(beam.alpha_x(), alpha_x, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.alpha_y(), alpha_y, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.beta_x(), beta_x, rtol=1e-3, atol=0.0)
    assert np.isclose(beam.beta_y(), beta_y, rtol=3e-3, atol=0.0)
    assert np.isclose(beam.geom_emittance_x(), geo_emitt_x, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.geom_emittance_y(), geo_emitt_y, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_x(), geo_emitt_x*beam.gamma(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), geo_emitt_y*beam.gamma(), rtol=1e-2, atol=0.0)

    # Perform the same control with a symmetrised phase space
    xs, xps, ys, yps = generate_trace_space_xy(geo_emitt_x, beta_x, alpha_x, geo_emitt_y, beta_y, alpha_y, num_particles, L=0, symmetrize=True)
    num_tiling = 4
    num_particles_actual = round(num_particles/num_tiling)
    zs = np.random.normal(loc=0.0, scale=40.0e-6, size=num_particles_actual)
    Es = np.random.normal(loc=3e9, scale=0.02*3e9, size=num_particles_actual)
    zs = np.tile(zs, num_tiling)
    Es = np.tile(Es, num_tiling)

    beam = Beam()
    beam.set_phase_space(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=-e*1.0e10)

    assert np.isclose(np.std(xs), np.sqrt(geo_emitt_x*beta_x), rtol=1e-2, atol=0.0)  # Beam size
    assert np.isclose(np.std(ys), np.sqrt(geo_emitt_y*beta_y), rtol=5e-2, atol=0.0)
    assert np.isclose(np.std(Es), 0.02*3e9, rtol=0.005, atol=0.0)
    assert np.isclose(np.std(zs), 50.0e-6, rtol=0.005, atol=0.03)
    assert np.isclose(beam.beam_size_x(), np.sqrt(geo_emitt_x*beta_x), rtol=1e-3, atol=0.0)  # Beam size
    assert np.isclose(beam.beam_size_y(), np.sqrt(geo_emitt_y*beta_y), rtol=5e-3, atol=0.0)
    assert np.isclose(beam.divergence_x(), np.sqrt(geo_emitt_x/beta_x), rtol=5e-2, atol=0.0)
    assert np.isclose(beam.divergence_y(), np.sqrt(geo_emitt_y/beta_y), rtol=5e-2, atol=0.0)
    assert np.isclose(beam.energy_spread(), 0.02*3e9, rtol=0.01, atol=0.0)
    assert np.isclose(beam.bunch_length(), 50.0e-6, rtol=0.005, atol=0.03)

    assert np.isclose(beam.alpha_x(), alpha_x, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.alpha_y(), alpha_y, rtol=4e-2, atol=0.0)
    assert np.isclose(beam.beta_x(), beta_x, rtol=3e-3, atol=0.0)
    assert np.isclose(beam.beta_y(), beta_y, rtol=5e-3, atol=0.0)
    assert np.isclose(beam.geom_emittance_x(), geo_emitt_x, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.geom_emittance_y(), geo_emitt_y, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_x(), geo_emitt_x*beam.gamma(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), geo_emitt_y*beam.gamma(), rtol=1e-2, atol=0.0)


@pytest.mark.beam
def test_param_calcs_generate_symm_trace_space_xyz():
    "Test the beam parameter calculations using a trace space generated with generate_symm_trace_space_xyz()."
    
    np.random.seed(42)

    alpha_x = -0.239
    alpha_y = -0.171
    beta_x = 0.120                                                                    # [m]
    beta_y = 0.120                                                                    # [m]
    geo_emitt_x = 2.552865e-09                                                        # [m rad]
    geo_emitt_y = 1.750833e-11                                                        # [m rad]
    num_particles = 200011

    # Generate trace space
    xs, xps, ys, yps, zs, Es = generate_symm_trace_space_xyz(geo_emitt_x, beta_x, alpha_x, geo_emitt_y, beta_y, alpha_y, num_particles, bunch_length=50.0e-6, energy_spread=0.02*3e9, L=0)
    Es += 3e9  # Add offset.

    beam = Beam()
    beam.set_phase_space(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=-e*1.0e10)

    # Examine the beam parameters
    assert np.isclose(np.std(xs), np.sqrt(geo_emitt_x*beta_x), rtol=1e-2, atol=0.0)  # Beam size
    assert np.isclose(np.std(ys), np.sqrt(geo_emitt_y*beta_y), rtol=1e-2, atol=0.0)
    assert np.isclose(np.std(Es), 0.02*3e9, rtol=0.005, atol=0.0)
    assert np.isclose(np.std(zs), 50.0e-6, rtol=0.005, atol=0.03)
    assert np.isclose(beam.beam_size_x(), np.sqrt(geo_emitt_x*beta_x), rtol=1e-3, atol=0.0)  # Beam size
    assert np.isclose(beam.beam_size_y(), np.sqrt(geo_emitt_y*beta_y), rtol=5e-3, atol=0.0)
    assert np.isclose(beam.divergence_x(), np.sqrt(geo_emitt_x/beta_x), rtol=5e-2, atol=0.0)
    assert np.isclose(beam.divergence_y(), np.sqrt(geo_emitt_y/beta_y), rtol=5e-2, atol=0.0)
    assert np.isclose(beam.energy_spread(), 0.02*3e9, rtol=0.005, atol=0.0)
    assert np.isclose(beam.bunch_length(), 50.0e-6, rtol=0.005, atol=0.03)
    assert np.isclose(beam.peak_current(), -beam.charge()/(np.sqrt(2*np.pi)*beam.bunch_length())*SI.c, rtol=0.0, atol=3e2)
    assert np.isclose(beam.peak_density(), beam.charge()/(SI.e*np.sqrt(2*np.pi)**3*beam.beam_size_x()*beam.beam_size_y()*beam.bunch_length()), rtol=1e-8, atol=0.0)

    assert np.isclose(beam.alpha_x(), alpha_x, rtol=3e-2, atol=0.0)
    assert np.isclose(beam.alpha_y(), alpha_y, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.beta_x(), beta_x, rtol=3e-3, atol=0.0)
    assert np.isclose(beam.beta_y(), beta_y, rtol=3e-3, atol=0.0)
    assert np.isclose(beam.geom_emittance_x(), geo_emitt_x, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.geom_emittance_y(), geo_emitt_y, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_x(), geo_emitt_x*beam.gamma(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), geo_emitt_y*beam.gamma(), rtol=1e-2, atol=0.0)



############# Tests for beam projections #############
@pytest.mark.beam
def test_current_profile():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=1.0e-6, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()

    dIdt, ts = beam.current_profile()
    assert np.isclose(np.max(np.abs(dIdt)), beam.peak_current(), rtol=1e-15, atol=0.0)
    assert np.isclose(SI.c*ts.mean(), beam.z_offset(), rtol=1e-10, atol=0.0)

    Nbins = int(np.sqrt(len(beam)/2))
    bins = np.linspace(min(beam.ts()), max(beam.ts()), Nbins)
    dIdt, ts = beam.current_profile(bins=bins)
    assert np.isclose(np.max(np.abs(dIdt)), beam.peak_current(), rtol=1e-15, atol=0.0)
    assert len(dIdt) == Nbins - 1
    assert len(ts) == Nbins - 1
    assert np.isclose(SI.c*ts.mean(), beam.z_offset(), rtol=1e-10, atol=0.0)


@pytest.mark.beam
def test_longitudinal_num_density():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=1.0e-6, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    dNdz, zs = beam.longitudinal_num_density()
    assert np.isclose( np.sum( dNdz*np.diff(zs)[0] ), beam.total_particles(), rtol=1e-10, atol=0.0 )
    assert np.isclose(zs.mean(), beam.z_offset(), rtol=1e-10, atol=0.0)


@pytest.mark.beam
def test_energy_spectrum():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    dQdE, Es = beam.energy_spectrum()
    assert np.isclose( np.sum( dQdE*np.diff(Es)[0] ), beam.charge(), rtol=1e-10, atol=0.0 )
    assert np.isclose( Es.mean(), beam.energy(), rtol=1e-10, atol=0.0 )


@pytest.mark.beam
def test_rel_energy_spectrum():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    dQdE, rel_Es = beam.rel_energy_spectrum()
    assert np.isclose( np.sum( dQdE*np.diff(rel_Es)[0] ), beam.charge(), rtol=1e-10, atol=0.0 )
    assert np.isclose( rel_Es.mean(), 0.0, rtol=0.0, atol=1e-10 )


@pytest.mark.beam
def test_transverse_profile_x():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=1.0e-6, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    dQdx, xs = beam.transverse_profile_x()
    assert np.isclose( np.sum( dQdx*np.diff(xs)[0] ), beam.charge(), rtol=1e-10, atol=0.0 )
    assert np.isclose( xs.mean(), beam.x_offset(), rtol=1e-10, atol=0.0 )


@pytest.mark.beam
def test_transverse_profile_y():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=1.0e-6, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    dQdy, ys = beam.transverse_profile_y()
    assert np.isclose( np.sum( dQdy*np.diff(ys)[0] ), beam.charge(), rtol=1e-8, atol=0.0 )
    assert np.isclose( ys.mean(), beam.y_offset(), rtol=1e-10, atol=0.0 )


@pytest.mark.beam
def test_transverse_profile_xp():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=1.0e-6, y_angle=0.0)
    beam = source.track()
    dQdxp, xps = beam.transverse_profile_xp()
    assert np.isclose( np.sum( dQdxp*np.diff(xps)[0] ), beam.charge(), rtol=1e-10, atol=0.0 )
    assert np.isclose( xps.mean(), beam.x_angle(), rtol=1e-10, atol=0.0 )


@pytest.mark.beam
def test_transverse_profile_yp():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=1.0e-6)
    beam = source.track()
    dQdyp, yps = beam.transverse_profile_yp()
    assert np.isclose( np.sum( dQdyp*np.diff(yps)[0] ), beam.charge(), rtol=1e-10, atol=0.0 )
    assert np.isclose( yps.mean(), beam.y_angle(), rtol=1e-10, atol=0.0 )


@pytest.mark.beam
def test_phase_space_density():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=1.2e-6, y_angle=0.3e-6)
    beam = source.track()
    density, xps, yps = beam.phase_space_density(beam.xps, beam.yps, hbins=None, vbins=None)
    density_1d = np.sum(density*np.diff(xps)[0], axis=1)
    assert np.isclose( np.sum(density_1d*np.diff(yps)[0]), beam.charge(), rtol=1e-10, atol=0.0 )
    assert np.isclose( xps.mean(), beam.x_angle(), rtol=1e-10, atol=0.0 )
    assert np.isclose( yps.mean(), beam.y_angle(), rtol=1e-10, atol=0.0 )

    Nbins = int(np.sqrt(len(beam)/2))
    hbins = np.linspace(min(beam.xps()), max(beam.xps()), Nbins+1)
    vbins = np.linspace(min(beam.yps()), max(beam.yps()), Nbins+2)
    density, xps, yps = beam.phase_space_density(beam.xps, beam.yps, hbins, vbins)
    density_1d = np.sum(density*np.diff(xps)[0], axis=1)
    assert np.isclose( np.sum(density_1d*np.diff(yps)[0]), beam.charge(), rtol=1e-10, atol=0.0 )
    assert np.isclose( xps.mean(), beam.x_angle(), rtol=1e-10, atol=0.0 )
    assert np.isclose( yps.mean(), beam.y_angle(), rtol=1e-10, atol=0.0 )


@pytest.mark.beam
def test_density_lps():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=1.0e-6, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    density, zs, Es = beam.density_lps(hbins=None, vbins=None)
    density_1d = np.sum(density*np.diff(zs)[0], axis=1)
    assert np.isclose( np.sum(density_1d*np.diff(Es)[0]), beam.charge(), rtol=1e-10, atol=0.0 )
    assert np.isclose( Es.mean(), beam.energy(), rtol=1e-10, atol=0.0 )
    assert np.isclose( zs.mean(), beam.z_offset(), rtol=1e-10, atol=0.0 )

    Nbins = int(np.sqrt(len(beam)/2))
    hbins = np.linspace(min(beam.zs()), max(beam.zs()), Nbins+1)
    vbins = np.linspace(min(beam.Es()), max(beam.Es()), Nbins+2)
    density, zs, Es = beam.density_lps(hbins, vbins)
    density_1d = np.sum(density*np.diff(zs)[0], axis=1)
    assert np.isclose( np.sum(density_1d*np.diff(Es)[0]), beam.charge(), rtol=1e-10, atol=0.0 )
    assert np.isclose( Es.mean(), beam.energy(), rtol=1e-10, atol=0.0 )
    assert np.isclose( zs.mean(), beam.z_offset(), rtol=1e-10, atol=0.0 )


@pytest.mark.beam
def test_density_transverse():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.2e-6, y_offset=3.3e-6, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    density, xs, ys = beam.density_transverse()
    density_1d = np.sum(density*np.diff(xs)[0], axis=1)
    assert np.isclose( np.sum(density_1d*np.diff(ys)[0]), beam.charge(), rtol=1e-10, atol=0.0 )
    assert np.isclose( xs.mean(), beam.x_offset(), rtol=1e-10, atol=0.0 )
    assert np.isclose( ys.mean(), beam.y_offset(), rtol=1e-10, atol=0.0 )



############# Tests of coordinate rotation methods #############
def active_rotate_arrs(x_comps, y_comps, z_comps, x_angle, y_angle):
    "Active rotation of ndarrays ``x_comps``, ``y_comps`` and ``z_comps`` containing the xyz-components of vectors such as position and proper velocity first with ``x_angle`` around the y-axis then with ``y_angle`` around the x-axis."
    rotated_z_comps = z_comps * np.cos(x_angle)*np.cos(y_angle) - x_comps * np.sin(x_angle)*np.cos(y_angle) + y_comps * np.sin(y_angle)
    rotated_x_comps = z_comps * np.sin(x_angle) + x_comps * np.cos(x_angle)
    rotated_y_comps = -z_comps * np.cos(x_angle)*np.sin(y_angle) + x_comps * np.sin(x_angle)*np.sin(y_angle) + y_comps * np.cos(y_angle)

    return rotated_x_comps, rotated_y_comps, rotated_z_comps


def passive_rotate_arrs(x_comps, y_comps, z_comps, x_angle, y_angle):
    "Passive rotation of ndarrays ``x_comps``, ``y_comps`` and ``z_comps`` containing the xyz-components of vectors such as position and proper velocity first with ``x_angle`` around the y-axis then with ``y_angle`` around the x-axis."
    rotated_z_comps = z_comps * np.cos(x_angle)*np.cos(y_angle) + x_comps * np.sin(x_angle)*np.cos(y_angle) - y_comps * np.sin(y_angle)
    rotated_x_comps = -z_comps * np.sin(x_angle) + x_comps * np.cos(x_angle)
    rotated_y_comps = z_comps * np.cos(x_angle)*np.sin(y_angle) + x_comps * np.sin(x_angle)*np.sin(y_angle) + y_comps * np.cos(y_angle)

    return rotated_x_comps, rotated_y_comps, rotated_z_comps


# def test_slice_centroids():
#     source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0)
#     beam = source.track()
#     beam_quant = beam.xs()  # Use x positions as the beam quantity
#     x_slices, z_centroids = beam.slice_centroids(beam_quant, bin_number=5)

#     # Check that the output shapes are correct
#     assert x_slices.shape[0] == z_centroids.shape[0]
#     assert len(x_slices) == 5  # Should match the number of bins

#     # Check that the centroids are within the expected range
#     assert np.all(z_centroids >= 0)
#     assert np.all(z_centroids <= 10)


@pytest.mark.beam
def test_x_tilt_angle():
    "Test the retrieval of the beam tilt angle in the zx-plane."
    source = setup_trapezoid_source(current_head=0.1e3, bunch_length=1050e-6, z_offset=1615e-6, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0, enable_xy_jitter=False, enable_xpyp_jitter=False)
    beam = source.track()
    nom_x_angle = -2.82e-8  # [rad]

    # Add tilt in x
    rotated_xs, rotated_ys, rotated_zs = passive_rotate_arrs(beam.xs(), beam.ys(), beam.zs(), nom_x_angle, y_angle=0.0)
    rotated_uxs, rotated_uys, rotated_uzs = passive_rotate_arrs(beam.uxs(), beam.uys(), beam.uzs(), nom_x_angle, y_angle=0.0)
    beam.set_phase_space(beam.charge(), rotated_xs, rotated_ys, rotated_zs, rotated_uxs, rotated_uys, rotated_uzs)

    # Compare the angles
    x_angle = beam.x_tilt_angle()
    y_angle = beam.y_tilt_angle()
    assert np.isclose(x_angle, -nom_x_angle, rtol=1e-3, atol=1e-8)
    assert np.isclose(y_angle, 0.0, rtol=1e-7, atol=1e-8)


@pytest.mark.beam
def test_y_tilt_angle():
    "Test the retrieval of the beam tilt angle in the zy-plane."
    source = setup_trapezoid_source(current_head=0.1e3, bunch_length=1050e-6, z_offset=1615e-6, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0, enable_xy_jitter=False, enable_xpyp_jitter=False)
    beam = source.track()
    nom_y_angle = 7.04e-9  # [rad]

    # Add tilt in y
    rotated_xs, rotated_ys, rotated_zs = passive_rotate_arrs(beam.xs(), beam.ys(), beam.zs(), 0.0, y_angle=nom_y_angle)
    rotated_uxs, rotated_uys, rotated_uzs = passive_rotate_arrs(beam.uxs(), beam.uys(), beam.uzs(), 0.0, y_angle=nom_y_angle)
    beam.set_phase_space(beam.charge(), rotated_xs, rotated_ys, rotated_zs, rotated_uxs, rotated_uys, rotated_uzs)

    # Compare the angles
    x_angle = beam.x_tilt_angle()
    y_angle = beam.y_tilt_angle()
    assert np.isclose(y_angle, nom_y_angle, rtol=1e-3, atol=1e-11)
    assert np.isclose(x_angle, 0.0, rtol=1e-7, atol=1e-8)


@pytest.mark.beam
def test_beam_alignment_angles():
    "Test ``Beam.beam_alignment_angles()``, which calculates the angles for rotation around the y- and x-axis to align the z-axis to the beam proper velocity."
    
    start_x_angle = 1.2e-7  # [rad], define an angle in the zx-plane.
    start_y_angle = -2.3e-7  # [rad], define an angle in the zy-plane.

    # Create a source with angular offsets, but not tilted in space
    source = setup_trapezoid_source(current_head=0.1e3, bunch_length=1050e-6, z_offset=1615e-6, x_offset=0.0, y_offset=0.0, x_angle=start_x_angle, y_angle=start_y_angle, enable_xy_jitter=False, enable_xpyp_jitter=False)

    # Generate a beam
    beam = source.track()

    # Get the angles
    align_x_angle, align_y_angle = beam.beam_alignment_angles()

    # Compare to the chosen angular offsets
    assert np.isclose(align_x_angle, start_x_angle, rtol=1e-5, atol=1e-13)
    assert np.isclose(align_y_angle, start_y_angle, rtol=1e-5, atol=1e-13)
    

@pytest.mark.beam
def test_add_pointing_tilts():
    "Test ``Beam.add_pointing_tilts()``, which uses active transformation to tilt the beam in the zx- and zy-planes."

    start_x_angle = -2.82328e-08 # [rad], define an angle in the zx-plane.
    start_y_angle = 7.040999e-09   # [rad], define an angle in the zy-plane.

    # Create a source with angular offsets, but not tilted in space
    source = setup_trapezoid_source(current_head=0.1e3, bunch_length=1050e-6, z_offset=1615e-6, x_offset=0.0, y_offset=0.0, x_angle=start_x_angle, y_angle=start_y_angle, enable_xy_jitter=False, enable_xpyp_jitter=False)

    # Generate a beam
    beam = source.track()
    initial_beam = copy.deepcopy(beam)

    # Tilt the beam using its angular offset
    beam.add_pointing_tilts()

    # Rotate the positions of all particles with written out formulas for comparison. Note the sign convention for y_angle.
    rotated_xs, rotated_ys, rotated_zs = active_rotate_arrs(initial_beam.xs(), initial_beam.ys(), initial_beam.zs(), start_x_angle, -start_y_angle)

    # Examine the resulting beam
    assert np.isclose(beam.x_angle(), start_x_angle, rtol=1e-3, atol=1e-11)
    assert np.isclose(beam.y_angle(), start_y_angle, rtol=1e-3, atol=1e-11)
    assert np.allclose(beam.uxs(), initial_beam.uxs(), rtol=1e-15, atol=0.0)  # The proper velocities should not change.
    assert np.allclose(beam.uys(), initial_beam.uys(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uzs(), initial_beam.uzs(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.xs(), rotated_xs, rtol=1e-8, atol=0.0)
    assert np.allclose(beam.ys(), rotated_ys, rtol=1e-8, atol=0.0)
    assert np.allclose(beam.zs(), rotated_zs, rtol=1e-8, atol=0.0)


@pytest.mark.beam
def test_xy_rotate_coord_sys1():
    "Test correct rotation of beam coordinate system around the y-axis."

    np.random.seed(42)

    start_x_angle = 1e-7  # [rad], define an angle in the zx-plane.

    # Source with angular offset, but no tilt
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=start_x_angle, y_angle=0.0)
    
    nom_x_angles = np.concatenate((-np.logspace(-9, np.log10(0.11), 10), np.logspace(-9, -1, 10)))  # [rad]

    for i in range(len(nom_x_angles)):
        
        beam = source.track()
        beam.add_pointing_tilts()  # Add an initial tilt in the zx-plane.
        old_beam = copy.deepcopy(beam)

        nom_x_angle = nom_x_angles[i]
        beam.xy_rotate_coord_sys(x_angle=nom_x_angle, y_angle=None, invert=False)

        # Rotate the positions and proper velocities of all particles with written out formulas for comparison
        rotated_xs, rotated_ys, rotated_zs = passive_rotate_arrs(old_beam.xs(), old_beam.ys(), old_beam.zs(), nom_x_angle, y_angle=0.0)
        rotated_uxs, rotated_uys, rotated_uzs = passive_rotate_arrs(old_beam.uxs(), old_beam.uys(), old_beam.uzs(), nom_x_angle, y_angle=0.0)
        rotated_ux_offset, rotated_uy_offset, rotated_uz_offset = passive_rotate_arrs(old_beam.ux_offset(), old_beam.uy_offset(), old_beam.uz_offset(), nom_x_angle, y_angle=0.0)

        # Compare the rotated arrays
        assert np.allclose(beam.zs(), rotated_zs, rtol=1e-7, atol=1e-13)
        assert np.allclose(beam.xs(), rotated_xs, rtol=1e-7, atol=1e-13)
        assert np.allclose(beam.ys(), rotated_ys, rtol=1e-7, atol=1e-13)
        assert np.allclose(beam.uzs(), rotated_uzs, rtol=1e-7, atol=1e-13)
        assert np.allclose(beam.uxs(), rotated_uxs, rtol=1e-7, atol=1e-13)
        assert np.allclose(beam.uys(), rotated_uys, rtol=1e-7, atol=1e-13)
        assert np.isclose(beam.x_angle(), np.arcsin(rotated_ux_offset/rotated_uz_offset), rtol=np.abs(beam.x_angle())*1e-3, atol=np.abs(beam.x_angle())*2e-3)
        assert np.isclose(beam.y_angle(), np.arcsin(rotated_uy_offset/rotated_uz_offset), rtol=1e-7, atol=1e-13)


@pytest.mark.beam
def test_xy_rotate_coord_sys2():
    "Test correct rotation of beam coordinate system around the y-axis."

    np.random.seed(42)

    start_y_angle = 1e-6  # [rad], define an angle in the zy-plane.
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=start_y_angle)
    
    nom_y_angles = np.concatenate((-np.logspace(-9, np.log10(0.11), 10), np.logspace(-9, -1, 10)))  # [rad]

    for i in range(len(nom_y_angles)):
        
        beam = source.track()
        beam.add_pointing_tilts()  # Add an initial tilt in the zy-plane.
        old_beam = copy.deepcopy(beam)

        nom_y_angle = nom_y_angles[i]
        beam.xy_rotate_coord_sys(x_angle=None, y_angle=nom_y_angle, invert=False)

        # Roatate the positions and proper velocities of all particles with written out formulas for comparison
        rotated_xs, rotated_ys, rotated_zs = passive_rotate_arrs(old_beam.xs(), old_beam.ys(), old_beam.zs(), 0.0, y_angle=nom_y_angle)
        rotated_uxs, rotated_uys, rotated_uzs = passive_rotate_arrs(old_beam.uxs(), old_beam.uys(), old_beam.uzs(), 0.0, y_angle=nom_y_angle)
        rotated_ux_offset, rotated_uy_offset, rotated_uz_offset = passive_rotate_arrs(old_beam.ux_offset(), old_beam.uy_offset(), old_beam.uz_offset(), 0.0, y_angle=nom_y_angle)

        # Compare the rotated arrays
        assert np.allclose(beam.zs(), rotated_zs, rtol=1e-7, atol=1e-13)
        assert np.allclose(beam.xs(), rotated_xs, rtol=1e-7, atol=1e-13)
        assert np.allclose(beam.ys(), rotated_ys, rtol=1e-7, atol=1e-13)
        assert np.allclose(beam.uzs(), rotated_uzs, rtol=1e-7, atol=1e-13)
        assert np.allclose(beam.uxs(), rotated_uxs, rtol=1e-7, atol=1e-13)
        assert np.allclose(beam.uys(), rotated_uys, rtol=1e-7, atol=1e-13)
        assert np.isclose(beam.x_angle(), np.arcsin(rotated_ux_offset/rotated_uz_offset), rtol=1e-7, atol=1e-13)
        assert np.isclose(beam.y_angle(), np.arcsin(rotated_uy_offset/rotated_uz_offset), rtol=np.abs(nom_y_angle)*1e-3, atol=np.abs(nom_y_angle)*2e-3)


@pytest.mark.beam
def test_xy_rotate_coord_sys3():
    "Test adding creating a tilted beam and then align the z-axis to the drive beam propagation axis using passive transformation to rotate the frame of the beam."

    np.random.seed(42)

    start_x_angle = -2.82328e-08 # [rad], define an angle in the zx-plane.
    start_y_angle = 7.040999e-09   # [rad], define an angle in the zy-plane.

    # Create a source with angular offsets, but not tilted in space
    source = setup_trapezoid_source(current_head=0.1e3, bunch_length=1050e-6, z_offset=1615e-6, x_offset=0.0, y_offset=0.0, x_angle=start_x_angle, y_angle=start_y_angle, enable_xy_jitter=False, enable_xpyp_jitter=False)

    # Generate a beam
    beam = source.track()
    initial_beam = copy.deepcopy(beam)  # Has angular offsets, but not tilted in space

    # Tilt the beam using its angular offset
    beam.add_pointing_tilts()

    # Rotate the positions of all particles with written out formulas for comparison. Note the sign convention for y_angle.
    rotated_xs, rotated_ys, rotated_zs = active_rotate_arrs(initial_beam.xs(), initial_beam.ys(), initial_beam.zs(), start_x_angle, -start_y_angle)

    # Examine the tilted beam
    assert np.isclose(beam.x_angle(), start_x_angle, rtol=1e-3, atol=1e-11)
    assert np.isclose(beam.y_angle(), start_y_angle, rtol=1e-3, atol=1e-11)
    assert np.allclose(beam.uxs(), initial_beam.uxs(), rtol=1e-15, atol=0.0)  # The proper velocities should not change.
    assert np.allclose(beam.uys(), initial_beam.uys(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uzs(), initial_beam.uzs(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.xs(), rotated_xs, rtol=1e-8, atol=0.0)
    assert np.allclose(beam.ys(), rotated_ys, rtol=1e-8, atol=0.0)
    assert np.allclose(beam.zs(), rotated_zs, rtol=1e-8, atol=0.0)

    # Calculate the angles that will be used to rotate the beams' frame
    rotation_angle_x, rotation_angle_y = beam.beam_alignment_angles()
    rotation_angle_y = -rotation_angle_y  # Minus due to right hand rule.

    assert np.isclose(rotation_angle_x, start_x_angle, rtol=1e-10, atol=0.0)
    assert np.isclose(rotation_angle_y, -start_y_angle, rtol=1e-10, atol=0.0)


    # Use passive transformation to rotate the frame of the beam
    beam.xy_rotate_coord_sys(rotation_angle_x, rotation_angle_y)  # Align the z-axis to the drive beam propagation.

    # Rotate the proper velocities of all particles with written out formulas for comparison. Note the sign convention for y_angle.
    rotated_uxs, rotated_uys, rotated_uzs = passive_rotate_arrs(initial_beam.uxs(), initial_beam.uys(), initial_beam.uzs(), rotation_angle_x, rotation_angle_y)

    # Examine the aligned beam
    assert beam.x_angle() < 1e-15  # The angular offsets should be very small
    assert beam.y_angle() < 1e-15
    assert np.allclose(beam.uxs(), rotated_uxs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uys(), rotated_uys, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uzs(), rotated_uzs, rtol=1e-15, atol=0.0)
    # The position arrays should have been rotated back to when the beam only had angular offsets, but not tilted in space
    assert np.allclose(beam.xs(), initial_beam.xs(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.ys(), initial_beam.ys(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.zs(), initial_beam.zs(), rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_rotate_coord_sys_3D():
    "Test correct passive rotation of beam coordinate system."

    np.random.seed(42)

    start_x_angle = -2.82328e-08 # [rad], define an angle in the zx-plane.
    start_y_angle = 7.040999e-09   # [rad], define an angle in the zy-plane.
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=start_x_angle, y_angle=start_y_angle)

    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0)
    old_beam = source.track()
    old_beam.add_pointing_tilts()  # Add an initial tilt in the zy-plane.

    x_axis = np.array([0, 1, 0])  # Axis as an unit vector. Axis permutaton is zxy.
    y_axis = np.array([0, 0, 1])
    nom_x_angles = np.concatenate((-np.logspace(np.log10(0.11), -9, 10), np.logspace(-8.9, -1, 10)))  # [rad]
    nom_y_angles = np.concatenate((-np.logspace(np.log10(0.09), -8.1, 10), np.logspace(-9, -0.97, 10)))  # [rad]

    for i in range(len(nom_x_angles)):
        for j in range(len(nom_y_angles)):

            beam = copy.deepcopy(old_beam)

            nom_x_angle = nom_x_angles[i]
            nom_y_angle = nom_y_angles[j]
            beam.rotate_coord_sys_3D(y_axis, nom_x_angle, x_axis, nom_y_angle, invert=False)

            # Roatate the positions and proper velocities of all particles with written out formulas for comparison
            rotated_xs, rotated_ys, rotated_zs = passive_rotate_arrs(old_beam.xs(), old_beam.ys(), old_beam.zs(), nom_x_angle, nom_y_angle)
            rotated_uxs, rotated_uys, rotated_uzs = passive_rotate_arrs(old_beam.uxs(), old_beam.uys(), old_beam.uzs(), nom_x_angle, nom_y_angle)
            rotated_ux_offset, rotated_uy_offset, rotated_uz_offset = passive_rotate_arrs(old_beam.ux_offset(), old_beam.uy_offset(), old_beam.uz_offset(), nom_x_angle, nom_y_angle)

            # Compare the rotated arrays
            assert np.allclose(beam.zs(), rotated_zs, rtol=1e-7, atol=1e-13)
            assert np.allclose(beam.xs(), rotated_xs, rtol=1e-7, atol=1e-13)
            assert np.allclose(beam.ys(), rotated_ys, rtol=1e-7, atol=1e-13)
            assert np.allclose(beam.uzs(), rotated_uzs, rtol=1e-7, atol=1e-13)
            assert np.allclose(beam.uxs(), rotated_uxs, rtol=1e-7, atol=1e-13)
            assert np.allclose(beam.uys(), rotated_uys, rtol=1e-7, atol=1e-13)
            assert np.isclose(beam.x_angle(), np.arcsin(rotated_ux_offset/rotated_uz_offset), rtol=np.abs(nom_x_angle)*1e-3, atol=np.abs(nom_x_angle)*2e-3)
            assert np.isclose(beam.y_angle(), np.arcsin(rotated_uy_offset/rotated_uz_offset), rtol=np.abs(nom_y_angle)*1e-3, atol=np.abs(nom_y_angle)*2e-3)

            # Un-rotate
            beam.rotate_coord_sys_3D(y_axis, nom_x_angle, x_axis, nom_y_angle, invert=True)

            # Compare the rotated arrays
            assert np.allclose(beam.zs(), old_beam.zs(), rtol=1e-7, atol=1e-13)
            assert np.allclose(beam.xs(), old_beam.xs(), rtol=1e-7, atol=1e-13)
            assert np.allclose(beam.ys(), old_beam.ys(), rtol=1e-7, atol=1e-13)
            assert np.allclose(beam.uzs(), old_beam.uzs(), rtol=1e-7, atol=1e-13)
            assert np.allclose(beam.uxs(), old_beam.uxs(), rtol=1e-7, atol=1e-13)
            assert np.allclose(beam.uys(), old_beam.uys(), rtol=1e-7, atol=1e-13)
            assert np.isclose(beam.x_angle(), old_beam.x_angle(), rtol=1e-7, atol=1e-13)
            assert np.isclose(beam.y_angle(), old_beam.y_angle(), rtol=1e-7, atol=1e-13)



############# Tests of in-house beam field calculation methods #############
@pytest.mark.beam
def test_charge_density_3D():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=123.4e-6, x_offset=0.1e-6, y_offset=4.5e-6, x_angle=1.2e-6, y_angle=0.3e-6)
    beam = source.track()
    dQ_dzdxdy, zctrs, xctrs, yctrs, edges_z, edges_x, edges_y = beam.charge_density_3D(zbins=None, xbins=None, ybins=None)

    # Calculate volume of each bin
    dz = np.diff(edges_z)
    dx = np.diff(edges_x)
    dy = np.diff(edges_y)
    bin_volumes = dz[:, None, None] * dx[None, :, None] * dy[None, None, :]  # The None indexing is used to add new axes to the differences arrays, allowing them to be broadcasted properly for division with counts. This ensures that each element of counts is divided by the corresponding bin volume (element-wise division).

    assert np.isclose(np.sum(dQ_dzdxdy*bin_volumes), beam.charge(), rtol=1e-15, atol=0.0)
    assert np.isclose( xctrs.mean(), beam.x_offset(), rtol=1e-10, atol=0.0 )
    assert np.isclose( yctrs.mean(), beam.y_offset(), rtol=1e-10, atol=0.0 )
    assert np.isclose( zctrs.mean(), beam.z_offset(), rtol=1e-10, atol=0.0 )

# TODO: test Dirichlet_BC_system_matrix(), Ex_Ey_2D(), Ex_Ey()


############# Tests of plotting methods #############