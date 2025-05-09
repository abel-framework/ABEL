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
from abel.utilities.beam_physics import generate_trace_space, generate_trace_space_xy, generate_symm_trace_space_xyz
from abel.utilities.plasma_physics import beta_matched
from abel.utilities.relativity import energy2proper_velocity, energy2gamma, gamma2momentum, proper_velocity2gamma, proper_velocity2energy
from abel.utilities.statistics import weighted_mean, weighted_std, weighted_cov
import random
import scipy.constants as SI
import numpy as np
import copy, shutil
from matplotlib import pyplot as plt
import warnings


def setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, bunch_length=40.0e-06, energy=3e9, rel_energy_spread=0.02, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0):
    source = SourceBasic()
    source.bunch_length = bunch_length                                              # [m], rms.
    source.num_particles = 10000                                               
    source.charge = -SI.e * 1.0e10                                                     # [C]

    # Energy parameters
    source.energy = energy                                                          # [eV]
    source.rel_energy_spread = rel_energy_spread                                    # Relative rms energy spread

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

    # Purposedly trigger exceptions
    with pytest.raises(ValueError):
        beam = Beam(num_particles=-1)  # Should raise an error if attempted to initiate with negative number of particles.
    with pytest.raises(ValueError):
        beam = Beam(num_particles=1000.2)
    with pytest.raises(ValueError):
        beam = Beam(num_bunches_in_train=-1)
    with pytest.raises(ValueError):
        beam = Beam(num_bunches_in_train=1300.2)
    with pytest.raises(ValueError):
        beam = Beam(bunch_separation=-1)


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

    energy_thres = 13*SI.m_e*SI.c**2/SI.e  # [eV], 13 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(10e12, unit='eV', m=SI.m_e), size=num_particles)

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
def test_set_phase_space5():
    "Verify that the phase space is set correctly."
    
    num_particles = 10042
    Q = -SI.e * 1.0e10
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)

    energy_thres = 13*SI.m_e*SI.c**2/SI.e  # [eV], 13 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(10e12, unit='eV', m=SI.m_e), size=num_particles)
    pz_thres = gamma2momentum(energy2gamma(energy_thres, unit='eV', m=SI.m_e))
    pzs = np.random.normal(pz_thres, 0.02*pz_thres, num_particles)
    Es = np.random.normal(energy_thres, 0.02*energy_thres, num_particles)


    ## Purposedly trigger exceptions
    # Coordinates type and length
    with pytest.raises(TypeError):
        beam = Beam()
        beam.set_phase_space(Q, 3.14, ys, 1.0)
    with pytest.raises(TypeError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, 1.0)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, np.random.rand(num_particles+1), ys, zs)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, np.random.rand(num_particles-1), zs)

    # Momenta type and length
    with pytest.raises(TypeError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uxs=2.9)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, np.random.rand(num_particles+1), uys, uzs)
    with pytest.raises(TypeError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uxs, 15.16, uzs)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uxs, np.random.rand(num_particles-1), uzs)
    with pytest.raises(TypeError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uz_thres)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uxs, uys, np.append(uzs, uz_thres))

    with pytest.raises(TypeError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, pxs=2.9)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, pxs=np.random.rand(num_particles+1), pys=uys, pzs=pzs)
    with pytest.raises(TypeError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, pxs=uxs, pys=15.16, pzs=pzs)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, pxs=uxs, pys=np.random.rand(num_particles-1), pzs=pzs)
    with pytest.raises(TypeError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, pxs=uxs, pys=uys, pzs=1.6e-18)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, pxs=uxs, pys=uys, pzs=np.append(pzs, 1.6e-18))

    with pytest.raises(TypeError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, xps=2.9)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, xps=np.random.rand(num_particles+1))
    with pytest.raises(TypeError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, xps=uxs, yps=15.16)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, xps=uxs, yps=np.random.rand(num_particles-1), Es=Es)
    with pytest.raises(TypeError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, xps=uxs, yps=uys, Es=100e9)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, xps=uxs, yps=uys, Es=np.append(Es, 10e9))

    # Define proper velocity, momentum or angle in the same direction
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uxs=uxs, pxs=uxs)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uxs=uxs, xps=uxs)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, pxs=uxs, xps=uxs)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uys=uys, pys=uys)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uys=uys, yps=uys)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, pys=uys, yps=uys)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uzs=uys, pzs=pzs)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uzs=uzs, Es=Es)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, pzs=pzs, Es=Es)
    
    # Set energies below the accepted threshold
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs=np.random.rand(num_particles))
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, pzs=pzs*0.01)
    with pytest.raises(ValueError):
        beam = Beam()
        beam.set_phase_space(Q, xs, ys, zs, Es=np.random.rand(num_particles))
    


@pytest.mark.beam
def test_reset_phase_space():
    "Test reset_phase_space to ensure it initializes an 8xN zero matrix for the specified number of particles."
    beam = Beam()
    beam.reset_phase_space(10)
    assert beam._Beam__phasespace.shape == (8, 10)
    assert (beam._Beam__phasespace == 0).all()

    # Purposedly trigger exceptions
    with pytest.raises(ValueError):
        beam.reset_phase_space(-11)  # Should raise an error if attempted to initiate with negative number of particles.
    with pytest.raises(ValueError):
        beam.reset_phase_space(10.001)


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


@pytest.mark.beam
def test_str():
    "Test the __str__ method for correct formatting of beam properties."

    beam = Beam()
    num_particles = 5
    beam.reset_phase_space(num_particles)
    beam_str = str(beam)
    assert "Beam:" in beam_str
    assert "0.00 nC" in beam_str
    assert str(num_particles) + " macroparticles" in beam_str

    num_particles = 5005
    Es = np.random.normal(151.567e9, 0.02*151.567e9, num_particles)
    qs = np.full(num_particles, -1.5e10*SI.e/num_particles)
    beam = Beam()
    beam.reset_phase_space(num_particles)
    beam.particle_mass = SI.m_e
    beam.set_qs(qs)
    beam.set_Es(Es)
    beam_str = str(beam)
    assert "Beam:" in beam_str
    assert "{:.2f}".format(-1.5e10*SI.e*1e9) in beam_str
    assert str(num_particles) + " macroparticles" in beam_str
    assert "{:.2f}".format(beam.energy()/1e9) + " GeV" in beam_str


@pytest.mark.beam
def test_copy_particle_charge():

    num_particles = 8051
    energy_thres = 11*SI.m_e*SI.c**2/SI.e  # [eV], 11 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam1 = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(10e12, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.0e10
    beam1.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    beam2 = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(10e12, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 3.4e10
    ws = np.random.rand(num_particles)*Q/(-SI.e)
    beam2.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs, weightings=ws)
    median_charge = np.median(beam2._Beam__phasespace[6, :])  # Extract expected median charge

    beam1.copy_particle_charge(beam2)
    assert (beam1._Beam__phasespace[6, :] == median_charge).all()  # Verify charge is copied correctly

    # Purposedly trigger exceptions
    with pytest.raises(ValueError):
        beam1 = Beam()
        beam2 = Beam()
        beam1.copy_particle_charge(beam2)
    with pytest.raises(ValueError):
        beam1 = Beam()
        beam2 = Beam()
        beam2.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs, weightings=ws)
        beam1.copy_particle_charge(beam2)
    with pytest.raises(ValueError):
        beam1 = Beam()
        beam2 = Beam()
        beam1.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs, weightings=ws)
        beam1.copy_particle_charge(beam2)


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


@pytest.mark.beam
def test_remove_halo_particles():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.1516e-6, y_offset=-5.354e-6, x_angle=-1.3e-6, y_angle=0.7e-6)
    beam = source.track()
    initial_beam = copy.deepcopy(beam)

    nsigma = 3
    xfilter = np.abs(beam.xs()-beam.x_offset(clean=True)) > nsigma*beam.beam_size_x(clean=True)
    xpfilter = np.abs(beam.xps()-beam.x_angle(clean=True)) > nsigma*beam.divergence_x(clean=True)
    yfilter = np.abs(beam.ys()-beam.y_offset(clean=True)) > nsigma*beam.beam_size_y(clean=True)
    ypfilter = np.abs(beam.yps()-beam.y_angle(clean=True)) > nsigma*beam.divergence_y(clean=True)
    filter = np.logical_or(np.logical_or(xfilter, xpfilter), np.logical_or(yfilter, ypfilter))
    count = np.count_nonzero(filter)  # Number of particles to be filtered away.
    num_particles_left = len(beam) - count

    beam.remove_halo_particles(nsigma=3)

    assert len(beam) == num_particles_left
    assert np.all( np.abs(beam.xs()-beam.x_offset(clean=True)) < nsigma*initial_beam.beam_size_x(clean=True) )
    assert np.all( np.abs(beam.xps()-beam.x_angle(clean=True)) < nsigma*initial_beam.divergence_x(clean=True) )
    assert np.all( np.abs(beam.ys()-beam.y_offset(clean=True)) < nsigma*initial_beam.beam_size_y(clean=True) )
    assert np.all( np.abs(beam.yps()-beam.y_angle(clean=True)) < nsigma*initial_beam.divergence_y(clean=True) )


@pytest.mark.beam
def test_rs():
    beam = Beam()
    num_particles = 4352
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs)

    assert np.allclose(beam.rs(), np.sqrt(xs**2 + ys**2), rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_deltas():
    beam = Beam()
    num_particles = 4352
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    Es = np.random.normal(1e9, 0.02*1e9, num_particles)
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs, Es=Es)

    assert np.allclose(beam.deltas(), beam.pzs()/np.mean(beam.pzs())-1, rtol=1e-15, atol=0.0)
    pz0 = np.mean(beam.pzs())*1.1
    assert np.allclose(beam.deltas(pz0=pz0), beam.pzs()/pz0-1, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_transverse_vector():
    num_particles = 8989
    energy_thres = 11*SI.m_e*SI.c**2/SI.e  # [eV], 11 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(10e12, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    vector = np.zeros((4,len(beam))) 
    vector[0,:] = beam.xs()
    vector[1,:] = beam.xps()
    vector[2,:] = beam.ys()
    vector[3,:] = beam.yps()

    assert np.allclose(beam.transverse_vector(), vector, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_norm_transverse_vector():
    num_particles = 8989
    energy_thres = 11*SI.m_e*SI.c**2/SI.e  # [eV], 11 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(10e12, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    vector = np.zeros((4,len(beam))) 
    vector[0,:] = beam.xs()
    vector[1,:] = beam.uxs()/SI.c 
    vector[2,:] = beam.ys()
    vector[3,:] = beam.uys()/SI.c 

    assert np.allclose(beam.norm_transverse_vector(), vector, rtol=1e-15, atol=0.0)




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
    beam.set_phase_space(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=-SI.e*1.0e10)

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
    num_particles = 200012

    # Generate trace space
    xs, xps, ys, yps = generate_trace_space_xy(geo_emitt_x, beta_x, alpha_x, geo_emitt_y, beta_y, alpha_y, num_particles, L=0, symmetrize=False)
    zs = np.random.normal(loc=0.0, scale=40.0e-6, size=num_particles)
    Es = np.random.normal(loc=3e9, scale=0.02*3e9, size=num_particles)

    beam = Beam()
    beam.set_phase_space(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=-SI.e*1.0e10)

    # Examine the beam parameters
    assert len(beam) == num_particles
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
    assert np.isclose(beam.alpha_y(), alpha_y, rtol=2e-2, atol=0.0)
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
    beam.set_phase_space(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=-SI.e*1.0e10)

    # Examine the beam parameters
    assert len(beam) == num_particles
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
    num_particles = 200016

    # Generate trace space
    xs, xps, ys, yps, zs, Es = generate_symm_trace_space_xyz(geo_emitt_x, beta_x, alpha_x, geo_emitt_y, beta_y, alpha_y, num_particles, bunch_length=50.0e-6, energy_spread=0.02*3e9, L=0)
    Es += 3e9  # Add offset.

    beam = Beam()
    beam.set_phase_space(xs=xs, ys=ys, zs=zs, xps=xps, yps=yps, Es=Es, Q=-SI.e*1.0e10)

    # Examine the beam parameters
    assert len(beam) == num_particles
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

    assert np.isclose(beam.alpha_x(), alpha_x, rtol=5e-2, atol=0.0)
    assert np.isclose(beam.alpha_y(), alpha_y, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.beta_x(), beta_x, rtol=3e-3, atol=0.0)
    assert np.isclose(beam.beta_y(), beta_y, rtol=3e-3, atol=0.0)
    assert np.isclose(beam.geom_emittance_x(), geo_emitt_x, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.geom_emittance_y(), geo_emitt_y, rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_x(), geo_emitt_x*beam.gamma(), rtol=1e-2, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), geo_emitt_y*beam.gamma(), rtol=1e-2, atol=0.0)


@pytest.mark.beam
def test_total_particles():
    
    num_particles = 4344
    energy_thres = 1200*SI.m_e*SI.c**2/SI.e  # [eV], 1200 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.8e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    total_particles = int(np.nansum(np.full(num_particles, Q/num_particles/-SI.e)))
    assert beam.total_particles() == total_particles

    # TODO: set some nans in beam


@pytest.mark.beam
def test_charge():
    
    num_particles = 4344
    energy_thres = 1200*SI.m_e*SI.c**2/SI.e  # [eV], 1200 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.8e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    charge = np.nansum(np.full(num_particles, Q/num_particles))
    assert np.isclose(beam.charge(), charge, rtol=1e-15, atol=0.0)

    # TODO: set some nans in beam


@pytest.mark.beam
def test_energy():
    
    num_particles = 4344
    energy_thres = 1200*SI.m_e*SI.c**2/SI.e  # [eV], 1200 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.8e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    Es = proper_velocity2energy(uzs, unit='eV', m=SI.m_e)
    energy = weighted_mean(Es, beam.weightings(), clean=False) 
    assert np.isclose(beam.energy(clean=False), energy, rtol=1e-15, atol=0.0)
    energy = weighted_mean(Es, beam.weightings(), clean=True) 
    assert np.isclose(beam.energy(clean=True), energy, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_gamma():
    
    num_particles = 4344
    energy_thres = 1200*SI.m_e*SI.c**2/SI.e  # [eV], 1200 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.8e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    gammas = proper_velocity2gamma(uzs)
    gamma = weighted_mean(gammas, beam.weightings(), clean=False) 
    assert np.isclose(beam.gamma(clean=False), gamma, rtol=1e-15, atol=0.0)
    gamma = weighted_mean(gammas, beam.weightings(), clean=True) 
    assert np.isclose(beam.gamma(clean=True), gamma, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_total_energy():
    
    num_particles = 4344
    energy_thres = 1200*SI.m_e*SI.c**2/SI.e  # [eV], 1200 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.8e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    total_energy = SI.e * np.nansum(beam.weightings()*beam.Es()) 
    assert np.isclose(beam.total_energy(), total_energy, rtol=1e-15, atol=0.0)

    # TODO: set some nans in beam


@pytest.mark.beam
def test_energy_spread():
    
    num_particles = 4344
    energy_thres = 1200*SI.m_e*SI.c**2/SI.e  # [eV], 1200 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.8e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    Es = proper_velocity2energy(uzs, unit='eV', m=SI.m_e)
    energy_spread = weighted_std(Es, beam.weightings(), clean=False) 
    assert np.isclose(beam.energy_spread(clean=False), energy_spread, rtol=1e-15, atol=0.0)
    energy_spread = weighted_std(Es, beam.weightings(), clean=True) 
    assert np.isclose(beam.energy_spread(clean=True), energy_spread, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_rel_energy_spread():
    
    num_particles = 4344
    energy_thres = 1200*SI.m_e*SI.c**2/SI.e  # [eV], 1200 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.8e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    Es = proper_velocity2energy(uzs, unit='eV', m=SI.m_e)
    rel_energy_spread = weighted_std(Es, beam.weightings(), clean=False)/weighted_mean(Es, beam.weightings(), clean=False) 
    assert np.isclose(beam.rel_energy_spread(clean=False), rel_energy_spread, rtol=1e-15, atol=0.0)
    rel_energy_spread = weighted_std(Es, beam.weightings(), clean=True)/weighted_mean(Es, beam.weightings(), clean=True) 
    assert np.isclose(beam.rel_energy_spread(clean=True), rel_energy_spread, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_xyz_offset():
    
    num_particles = 4344
    energy_thres = 1200*SI.m_e*SI.c**2/SI.e  # [eV], 1200 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.8e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    x_offset = weighted_mean(xs, beam.weightings(), clean=False)
    y_offset = weighted_mean(ys, beam.weightings(), clean=False)
    z_offset = weighted_mean(zs, beam.weightings(), clean=False)
    assert np.isclose(beam.x_offset(clean=False), x_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.y_offset(clean=False), y_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.z_offset(clean=False), z_offset, rtol=1e-15, atol=0.0)
    x_offset = weighted_mean(xs, beam.weightings(), clean=True)
    y_offset = weighted_mean(ys, beam.weightings(), clean=True)
    z_offset = weighted_mean(zs, beam.weightings(), clean=True)
    assert np.isclose(beam.x_offset(clean=True), x_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.y_offset(clean=True), y_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.z_offset(clean=True), z_offset, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_xpyp_offset():
    
    num_particles = 4344
    energy_thres = 1200*SI.m_e*SI.c**2/SI.e  # [eV], 1200 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    xps = np.random.rand(num_particles)
    yps = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.8e10
    beam.set_phase_space(Q, xs, ys, zs, uzs=uzs, xps=xps, yps=yps)

    x_angle = weighted_mean(xps, beam.weightings(), clean=False)
    y_angle = weighted_mean(yps, beam.weightings(), clean=False)
    assert np.isclose(beam.x_angle(clean=False), x_angle, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.y_angle(clean=False), y_angle, rtol=1e-15, atol=0.0)
    x_angle = weighted_mean(xps, beam.weightings(), clean=True)
    y_angle = weighted_mean(yps, beam.weightings(), clean=True)
    assert np.isclose(beam.x_angle(clean=True), x_angle, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.y_angle(clean=True), y_angle, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_uxuyuz_offset():
    
    num_particles = 4344
    energy_thres = 1200*SI.m_e*SI.c**2/SI.e  # [eV], 1200 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.8e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    ux_offset = weighted_mean(uxs, beam.weightings(), clean=False)
    uy_offset = weighted_mean(uys, beam.weightings(), clean=False)
    uz_offset = weighted_mean(uzs, beam.weightings(), clean=False)
    assert np.isclose(beam.ux_offset(clean=False), ux_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.uy_offset(clean=False), uy_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.uz_offset(clean=False), uz_offset, rtol=1e-15, atol=0.0)
    ux_offset = weighted_mean(uxs, beam.weightings(), clean=True)
    uy_offset = weighted_mean(uys, beam.weightings(), clean=True)
    uz_offset = weighted_mean(uzs, beam.weightings(), clean=True)
    assert np.isclose(beam.ux_offset(clean=True), ux_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.uy_offset(clean=True), uy_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.uz_offset(clean=True), uz_offset, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_geom_emittance_xy():
    num_particles = 4444
    energy_thres = 1400*SI.m_e*SI.c**2/SI.e  # [eV], 1400 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    xps = np.random.rand(num_particles)
    yps = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.4e10
    beam.set_phase_space(Q, xs, ys, zs, uzs=uzs, xps=xps, yps=yps)

    geom_emittance_x = np.sqrt(np.linalg.det(weighted_cov(xs, xps, beam.weightings(), clean=False)))
    geom_emittance_y = np.sqrt(np.linalg.det(weighted_cov(ys, yps, beam.weightings(), clean=False)))
    assert np.isclose(beam.geom_emittance_x(clean=False), geom_emittance_x, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.geom_emittance_y(clean=False), geom_emittance_y, rtol=1e-15, atol=0.0)

    geom_emittance_x = np.sqrt(np.linalg.det(weighted_cov(xs, xps, beam.weightings(), clean=True)))
    geom_emittance_y = np.sqrt(np.linalg.det(weighted_cov(ys, yps, beam.weightings(), clean=True)))
    assert np.isclose(beam.geom_emittance_x(clean=True), geom_emittance_x, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.geom_emittance_y(clean=True), geom_emittance_y, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_norm_emittance_xy():
    num_particles = 4444
    energy_thres = 1400*SI.m_e*SI.c**2/SI.e  # [eV], 1400 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.4e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    norm_emittance_x = np.sqrt(np.linalg.det(weighted_cov(xs, uxs/SI.c, beam.weightings(), clean=False)))
    norm_emittance_y = np.sqrt(np.linalg.det(weighted_cov(ys, uys/SI.c, beam.weightings(), clean=False)))
    assert np.isclose(beam.norm_emittance_x(clean=False), norm_emittance_x, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(clean=False), norm_emittance_y, rtol=1e-15, atol=0.0)

    norm_emittance_x = np.sqrt(np.linalg.det(weighted_cov(xs, uxs/SI.c, beam.weightings(), clean=True)))
    norm_emittance_y = np.sqrt(np.linalg.det(weighted_cov(ys, uys/SI.c, beam.weightings(), clean=True)))
    assert np.isclose(beam.norm_emittance_x(clean=True), norm_emittance_x, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(clean=True), norm_emittance_y, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_beta_xy():
    num_particles = 4444
    energy_thres = 1400*SI.m_e*SI.c**2/SI.e  # [eV], 1400 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.4e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    covx = weighted_cov(xs, beam.xps(), beam.weightings(), clean=False)
    beta_x = covx[0,0]/np.sqrt(np.linalg.det(covx))
    covy = weighted_cov(ys, beam.yps(), beam.weightings(), clean=False)
    beta_y = covy[0,0]/np.sqrt(np.linalg.det(covy))
    assert np.isclose(beam.beta_x(clean=False), beta_x, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.beta_y(clean=False), beta_y, rtol=1e-15, atol=0.0)

    covx = weighted_cov(xs, beam.xps(), beam.weightings(), clean=True)
    beta_x = covx[0,0]/np.sqrt(np.linalg.det(covx))
    covy = weighted_cov(ys, beam.yps(), beam.weightings(), clean=True)
    beta_y = covy[0,0]/np.sqrt(np.linalg.det(covy))
    assert np.isclose(beam.beta_x(clean=True), beta_x, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.beta_y(clean=True), beta_y, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_gamma_xy():
    num_particles = 8989
    energy_thres = 11*SI.m_e*SI.c**2/SI.e  # [eV], 11 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(10e12, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.0e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    covx = weighted_cov(xs, beam.xps(), beam.weightings(), clean=False)
    gamma_x = covx[1,1]/np.sqrt(np.linalg.det(covx))
    covy = weighted_cov(ys, beam.yps(), beam.weightings(), clean=False)
    gamma_y = covy[1,1]/np.sqrt(np.linalg.det(covy))
    assert np.isclose(beam.gamma_x(), gamma_x, rtol=1e-10, atol=0.0)
    assert np.isclose(beam.gamma_y(), gamma_y, rtol=1e-10, atol=0.0)

    covx = weighted_cov(xs, beam.xps(), beam.weightings(), clean=True)
    gamma_x = covx[1,1]/np.sqrt(np.linalg.det(covx))
    covy = weighted_cov(ys, beam.yps(), beam.weightings(), clean=True)
    gamma_y = covy[1,1]/np.sqrt(np.linalg.det(covy))
    assert np.isclose(beam.gamma_x(clean=True), gamma_x, rtol=1e-10, atol=0.0)
    assert np.isclose(beam.gamma_y(clean=True), gamma_y, rtol=1e-10, atol=0.0)


@pytest.mark.beam
def test_beam_size_xy():
    
    num_particles = 4444
    energy_thres = 1400*SI.m_e*SI.c**2/SI.e  # [eV], 1400 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.4e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    beam_size_x = weighted_std(xs, beam.weightings(), clean=False)
    beam_size_y = weighted_std(ys, beam.weightings(), clean=False)
    assert np.isclose(beam.beam_size_x(clean=False), beam_size_x, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.beam_size_y(clean=False), beam_size_y, rtol=1e-15, atol=0.0)

    beam_size_x = weighted_std(xs, beam.weightings(), clean=True)
    beam_size_y = weighted_std(ys, beam.weightings(), clean=True) 
    assert np.isclose(beam.beam_size_x(clean=True), beam_size_x, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.beam_size_y(clean=True), beam_size_y, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_bunch_length():
    num_particles = 4444
    energy_thres = 1400*SI.m_e*SI.c**2/SI.e  # [eV], 1400 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), num_particles)
    Q = -SI.e * 1.4e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    bunch_length = weighted_std(zs, beam.weightings(), clean=False)
    assert np.isclose(beam.bunch_length(clean=False), bunch_length, rtol=1e-15, atol=0.0)

    bunch_length = weighted_std(zs, beam.weightings(), clean=True)
    assert np.isclose(beam.bunch_length(clean=True), bunch_length, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_divergence_xy():
    
    num_particles = 4444
    energy_thres = 1400*SI.m_e*SI.c**2/SI.e  # [eV], 1400 * particle rest energy.
    uz_thres = energy2proper_velocity(energy_thres, unit='eV', m=SI.m_e)
    
    beam = Beam()
    xs = np.random.rand(num_particles)
    ys = np.random.rand(num_particles)
    zs = np.random.rand(num_particles)
    uxs = np.random.rand(num_particles)
    uys = np.random.rand(num_particles)
    uzs = np.random.uniform(uz_thres, energy2proper_velocity(energy_thres*1.01, unit='eV', m=SI.m_e), size=num_particles)
    Q = -SI.e * 1.4e10
    beam.set_phase_space(Q, xs, ys, zs, uxs, uys, uzs)

    divergence_x = weighted_std(beam.xps(), beam.weightings(), clean=False)
    divergence_y = weighted_std(beam.yps(), beam.weightings(), clean=False)
    assert np.isclose(beam.divergence_x(clean=False), divergence_x, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.divergence_y(clean=False), divergence_y, rtol=1e-15, atol=0.0)

    divergence_x = weighted_std(beam.xps(), beam.weightings(), clean=True)
    divergence_y = weighted_std(beam.yps(), beam.weightings(), clean=True)
    assert np.isclose(beam.divergence_x(clean=True), divergence_x, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.divergence_y(clean=True), divergence_y, rtol=1e-15, atol=0.0)



############# Tests for beam projections #############
@pytest.mark.beam
def test_current_profile():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=1.0e-6, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()

    dQdt, ts = beam.current_profile()
    assert np.isclose(np.max(np.abs(dQdt)), beam.peak_current(), rtol=1e-15, atol=0.0)
    assert np.isclose( np.sum( dQdt*np.diff(ts)[0] ), beam.charge(), rtol=1e-10, atol=0.0 )
    assert np.isclose(SI.c*ts.mean(), beam.z_offset(), rtol=1e-10, atol=0.0)

    Nbins = int(np.sqrt(len(beam)/2))
    bins = np.linspace(min(beam.ts()), max(beam.ts()), Nbins)
    dQdt, ts = beam.current_profile(bins=bins)
    assert np.isclose(np.max(np.abs(dQdt)), beam.peak_current(), rtol=1e-15, atol=0.0)
    assert len(dQdt) == Nbins - 1
    assert len(ts) == Nbins - 1
    assert np.isclose( np.sum( dQdt*np.diff(ts)[0] ), beam.charge(), rtol=1e-10, atol=0.0 )
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

    # Purpposedly trigger exceptions
    with pytest.raises(ValueError):
        beam = source.track()
        axis1 = np.array([1, 1.1, 0])  # Axis as an non-unit vector.
        axis2 = np.array([0, 1, 0])
        axis3 = np.array([1, 0, 0])
        beam.rotate_coord_sys_3D(axis1, angle1=nom_x_angle, axis2=axis2, angle2=nom_y_angle, axis3=axis3, angle3=0.0, invert=False)
    with pytest.raises(ValueError):
        beam = source.track()
        axis1 = np.array([1, 0, 0])  # Axis as an non-unit vector.
        axis2 = np.array([0, 0.45, 0])
        axis3 = np.array([1, 0, 0])
        beam.rotate_coord_sys_3D(axis1, angle1=nom_x_angle, axis2=axis2, angle2=nom_y_angle, axis3=axis3, angle3=0.0, invert=False)
    with pytest.raises(ValueError):
        beam = source.track()
        axis1 = np.array([1, 0, 0])  # Axis as an non-unit vector.
        axis2 = np.array([0, 1, 0])
        axis3 = np.array([-0.1, 0, 0])
        beam.rotate_coord_sys_3D(axis1, angle1=nom_x_angle, axis2=axis2, angle2=nom_y_angle, axis3=axis3, angle3=0.0, invert=False)
    with pytest.raises(ValueError):
        beam = source.track()
        axis1 = np.array([1, 0, -3.14])  # Axis as an non-unit vector.
        axis2 = np.array([0.3, 1, 0])
        axis3 = np.array([-0.1, 0, 0])
        beam.rotate_coord_sys_3D(axis1, angle1=nom_x_angle, axis2=axis2, angle2=nom_y_angle, axis3=axis3, angle3=0.0, invert=False)



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



############# Tests of methods changing the beam #############
@pytest.mark.beam
def test_accelerate_nominal():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    energy_gain = 1e9  # 1 GeV energy gain
    chirp = 0  # No chirp
    initial_Es = beam.Es()
        
    beam.accelerate(energy_gain=energy_gain, chirp=chirp)
    expected_Es = initial_Es + energy_gain

    assert np.allclose(beam.Es(), expected_Es, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_accelerate_negative():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    energy_gain = -beam.energy()  # Energy decrease
    chirp = 0  # No chirp

    # Suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        try:
            beam.accelerate(energy_gain=energy_gain, chirp=chirp)
        except ValueError as err:
            assert str(err) == 'uzs contains values that are too small.'


@pytest.mark.beam
def test_accelerate_chirp():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    energy_gain = 1e9  # 1 GeV energy gain
    chirp = 1e8  # Energy chirp dE/dz
    zs = beam.zs()
    initial_Es = beam.Es()

    expected_Es = energy_gain + initial_Es + np.sign(beam.qs()) * zs * chirp
    beam.accelerate(energy_gain=energy_gain, chirp=chirp)

    assert np.allclose(beam.Es(), expected_Es, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_compress_nominal():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()

    R_56 = 0.5**2*SI.c**2*np.sqrt(beam.energy()/10e9)**3/(3*beam.energy()**2)
    initial_zs = beam.zs()
    initial_Es = beam.Es()
    
    expected_zs = initial_zs + (1 - initial_Es / beam.energy()) * R_56
    beam.compress(R_56=R_56, nom_energy=beam.energy())
    
    assert np.allclose(beam.zs(), expected_zs, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_compress_null():

    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()

    R_56 = 0  # No compression
    initial_zs = beam.zs()
    
    beam.compress(R_56=R_56, nom_energy=beam.energy())
    
    # zs should remain unchanged
    assert np.allclose(beam.zs(), initial_zs, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_compress_uniform():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()

    R_56 = 0.5**2*SI.c**2*np.sqrt(beam.energy()/10e9)**3/(3*beam.energy()**2)
    uniform_energy = np.full(len(beam.Es()), beam.energy())
    beam.set_Es(uniform_energy)
    initial_zs = beam.zs()

    beam.compress(R_56=R_56, nom_energy=beam.energy())
    
    assert np.allclose(beam.zs(), initial_zs, rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_scale_to_length():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    
    beam = source.track()
    new_bunch_length = beam.bunch_length()/2
    zs_scaled = beam.z_offset() + (beam.zs()-beam.z_offset())*new_bunch_length/beam.bunch_length()
    beam.scale_to_length(bunch_length=new_bunch_length)
    assert np.allclose(beam.zs(), zs_scaled, rtol=1e-15, atol=0.0)
        
    beam = source.track()
    new_bunch_length = beam.bunch_length()*5.354
    zs_scaled = beam.z_offset() + (beam.zs()-beam.z_offset())*new_bunch_length/beam.bunch_length()
    beam.scale_to_length(bunch_length=new_bunch_length)
    assert np.allclose(beam.zs(), zs_scaled, rtol=1e-15, atol=0.0)

    beam = source.track()
    new_bunch_length = beam.bunch_length()*0
    beam.scale_to_length(bunch_length=new_bunch_length)
    assert np.allclose(beam.zs(), beam.z_offset(), rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_scale_norm_emittance_x():

    np.random.seed(42)
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    
    beam = source.track()
    scale_factor = 15.16
    expct_emit_nx = scale_factor * beam.norm_emittance_x()
    expected_xs = beam.xs() * np.sqrt(scale_factor)
    expected_uxs = beam.uxs() * np.sqrt(scale_factor)
    beam.scale_norm_emittance_x(scale_factor*beam.norm_emittance_x())
    assert np.allclose(beam.xs(), expected_xs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uxs(), expected_uxs, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.norm_emittance_x(), expct_emit_nx, rtol=1e-10, atol=0.0)

    beam = source.track()
    scale_factor = 0.5354
    expct_emit_nx = scale_factor * beam.norm_emittance_x()
    expected_xs = beam.xs() * np.sqrt(scale_factor)
    expected_uxs = beam.uxs() * np.sqrt(scale_factor)
    beam.scale_norm_emittance_x(scale_factor*beam.norm_emittance_x())
    assert np.allclose(beam.xs(), expected_xs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uxs(), expected_uxs, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.norm_emittance_x(), expct_emit_nx, rtol=1e-10, atol=0.0)

    beam = source.track()
    scale_factor = 0.0
    expct_emit_nx = scale_factor * beam.norm_emittance_x()
    expected_xs = beam.xs() * np.sqrt(scale_factor)
    expected_uxs = beam.uxs() * np.sqrt(scale_factor)
    beam.scale_norm_emittance_x(scale_factor*beam.norm_emittance_x())
    assert np.allclose(beam.xs(), expected_xs, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uxs(), expected_uxs, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.norm_emittance_x(), expct_emit_nx, rtol=1e-15, atol=0.0)

    try:
        beam.scale_norm_emittance_x(-8e-6)
    except ValueError as err:
        assert str(err) == 'Normalised emittance cannot be negative.'


@pytest.mark.beam
def test_scale_norm_emittance_y():

    np.random.seed(42)
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    
    beam = source.track()
    scale_factor = 15.16
    expct_emit_ny = scale_factor * beam.norm_emittance_y()
    expected_ys = beam.ys() * np.sqrt(scale_factor)
    expected_uys = beam.uys() * np.sqrt(scale_factor)
    beam.scale_norm_emittance_y(scale_factor*beam.norm_emittance_y())
    assert np.allclose(beam.ys(), expected_ys, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uys(), expected_uys, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), expct_emit_ny, rtol=1e-13, atol=0.0)

    beam = source.track()
    scale_factor = 0.5354
    expct_emit_ny = scale_factor * beam.norm_emittance_y()
    expected_ys = beam.ys() * np.sqrt(scale_factor)
    expected_uys = beam.uys() * np.sqrt(scale_factor)
    beam.scale_norm_emittance_y(scale_factor*beam.norm_emittance_y())
    assert np.allclose(beam.ys(), expected_ys, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uys(), expected_uys, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), expct_emit_ny, rtol=1e-13, atol=0.0)

    beam = source.track()
    scale_factor = 0.0
    expct_emit_ny = scale_factor * beam.norm_emittance_y()
    expected_ys = beam.ys() * np.sqrt(scale_factor)
    expected_uys = beam.uys() * np.sqrt(scale_factor)
    beam.scale_norm_emittance_y(scale_factor*beam.norm_emittance_y())
    assert np.allclose(beam.ys(), expected_ys, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uys(), expected_uys, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.norm_emittance_y(), expct_emit_ny, rtol=1e-15, atol=0.0)

    try:
        beam.scale_norm_emittance_y(-8e-6)
    except ValueError as err:
        assert str(err) == 'Normalised emittance cannot be negative.'


@pytest.mark.beam
def test_apply_betatron_damping():

    np.random.seed(42)

    x_offset = 5.354e-6
    y_offset = 0.1516e-6
    ux_offset = 739215.7882185677
    uy_offset = 1583681.8243787317
    deltaE = 1.516e9

    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=np.sqrt(3/4.516), energy=3e9, rel_energy_spread=0.0, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=ux_offset/energy2proper_velocity(3e9), y_angle=uy_offset/energy2proper_velocity(3e9))

    beam = source.track()
    initial_beam = copy.deepcopy(beam)

    # calculate beam (not beta) magnification
    gammasBoosted = energy2gamma(abs(initial_beam.Es() + deltaE))
    beta_mag = np.sqrt(initial_beam.gammas()/gammasBoosted)
    assert np.isclose(beta_mag.mean(), np.sqrt(3/4.516), rtol=1e-15, atol=0.0)
    mag = np.sqrt(beta_mag)

    beam.apply_betatron_damping(deltaE, axis_defining_beam=beam)

    # Examine beam
    assert len(beam) == len(initial_beam)
    assert np.isclose(beam.particle_mass, SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge(), initial_beam.charge(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge_sign(), -1, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.total_energy(), initial_beam.total_energy(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.weightings(), initial_beam.weightings(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.energy(), initial_beam.energy(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.rel_energy_spread(), initial_beam.rel_energy_spread(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.z_offset(), initial_beam.z_offset(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.bunch_length(), initial_beam.bunch_length(), rtol=1e-15, atol=0.0)

    assert np.isclose(beam.x_offset(), x_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.y_offset(), y_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.beam_size_x(), initial_beam.beam_size_x()*mag.mean(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.beam_size_y(), initial_beam.beam_size_y()*mag.mean(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.x_angle(), ux_offset/energy2proper_velocity(3e9), rtol=1e-12, atol=0.0)
    assert np.isclose(beam.y_angle(), uy_offset/energy2proper_velocity(3e9), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.divergence_x(), initial_beam.divergence_x()/mag.mean(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.divergence_y(), initial_beam.divergence_y()/mag.mean(), rtol=1e-15, atol=0.0)

    assert np.allclose(beam.xs(), (initial_beam.xs()-initial_beam.x_offset())*mag + initial_beam.x_offset(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.ys(), (initial_beam.ys()-initial_beam.y_offset())*mag + initial_beam.y_offset(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uxs(), (initial_beam.uxs()-initial_beam.ux_offset())/mag + initial_beam.ux_offset(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uys(), (initial_beam.uys()-initial_beam.uy_offset())/mag + initial_beam.uy_offset(), rtol=1e-15, atol=0.0)


    # Test no energy change
    deltaE = 0.0

    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=1.0, energy=3e9, rel_energy_spread=0.01, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=ux_offset/energy2proper_velocity(3e9), y_angle=uy_offset/energy2proper_velocity(3e9))

    beam = source.track()
    initial_beam = copy.deepcopy(beam)

    # calculate beam (not beta) magnification
    gammasBoosted = energy2gamma(abs(initial_beam.Es() + deltaE))
    beta_mag = np.sqrt(initial_beam.gammas()/gammasBoosted)
    assert np.isclose(beta_mag.mean(), 1.0, rtol=1e-15, atol=0.0)
    mag = np.sqrt(beta_mag)

    beam.apply_betatron_damping(deltaE, axis_defining_beam=beam)

    # Examine beam
    assert len(beam) == len(initial_beam)
    assert np.isclose(beam.particle_mass, SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge(), initial_beam.charge(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge_sign(), -1, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.total_energy(), initial_beam.total_energy(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.weightings(), initial_beam.weightings(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.energy(), initial_beam.energy(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.rel_energy_spread(), initial_beam.rel_energy_spread(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.z_offset(), initial_beam.z_offset(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.bunch_length(), initial_beam.bunch_length(), rtol=1e-15, atol=0.0)

    assert np.isclose(beam.x_offset(), x_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.y_offset(), y_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.beam_size_x(), initial_beam.beam_size_x()*mag.mean(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.beam_size_y(), initial_beam.beam_size_y()*mag.mean(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.x_angle(), ux_offset/energy2proper_velocity(3e9), rtol=1e-12, atol=0.0)
    assert np.isclose(beam.y_angle(), uy_offset/energy2proper_velocity(3e9), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.divergence_x(), initial_beam.divergence_x()/mag.mean(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.divergence_y(), initial_beam.divergence_y()/mag.mean(), rtol=1e-15, atol=0.0)

    assert np.allclose(beam.xs(), (initial_beam.xs()-initial_beam.x_offset())*mag + initial_beam.x_offset(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.ys(), (initial_beam.ys()-initial_beam.y_offset())*mag + initial_beam.y_offset(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uxs(), (initial_beam.uxs()-initial_beam.ux_offset())/mag + initial_beam.ux_offset(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uys(), (initial_beam.uys()-initial_beam.uy_offset())/mag + initial_beam.uy_offset(), rtol=1e-15, atol=0.0)
    

@pytest.mark.beam
def test_magnify_beta_function():
    "Tests for ``Beam.magnify_beta_function(beta_mag, axis_defining_beam=None)``, which magnifies beta functions (increases beam size, decreases divergence for beta_mag > 1.0)."

    np.random.seed(42)

    x_offset = 5.354e-6
    y_offset = 0.1516e-6
    ux_offset = 0.0
    uy_offset = 1583681.8243787317
    beta_mag = 5.0  # Magnify

    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=beta_mag, energy=3e9, rel_energy_spread=0.0, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=ux_offset, y_angle=uy_offset/energy2proper_velocity(3e9))
    beam = source.track()
    initial_beam = copy.deepcopy(beam)

    beam.magnify_beta_function(beta_mag=beta_mag, axis_defining_beam=beam)
    
    # calculate beam (not beta) magnification
    mag = np.sqrt(beta_mag)

    # Examine beam
    assert len(beam) == len(initial_beam)
    assert np.isclose(beam.particle_mass, SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge(), initial_beam.charge(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge_sign(), -1, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.total_energy(), initial_beam.total_energy(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.weightings(), initial_beam.weightings(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.energy(), initial_beam.energy(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.rel_energy_spread(), initial_beam.rel_energy_spread(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.z_offset(), initial_beam.z_offset(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.bunch_length(), initial_beam.bunch_length(), rtol=1e-15, atol=0.0)

    assert np.isclose(beam.x_offset(), x_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.y_offset(), y_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.beam_size_x(), initial_beam.beam_size_x()*mag, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.beam_size_y(), initial_beam.beam_size_y()*mag, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.x_angle(), ux_offset/energy2proper_velocity(3e9), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.y_angle(), uy_offset/energy2proper_velocity(3e9), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.divergence_x(), initial_beam.divergence_x()/mag, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.divergence_y(), initial_beam.divergence_y()/mag, rtol=1e-15, atol=0.0)

    assert np.allclose(beam.xs(), (initial_beam.xs()-initial_beam.x_offset())*mag + initial_beam.x_offset(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.ys(), (initial_beam.ys()-initial_beam.y_offset())*mag + initial_beam.y_offset(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uxs(), (initial_beam.uxs()-initial_beam.ux_offset())/mag + initial_beam.ux_offset(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uys(), (initial_beam.uys()-initial_beam.uy_offset())/mag + initial_beam.uy_offset(), rtol=1e-15, atol=0.0)


    beta_mag = 1/beta_mag  # De-magnify
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=beta_mag, energy=3e9, rel_energy_spread=0.0, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=ux_offset, y_angle=uy_offset/energy2proper_velocity(3e9))
    beam = source.track()
    initial_beam = copy.deepcopy(beam)

    beam.magnify_beta_function(beta_mag=beta_mag, axis_defining_beam=beam)
    
    # calculate beam (not beta) magnification
    mag = np.sqrt(beta_mag)

    # Examine beam
    assert len(beam) == len(initial_beam)
    assert np.isclose(beam.particle_mass, SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge(), initial_beam.charge(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.charge_sign(), -1, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.total_energy(), initial_beam.total_energy(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.weightings(), initial_beam.weightings(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.energy(), initial_beam.energy(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.rel_energy_spread(), initial_beam.rel_energy_spread(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.z_offset(), initial_beam.z_offset(), rtol=1e-15, atol=0.0)
    assert np.isclose(beam.bunch_length(), initial_beam.bunch_length(), rtol=1e-15, atol=0.0)

    assert np.isclose(beam.x_offset(), x_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.y_offset(), y_offset, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.beam_size_x(), initial_beam.beam_size_x()*mag, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.beam_size_y(), initial_beam.beam_size_y()*mag, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.x_angle(), ux_offset/energy2proper_velocity(3e9), rtol=0, atol=1e-15)
    assert np.isclose(beam.y_angle(), uy_offset/energy2proper_velocity(3e9), rtol=0, atol=1e-15)
    assert np.isclose(beam.divergence_x(), initial_beam.divergence_x()/mag, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.divergence_y(), initial_beam.divergence_y()/mag, rtol=1e-15, atol=0.0)

    assert np.allclose(beam.xs(), (initial_beam.xs()-initial_beam.x_offset())*mag + initial_beam.x_offset(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.ys(), (initial_beam.ys()-initial_beam.y_offset())*mag + initial_beam.y_offset(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uxs(), (initial_beam.uxs()-initial_beam.ux_offset())/mag + initial_beam.ux_offset(), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.uys(), (initial_beam.uys()-initial_beam.uy_offset())/mag + initial_beam.uy_offset(), rtol=1e-15, atol=0.0)


@pytest.mark.beam
def test_transport():

    x_offset = 0                                                                    # [m]
    y_offset = 0                                                                    # [m]
    x_angle = 0                                                                     # [rad]
    y_angle = 0                                                                     # [rad]
    L = 0                                                                           # [m]
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, rel_energy_spread=0.0, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=x_angle, y_angle=y_angle)
    beam = source.track()
    initial_beam = copy.deepcopy(beam)

    beam.transport(L)
    assert np.allclose(beam.xs(), initial_beam.xs()+L*initial_beam.xps())
    assert np.allclose(beam.ys(), initial_beam.ys()+L*initial_beam.yps())

    x_offset = 5.354e-6                                                             # [m]
    y_offset = 0.1516e-6                                                            # [m]
    x_angle = 6.6e-6                                                                # [rad]
    y_angle = 0.42e-6                                                               # [rad]
    L = 0.0                                                                         # [m]
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, rel_energy_spread=0.0, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=x_angle, y_angle=y_angle)
    beam = source.track()
    initial_beam = copy.deepcopy(beam)
    beam.transport(L)
    assert np.allclose(beam.xs(), initial_beam.xs()+L*initial_beam.xps())
    assert np.allclose(beam.ys(), initial_beam.ys()+L*initial_beam.yps())

    x_offset = 5.354e-6                                                             # [m]
    y_offset = 0.1516e-6                                                            # [m]
    x_angle = 6.6e-6                                                                # [rad]
    y_angle = 0.42e-6                                                               # [rad]
    L = 7.8                                                                         # [m]
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, rel_energy_spread=0.0, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=x_angle, y_angle=y_angle)
    beam = source.track()
    initial_beam = copy.deepcopy(beam)
    beam.transport(L)
    assert np.allclose(beam.xs(), initial_beam.xs()+L*initial_beam.xps())
    assert np.allclose(beam.ys(), initial_beam.ys()+L*initial_beam.yps())

    x_offset = -5.354e-6                                                            # [m]
    y_offset = 0.1516e-6                                                            # [m]
    x_angle = 6.6e-6                                                                # [rad]
    y_angle = -0.42e-6                                                              # [rad]
    L = 7.8                                                                         # [m]
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, rel_energy_spread=0.0, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=x_angle, y_angle=y_angle)
    beam = source.track()
    initial_beam = copy.deepcopy(beam)
    beam.transport(L)
    assert np.allclose(beam.xs(), initial_beam.xs()+L*initial_beam.xps())
    assert np.allclose(beam.ys(), initial_beam.ys()+L*initial_beam.yps())

    x_offset = -5.354e-6                                                            # [m]
    y_offset = 0.1516e-6                                                            # [m]
    x_angle = 6.6e-6                                                                # [rad]
    y_angle = -0.42e-6                                                              # [rad]
    L = -7.8                                                                        # [m]
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, rel_energy_spread=0.0, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=x_angle, y_angle=y_angle)
    beam = source.track()
    initial_beam = copy.deepcopy(beam)
    beam.transport(L)
    assert np.allclose(beam.xs(), initial_beam.xs()+L*initial_beam.xps())
    assert np.allclose(beam.ys(), initial_beam.ys()+L*initial_beam.yps())


@pytest.mark.beam
def test_flip_transverse_phase_spaces():

    x_offset = 0                                                                    # [m]
    y_offset = 0                                                                    # [m]
    x_angle = 0                                                                     # [rad]
    y_angle = 0                                                                     # [rad]
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, rel_energy_spread=0.0, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=x_angle, y_angle=y_angle)
    beam = source.track()
    initial_beam = copy.deepcopy(beam)
    beam.flip_transverse_phase_spaces(flip_momenta=False, flip_positions=False)
    assert np.allclose(beam.xs(), initial_beam.xs())
    assert np.allclose(beam.ys(), initial_beam.ys())
    assert np.allclose(beam.uxs(), initial_beam.uxs())
    assert np.allclose(beam.uys(), initial_beam.uys())

    x_offset = -5.354e-6                                                            # [m]
    y_offset = 0.1516e-6                                                            # [m]
    x_angle = 6.6e-6                                                                # [rad]
    y_angle = -0.42e-6                                                              # [rad]
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, rel_energy_spread=0.0, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=x_angle, y_angle=y_angle)
    beam = source.track()
    initial_beam = copy.deepcopy(beam)
    beam.flip_transverse_phase_spaces(flip_momenta=True, flip_positions=False)
    assert np.allclose(beam.xs(), initial_beam.xs())
    assert np.allclose(beam.ys(), initial_beam.ys())
    assert np.allclose(beam.uxs(), -initial_beam.uxs())
    assert np.allclose(beam.uys(), -initial_beam.uys())

    x_offset = 5.354e-6                                                             # [m]
    y_offset = -0.1516e-6                                                           # [m]
    x_angle = -6.6e-6                                                               # [rad]
    y_angle = -0.42e-6                                                              # [rad]
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, rel_energy_spread=0.0, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=x_angle, y_angle=y_angle)
    beam = source.track()
    initial_beam = copy.deepcopy(beam)
    beam.flip_transverse_phase_spaces(flip_momenta=False, flip_positions=True)
    assert np.allclose(beam.xs(), -initial_beam.xs())
    assert np.allclose(beam.ys(), -initial_beam.ys())
    assert np.allclose(beam.uxs(), initial_beam.uxs())
    assert np.allclose(beam.uys(), initial_beam.uys())

    x_offset = 5.354e-6                                                             # [m]
    y_offset = 0.1516e-6                                                            # [m]
    x_angle = 6.6e-6                                                                # [rad]
    y_angle = 0.42e-6                                                               # [rad]
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, rel_energy_spread=0.0, z_offset=0.0, y_offset=y_offset, x_offset=x_offset, x_angle=x_angle, y_angle=y_angle)
    beam = source.track()
    initial_beam = copy.deepcopy(beam)
    beam.flip_transverse_phase_spaces(flip_momenta=True, flip_positions=True)
    assert np.allclose(beam.xs(), -initial_beam.xs())
    assert np.allclose(beam.ys(), -initial_beam.ys())
    assert np.allclose(beam.uxs(), -initial_beam.uxs())
    assert np.allclose(beam.uys(), -initial_beam.uys())


@pytest.mark.beam
def test_apply_betatron_motion_emitt_pres():
    "Test of ``Beam.apply_betatron_motion()`` with no energy spread and radiation reaction so that the emittances are preserved."

    np.random.seed(42)

    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, rel_energy_spread=0.0, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    initial_beam = copy.deepcopy(beam)
    deltaEs = np.full(len(beam.Es()), 1e9)  # Homogeneous energy gain

    Es_final, evol = beam.apply_betatron_motion(L=1.0, n0=6.0e20, deltaEs=deltaEs, x0_driver=0, y0_driver=0, radiation_reaction=False, calc_evolution=True)

    assert np.allclose(Es_final.mean(), 4e9, rtol=1e-15, atol=0.0)
    assert np.allclose(np.std(Es_final), 1e-20, rtol=0.0, atol=1e-19)
    assert np.allclose(np.std(Es_final)/Es_final.mean(), 1e-20, rtol=0.0, atol=1e-19)
    assert np.allclose(beam.charge(), initial_beam.charge(), rtol=1e-10, atol=0.0)
    assert np.allclose(beam.z_offset(), initial_beam.z_offset(), rtol=1e-10, atol=0.0)
    assert np.allclose(beam.bunch_length(), initial_beam.bunch_length(), rtol=1e-10, atol=0.0)

    assert np.allclose(Es_final.mean(), evol.energy[-1], rtol=1e-20, atol=0.0)
    assert np.allclose(evol.energy_spread[-1], 1e-20, rtol=0.0, atol=1e-19)
    assert np.allclose(beam.charge(), evol.charge[-1], rtol=1e-10, atol=0.0)
    assert np.allclose(beam.norm_emittance_x(), evol.emit_nx[-1], rtol=0.0, atol=0.3e-5)
    assert np.allclose(beam.norm_emittance_y(), evol.emit_ny[-1], rtol=0.0, atol=0.5e-5)
    
    # Emittance is preserved when the energy spread is 0 (homogenneoud energy gain) and radiation reaction disabled.
    #assert np.isclose(beam.rel_energy_spread(), 1e-15, rtol=1e-10, atol=0.0) # Beam energies not updated after apply_betatron_motion(), so cannot check this.
    assert np.allclose(beam.norm_emittance_x(), initial_beam.norm_emittance_x(), rtol=1e-8, atol=0.0)
    assert np.allclose(beam.norm_emittance_y(), initial_beam.norm_emittance_y(), rtol=1e-8, atol=0.0)
    assert np.allclose(evol.emit_nx[0], evol.emit_nx[-1], rtol=1e-8, atol=0.0)
    assert np.allclose(evol.emit_ny[0], evol.emit_ny[-1], rtol=1e-8, atol=0.0)


@pytest.mark.beam
def test_apply_betatron_motion():
    "Test of ``Beam.apply_betatron_motion()`` with energy spread and radiation reaction."

    np.random.seed(42)

    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, rel_energy_spread=0.01, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    initial_beam = copy.deepcopy(beam)
    deltaEs = np.full(len(beam.Es()), 1e9)  # Homogeneous energy gain

    Es_final, evol = beam.apply_betatron_motion(L=1.0, n0=6.0e20, deltaEs=deltaEs, x0_driver=0, y0_driver=0, radiation_reaction=True, calc_evolution=True)

    assert np.allclose(Es_final.mean(), 4e9, rtol=1e-6, atol=0.0)
    assert np.allclose(np.std(Es_final), 0.01*3e9, rtol=5e-3, atol=0.0)
    #assert np.allclose(np.std(Es_final)/Es_final.mean(), 1e-20, rtol=0.0, atol=1e-19)
    assert np.allclose(beam.charge(), initial_beam.charge(), rtol=1e-10, atol=0.0)
    assert np.allclose(beam.z_offset(), initial_beam.z_offset(), rtol=1e-10, atol=0.0)
    assert np.allclose(beam.bunch_length(), initial_beam.bunch_length(), rtol=1e-10, atol=0.0)

    assert np.allclose(Es_final.mean(), evol.energy[-1], rtol=1e-3, atol=0.0)
    assert np.allclose(evol.energy_spread[-1], 0.01*3e9, rtol=5e-2, atol=0.0)
    assert np.allclose(beam.charge(), evol.charge[-1], rtol=1e-10, atol=0.0)
    assert np.allclose(beam.norm_emittance_x(), evol.emit_nx[-1], rtol=0.0, atol=0.3e-5)
    assert np.allclose(beam.norm_emittance_y(), evol.emit_ny[-1], rtol=0.0, atol=0.5e-5)


    #mag = np.sqrt(initial_beam.gamma()/energy2gamma(Es_final.mean()))  # Magnification factor
    #assert np.allclose(beam.beta_x(), initial_beam.beta_x()*mag, rtol=1e-15, atol=0.0)
    


############# Tests of plotting methods #############
@pytest.mark.beam
def test_plot_current_profile():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    beam.plot_current_profile()
    plt.close()


@pytest.mark.beam
def test_plot_lps():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    beam.plot_lps()
    plt.close()


@pytest.mark.beam
def test_plot_trace_space_x():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    beam.plot_trace_space_x()
    plt.close()


@pytest.mark.beam
def test_plot_trace_space_y():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    beam.plot_trace_space_y()
    plt.close()


@pytest.mark.beam
def test_plot_transverse_profile():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    beam.plot_transverse_profile()
    plt.close()


# @pytest.mark.beam
# def test_plot_bunch_pattern():
#     source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
#     beam = source.track()
#     beam.plot_bunch_pattern()


@pytest.mark.beam
def test_density_map_diags():

    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    beam.density_map_diags()
    plt.close()


@pytest.mark.beam
def test_print_summary():
    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()
    beam.beam_name = 'beam'
    beam.print_summary()



############# Tests of saving and loading #############
@pytest.mark.beam
def test_save_load():

    import os

    source = setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0, energy=3e9, rel_energy_spread=0.01, z_offset=0.0, x_offset=0.0, y_offset=0.0, x_angle=0.0, y_angle=0.0)
    beam = source.track()

    save_dir = 'tests' + os.sep + 'data' + os.sep + 'test_beam' + os.sep
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filename = save_dir + os.sep + 'beam_test_save.h5'

    beam.save(filename=filename)
    loaded_beam = Beam.load(filename=filename)

    Beam.comp_beams(beam, loaded_beam, comp_location=True, rtol=1e-15, atol=0.0)
    Beam.comp_beam_params(beam, loaded_beam, comp_location=True)
    shutil.rmtree(save_dir)