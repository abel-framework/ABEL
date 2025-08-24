# -*- coding: utf-8 -*-
"""
ABEL : unit tests for the RF-Track API
=======================================

This file is a part of ABEL.
Copyright 2022– C.A.Lindstrøm, J.B.B.Chen, O.G.Finnerud,
D.Kallvik, E.Hørlyk, K.N.Sjobak, E.Adli, University of Oslo

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
import numpy as np
import scipy.constants as SI
from abel.classes.source.impl.source_basic import SourceBasic
from abel.classes.beam import Beam

# def setup_basic_driver_source():
#     driver = SourceBasic()
#     driver.bunch_length = 42e-6                                                   # [m] This value is for trapezoid.
#     driver.z_offset = 300e-6                                                      # [m]

#     driver.num_particles = 100000                                                 
#     driver.charge = -2.7e10 * SI.e                                                # [C]
#     driver.energy = 31.25e9                                                       # [eV]
#     driver.rel_energy_spread = 0.01                                               # Relative rms energy spread

#     driver.emit_nx, driver.emit_ny = 50e-6, 100e-6                                # [m rad]
#     driver.beta_x, driver.beta_y = 0.5, 0.5                                       # [m]

#     driver.symmetrize = True

#     return driver


def setup_basic_main_source(plasma_density=6.0e20, ramp_beta_mag=5.0):
    from abel.utilities.plasma_physics import beta_matched

    main = SourceBasic()
    main.bunch_length = 40.0e-06                                                  # [m], rms.
    main.num_particles = 10000                                               
    main.charge = -SI.e * 1.0e10                                                  # [C]

    # Energy parameters
    main.energy = 3.0e9                                                           # [eV]
    main.rel_energy_spread = 0.02                                                 # Relative rms energy spread

    # Emittances
    main.emit_nx, main.emit_ny = 15e-6, 0.1e-6                                    # [m rad]

    # Beta functions
    main.beta_x = beta_matched(plasma_density, main.energy) * ramp_beta_mag       # [m]
    main.beta_y = main.beta_x                                                     # [m]

    # Offsets
    main.z_offset = 0.00e-6                                                       # [m]

    # Other
    main.symmetrize_6d = True

    return main


@pytest.mark.rft_api_unit_test
def test_abel_beam2rft_beam():
    """
    Tests for ``abel_beam2rft_beam()``.
    """

    from abel.apis.rf_track.rf_track_api import abel_beam2rft_beam

    source = setup_basic_main_source()
    beam = source.track()
    beam_rft = abel_beam2rft_beam(beam)


    # ========== Single beam, homogeneous charge ==========
    # Get the RF-Track beam phase space using get_phase_space() from RF-Track (see the RF-Track reference manual for updated reference) 
    # get_phase_space('%X %Px %Y %Py %Z %Pz %m %Q %N') returns the phase where the columns are
    #   X : [mm], column vector of the horizontal coordinates.
    #   Px : [MeV/c], column vector of the horizontal momenta.
    #   Y : [mm], column vector of the vertical coordinates.
    #   Py : [MeV/c], column vector of the vertical momenta.
    #   Z : [mm], column vector of the longitudinal coordinates.
    #   Pz : [MeV/c], column vector of the longitudinal momenta.
    #   m : [MeV/c^2], column vector of masses.
    #   Q : [e], column vector of single-particle charges.
    #   N : column vector of numbers of single particles per macro particle.

    phase_space_rft = beam_rft.get_phase_space('%X %Px %Y %Py %Z %Pz %m %Q %N') 

    assert np.allclose(phase_space_rft[:,0], beam.xs()*1e3, rtol=1e-15, atol=0.0)
    assert np.allclose(phase_space_rft[:,1], beam.pxs()*SI.c/SI.e/1e6, rtol=1e-15, atol=0.0)
    assert np.allclose(phase_space_rft[:,2], beam.ys()*1e3, rtol=1e-15, atol=0.0)
    assert np.allclose(phase_space_rft[:,3], beam.pys()*SI.c/SI.e/1e6, rtol=1e-15, atol=0.0)
    assert np.allclose(phase_space_rft[:,4], beam.zs()*1e3, rtol=1e-15, atol=0.0)
    assert np.allclose(phase_space_rft[:,5], beam.pzs()*SI.c/SI.e/1e6, rtol=1e-15, atol=0.0)
    assert np.all(phase_space_rft[:,6] == phase_space_rft[:,6][0])
    assert np.allclose(phase_space_rft[:,6][0], beam.particle_mass*SI.c**2/SI.e/1e6 , rtol=1e-15, atol=0.0)
    assert np.all(phase_space_rft[:,7] == phase_space_rft[:,7][0])
    assert np.allclose(phase_space_rft[:,7], beam.qs()/beam.weightings()/SI.e , rtol=1e-15, atol=0.0)
    assert np.all(phase_space_rft[:,8] == phase_space_rft[:,8][0])
    assert np.allclose(phase_space_rft[:,8], beam.weightings() , rtol=1e-15, atol=0.0)


    # ========== Inhomogeneous beam chaarge with chargeless ghost particles ==========
    x_min = beam.xs().min() + np.sign(beam.xs().min())*1e-6
    x_max = beam.xs().max() + np.sign(beam.xs().max())*1e-6
    y_min = beam.ys().min() + np.sign(beam.ys().min())*1e-6
    y_max = beam.ys().max() + np.sign(beam.ys().max())*1e-6
    z_end = beam.zs().min()
    z_start = beam.zs().max()
    
    X, Y, Z = np.meshgrid([x_min, x_max], [y_min, y_max], [z_start, z_end], indexing='ij')
    empty_beam = Beam()
    empty_beam.set_phase_space(Q=0,
                               xs=X.flatten(),
                               ys=Y.flatten(),
                               zs=Z.flatten(), 
                               uxs=np.ones_like(X.flatten())*beam.uxs()[0],
                               uys=np.ones_like(X.flatten())*beam.uys()[0],
                               uzs=np.ones_like(X.flatten())*beam.uzs()[0],
                               particle_mass=beam.particle_mass)
    
    assert len(empty_beam) == 8

    comb_beam = beam + empty_beam
    comb_beam.particle_mass = beam.particle_mass

    qs_abel = comb_beam.qs()
    weightings_abel = comb_beam.weightings()
    zero_mask = qs_abel == 0
    if sum(zero_mask) != 0:
        weightings_abel[zero_mask] = 1.0
    
    comb_beam_rft = abel_beam2rft_beam(comb_beam)
    phase_space_rft = comb_beam_rft.get_phase_space('%X %Px %Y %Py %Z %Pz %m %Q %N')

    assert np.allclose(phase_space_rft[:,0], comb_beam.xs()*1e3, rtol=1e-15, atol=0.0)
    assert np.allclose(phase_space_rft[:,1], comb_beam.pxs()*SI.c/SI.e/1e6, rtol=1e-15, atol=0.0)
    assert np.allclose(phase_space_rft[:,2], comb_beam.ys()*1e3, rtol=1e-15, atol=0.0)
    assert np.allclose(phase_space_rft[:,3], comb_beam.pys()*SI.c/SI.e/1e6, rtol=1e-15, atol=0.0)
    assert np.allclose(phase_space_rft[:,4], comb_beam.zs()*1e3, rtol=1e-15, atol=0.0)
    assert np.allclose(phase_space_rft[:,5], comb_beam.pzs()*SI.c/SI.e/1e6, rtol=1e-15, atol=0.0)
    
    assert phase_space_rft.shape[0] == len(empty_beam) + len(beam)
    assert np.all(phase_space_rft[:,6] == phase_space_rft[:,6][0])
    assert np.allclose(phase_space_rft[:,6][0], comb_beam.particle_mass*SI.c**2/SI.e/1e6 , rtol=1e-15, atol=0.0)
    assert not np.all(phase_space_rft[:,7] == phase_space_rft[:,7][0])
    assert np.allclose(phase_space_rft[:,7], comb_beam.qs()/weightings_abel/SI.e , rtol=1e-15, atol=0.0)
    assert np.allclose(phase_space_rft[:,8], weightings_abel , rtol=1e-15, atol=0.0)


@pytest.mark.rft_api_unit_test
def test_rft_beam2abel_beam():
    """
    Tests for ``rft_beam2abel_beam()``.
    """

    from RF_Track import Bunch6dT
    from abel.apis.rf_track.rf_track_api import rft_beam2abel_beam

    source = setup_basic_main_source()
    beam = source.track()
    xs_abel = beam.xs()    # [m]
    pxs_abel = beam.pxs()  # [kg m/s]
    ys_abel = beam.ys()
    pys_abel = beam.pys()
    zs_abel = beam.zs()
    pzs_abel = beam.pzs()
    qs_abel = beam.qs()
    weightings_abel = beam.weightings()
    particle_mass = beam.particle_mass*SI.c**2/SI.e/1e6  # [MeV/c^2]

    phase_space_rft = np.column_stack((xs_abel*1e3, pxs_abel*SI.c/SI.e/1e6, 
                                        ys_abel*1e3, pys_abel*SI.c/SI.e/1e6, 
                                        zs_abel*1e3, pzs_abel*SI.c/SI.e/1e6))

    single_particle_charge = qs_abel[0]/SI.e/weightings_abel[0]  # Charge of a single physical particle [e].
    beam_rft = Bunch6dT(particle_mass, beam.population(), single_particle_charge, phase_space_rft)

    converted_beam = rft_beam2abel_beam(beam_rft)
    Beam.comp_beams(converted_beam, beam)

    

@pytest.mark.rft_api_unit_test
def test_sc_fields_obj_and_beam_fields():
    """
    Tests for ``calc_sc_fields_obj()`` and ``rft_beam_fields()``.
    """
    
    from abel.apis.rf_track.rf_track_api import rft_beam_fields, calc_sc_fields_obj
    np.random.seed(42)
    
    # Parameters
    N_electrons = 1e6                                                               # Number of particles in the distribution
    Q = -1 * SI.e                                                                   # [C], charge of one electron
    N = 1000000                                                                     # Number of simulated macro particles

    radius = 1e-3                                                                   # [m], radius of the sphere
    rvals = 2 * np.random.rand(N) - 1
    elevation = np.arcsin(rvals)                                                    # [rad]
    azimuth = 2 * np.pi * np.random.rand(N)                                         # [rad]
    radii = radius * np.random.rand(N) ** (1/3)                                     # [m]

    # Convert spherical to Cartesian coordinates
    xs = radii * np.cos(elevation) * np.cos(azimuth)
    ys = radii * np.cos(elevation) * np.sin(azimuth)
    zs = radii * np.sin(elevation)
    
    # Create a beam
    O = np.zeros(N)
    beam = Beam()
    beam.allow_low_energy_particles = True
    beam.set_phase_space(Q=Q*N_electrons, xs=xs, ys=ys, zs=zs, uxs=O, uys=O, uzs=O, particle_mass=SI.m_e)

    # Assemble a RFT SpaceCharge_Field object
    sc_fields_obj = calc_sc_fields_obj(beam, num_x_cells=50, num_y_cells=50, num_z_cells=50, num_t_bins=1)

    # Evaluate the electric field in 1D
    x_lin = np.linspace(xs.min(), xs.max(), 100)                                    # [m]
    O = np.zeros(100)
    E_1d, B = sc_fields_obj.get_field(x_lin*1e3, O, O, O)

    # Compute the analytic solution
    Er = []
    rmax = radius                                                                   # [m]
    rs = np.linspace(-rmax, rmax, 100)                                              # [m]
    for r in rs:
        if r < radius:
            E = 1/(4*SI.pi*SI.epsilon_0) * Q * N_electrons * r / radius**3          # [V/m]
        else:
            E = 1/(4*SI.pi*SI.epsilon_0) * Q * N_electrons / r**2                   # [V/m]
        Er.append(E)
    Er = np.array(Er)

    # Compare the two arrays
    assert np.allclose(E_1d[:, 0], Er, rtol=0.0, atol=70.0)
    assert np.allclose(B[:, 0], 0.0, rtol=1e-15, atol=0.0)


    # Compute the fields at the location of the macroparticles
    E_fields_beam, B_fields_beam, xs_sorted, ys_sorted, zs_sorted = rft_beam_fields(beam, num_x_cells=50, num_y_cells=50, 
                                                                                    num_z_cells=50, num_t_bins=1, sort_zs=False)

    Exs = E_fields_beam[:,0]
    Eys = E_fields_beam[:,1]
    Ezs = E_fields_beam[:,2]

    Es = np.sqrt(Exs**2 + Eys**2 + Ezs**2)  # Field magnitude

    # Compute fields for comparison
    rs_sorted = np.sqrt(xs_sorted**2 + ys_sorted**2 + zs_sorted**2)
    Er_3d = []
    for r in rs_sorted:
        if r < radius:
            E = 1/(4*SI.pi*SI.epsilon_0) * Q * N_electrons * r / radius**3          # [V/m]
        else:
            E = 1/(4*SI.pi*SI.epsilon_0) * Q * N_electrons / r**2                   # [V/m]
        Er_3d.append(E)
    Er_3d = np.abs(np.array(Er_3d))

    # Compare the two arrays
    assert np.allclose(Es, Er_3d, rtol=0.0, atol=80.0)
    assert np.allclose(B_fields_beam[:, 0], 0.0, rtol=1e-15, atol=0.0)
    assert np.allclose(B_fields_beam[:, 1], 0.0, rtol=1e-15, atol=0.0)
    assert np.allclose(B_fields_beam[:, 2], 0.0, rtol=1e-15, atol=0.0)