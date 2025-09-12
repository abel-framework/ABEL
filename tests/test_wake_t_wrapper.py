# -*- coding: utf-8 -*-
"""
ABEL : unit tests for the Wake-T wrapper
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

import pytest, os, uuid, shutil
import numpy as np
import scipy.constants as SI
from abel.classes.beam import Beam
from abel.classes.source.impl.source_basic import SourceBasic
from abel.wrappers.wake_t.wake_t_wrapper import *


def setup_basic_source(plasma_density=6.0e20, ramp_beta_mag=5.0):
    from abel.utilities.plasma_physics import beta_matched

    source = SourceBasic()
    source.bunch_length = 40.0e-06                                                  # [m], rms.
    source.num_particles = 10000                                               
    source.charge = -SI.e * 1.0e10                                                  # [C]

    # Energy parameters
    source.energy = 3.0e9                                                           # [eV]
    source.rel_energy_spread = 0.02                                                 # Relative rms energy spread

    # Emittances
    source.emit_nx, source.emit_ny = 15e-6, 0.1e-6                                    # [m rad]

    # Beta functions
    source.beta_x = beta_matched(plasma_density, source.energy) * ramp_beta_mag       # [m]
    source.beta_y = source.beta_x                                                     # [m]

    # Offsets
    source.z_offset = 0.00e-6                                                       # [m]

    # Other
    source.symmetrize_6d = True

    return source


@pytest.mark.wake_t_wrapper_unit_test
def test_beam2wake_t_bunch():
    """
    Test for ``beam2wake_t_bunch()``.
    """

    source = setup_basic_source()
    beam = source.track()

    # Convert to a Wake-T particle bunch using beam2wake_t_bunch() 
    wake_t_bunch = beam2wake_t_bunch(beam)

    # Extract the phase space
    phasespace = wake_t_bunch.get_6D_matrix_with_charge()

    # Compare the ABEL phase space and Wake-T phase space
    assert np.allclose(len(beam.qs()), len(phasespace[6]), rtol=1e-15, atol=0.0)
    assert np.allclose(beam.qs(), phasespace[6], rtol=1e-15, atol=0.0)
    assert np.allclose(beam.xs(), phasespace[0], rtol=1e-15, atol=0.0)
    assert np.allclose(beam.ys(), phasespace[2], rtol=1e-15, atol=0.0)
    assert np.allclose(beam.zs(), phasespace[4], rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pxs(), phasespace[1]*SI.c*SI.m_e, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pys(), phasespace[3]*SI.c*SI.m_e, rtol=1e-15, atol=0.0)
    assert np.allclose(beam.pzs(), phasespace[5]*SI.c*SI.m_e, rtol=1e-15, atol=0.0)
    assert np.isclose(beam.location, wake_t_bunch.prop_distance, rtol=1e-15, atol=0.0)


@pytest.mark.wake_t_wrapper_unit_test
def test_wake_t_bunch2beam():
    """
    Test for ``wake_t_bunch2beam()``.
    """

    source = setup_basic_source()
    beam = source.track()

    # Convert to a Wake-T particle bunch using beam2wake_t_bunch() 
    wake_t_bunch = beam2wake_t_bunch(beam)

    # Convert back to an ABEL beam
    beam_test = wake_t_bunch2beam(wake_t_bunch)

    # Compare the beam to the original
    Beam.comp_beams(beam_test, beam, comp_location=True, rtol=1e-12, atol=0.0)


@pytest.mark.wake_t_wrapper_unit_test
def test_plasma_stage_setup():
    """
    Test for ``plasma_stage_setup()``.
    """

    import wake_t
    from abel.utilities.plasma_physics import k_p
    
    driver_source = setup_basic_source()
    driver_source.charge = -SI.e * 5.0e10 
    driver_source.z_offset = 1000e-6
    drive_beam = driver_source.track()

    source = setup_basic_source()
    beam = source.track()

    # ========== Single beam, default parameters only ==========
    plasma_density = 6.0e20

    # Set up a Wake-T PlasmaStage object
    plasma = plasma_stage_setup(plasma_density=plasma_density, abel_drive_beam=drive_beam, abel_main_beam=None, stage_length=None, dz_fields=None, 
                                num_cell_xy=256, n_out=1, box_size_r=None, box_min_z=None, box_max_z=None)
    
    k_beta = k_p(plasma_density)/np.sqrt(2*drive_beam.gamma())
    lambda_betatron = 2*np.pi/k_beta
    length = 0.05*lambda_betatron
    
    assert np.isclose(plasma.density(0), plasma_density, rtol=1e-15, atol=0.0)
    assert isinstance(plasma.fields[0], wake_t.physics_models.plasma_wakefields.qs_rz_baxevanis.wakefield.Quasistatic2DWakefield)
    assert np.isclose(plasma.length, length, rtol=1e-15, atol=0.0)
    assert plasma.n_out == 1
    assert isinstance(plasma.wakefield, wake_t.physics_models.plasma_wakefields.qs_rz_baxevanis.wakefield.Quasistatic2DWakefield)


    # ========== Drive beam and main beam, default parameters only ==========
    plasma_density = 6.0e20

    plasma2 = plasma_stage_setup(plasma_density=plasma_density, abel_drive_beam=drive_beam, abel_main_beam=beam, stage_length=None, dz_fields=None, 
                                num_cell_xy=256, n_out=1, box_size_r=None, box_min_z=None, box_max_z=None)
    
    k_beta2 = k_p(plasma_density)/np.sqrt(2*min(beam.gamma(), drive_beam.gamma()/2))
    lambda_betatron2 = 2*np.pi/k_beta2
    length2 = 0.05*lambda_betatron2

    assert np.isclose(plasma2.density(0), plasma_density, rtol=1e-15, atol=0.0)
    assert isinstance(plasma2.fields[0], wake_t.physics_models.plasma_wakefields.qs_rz_baxevanis.wakefield.Quasistatic2DWakefield)
    assert np.isclose(plasma2.length, length2, rtol=1e-15, atol=0.0)
    assert plasma2.n_out == 1
    assert isinstance(plasma2.wakefield, wake_t.physics_models.plasma_wakefields.qs_rz_baxevanis.wakefield.Quasistatic2DWakefield)


    # ========== Drive beam and main beam==========
    plasma_density3 = 5.0e20
    length3 = 1e-3

    plasma3 = plasma_stage_setup(plasma_density=plasma_density3, abel_drive_beam=drive_beam, abel_main_beam=beam, stage_length=length3, dz_fields=None, 
                                num_cell_xy=256, n_out=1, box_size_r=None, box_min_z=None, box_max_z=None)

    assert np.isclose(plasma3.density(0), plasma_density3, rtol=1e-15, atol=0.0)
    assert isinstance(plasma3.fields[0], wake_t.physics_models.plasma_wakefields.qs_rz_baxevanis.wakefield.Quasistatic2DWakefield)
    assert np.isclose(plasma3.length, length3, rtol=1e-15, atol=0.0)
    assert plasma3.n_out == 1
    assert isinstance(plasma3.wakefield, wake_t.physics_models.plasma_wakefields.qs_rz_baxevanis.wakefield.Quasistatic2DWakefield)


@pytest.mark.wake_t_wrapper_unit_test
def test_wake_t_hdf5_load():
    """
    Test for ``wake_t_hdf5_load()``.
    """

    # Create a temporary folder
    parent_dir = '.' + os.sep + 'tests' + os.sep + 'run_data' + os.sep + 'temp' + os.sep
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    tmpfolder = os.path.join(parent_dir, str(uuid.uuid4())) + os.sep
    os.mkdir(tmpfolder)

    # Set up a Wake-T PlasmaStage object
    source = setup_basic_source()
    beam = source.track()

    plasma = plasma_stage_setup(plasma_density=6.0e20, abel_drive_beam=beam, abel_main_beam=None, stage_length=None, dz_fields=None, 
                                num_cell_xy=256, n_out=1, box_size_r=None, box_min_z=None, box_max_z=None)
    
    # Perform tracking
    bunch_list = plasma.track(beam2wake_t_bunch(beam, name='beam'), opmd_diag=True, diag_dir=tmpfolder)

    # Also retrieve the initial beam from plasma.track() output
    bunch = bunch_list[0]
    beam_init = wake_t_bunch2beam(bunch)
    #Beam.comp_beams(beam_init, beam, comp_location=True, rtol=1e-5, atol=0.0)

    # Extract the initial beam from the hdf5 file
    data_dir = tmpfolder + 'hdf5' + os.sep
    files = sorted(os.listdir(data_dir))
    file_path = data_dir + files[0]
    beam_test = wake_t_hdf5_load(file_path=file_path, species='beam')

    # Compare the beam to the original
    Beam.comp_beam_params(beam_test, beam, comp_location=False)  # The beam in file_path may not be exactly the same as the original beam.
    Beam.comp_beams(beam_test, beam_init, comp_location=False, rtol=1e-12, atol=0.0)

    # Remove temporary files
    shutil.rmtree(tmpfolder)