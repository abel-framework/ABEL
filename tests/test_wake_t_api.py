# -*- coding: utf-8 -*-
"""
ABEL : unit tests for the Wake-T API
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
from abel.apis.wake_t.wake_t_api import *


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


@pytest.mark.wake_t_api_unit_test
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


@pytest.mark.wake_t_api_unit_test
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


@pytest.mark.wake_t_api_unit_test
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

    # Set up a Wake-T tracking
    source = setup_basic_source()
    beam = source.track()

    plasma = plasma_stage_setup(plasma_density=6.0e20, abel_drive_beam=beam, abel_main_beam=None, stage_length=None, dz_fields=None, 
                                num_cell_xy=256, n_out=1, box_size_r=None, box_min_z=None, box_max_z=None)
    
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