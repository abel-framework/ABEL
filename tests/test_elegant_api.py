# -*- coding: utf-8 -*-
"""
ABEL : unit tests for the ELEGANT API
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

import pytest, os
import numpy as np
import scipy.constants as SI
from abel.classes.source.impl.source_basic import SourceBasic
#from abel.classes.beam import Beam
from abel.apis.elegant.elegant_api import *


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


@pytest.mark.elegant_api_unit_test
def test_elegant_write_read_beam():
    """
    Test of ``elegant_write_beam()`` and ``elegant_read_beam()``.
    """

    # Create the temporary folder
    parent_dir = '.' + os.sep + 'tests' + os.sep + 'run_data' + os.sep + 'temp' + os.sep
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    tmpfolder = os.path.join(parent_dir, str(uuid.uuid4())) + os.sep
    os.mkdir(tmpfolder)

    # Set up the parths for the temporary ELEGANT beam files
    inputbeamfile = tmpfolder + 'input_beam.bun'
    outputbeamfile = tmpfolder + 'output_beam.bun'

    # Create a beam
    source = setup_basic_main_source()
    beam0 = source.track()

    # Write beam into an ELEGANT SDDS beam file
    elegant_write_beam(beam0, inputbeamfile, tmpfolder=tmpfolder)

    # Convert back to an ABEL beam
    beam = elegant_read_beam(inputbeamfile, tmpfolder=tmpfolder)
    beam.set_zs(-1*beam.zs()) # TODO: is this really correct?

    Beam.comp_beams(beam0, beam)

