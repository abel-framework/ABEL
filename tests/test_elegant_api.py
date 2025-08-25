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
#from abel.classes.interstage.impl.interstage_elegant import InterstageElegant
from string import Template


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


# def make_run_script(nom_energy, latticefile, inputbeamfile, tmpfolder=None):
        
#     # create temporary CSV file and folder
#     make_new_tmpfolder = tmpfolder is None
#     if make_new_tmpfolder:
#         tmpfolder = CONFIG.temp_path + str(uuid.uuid4())
#         os.mkdir(tmpfolder)
#     tmpfile = tmpfolder + 'runfile.ele'

#     # inputs
#     path_beam_centroid_file = tmpfolder + 'centroid_vs_s.cen'  # Used to output a file in tmpfolder containing data of the beam centroids as a function of s.
#     path_twiss_parameter_file =  tmpfolder + 'twiss_vs_s.twi'  # Used to output a file in tmpfolder containing data of the beam centroids as a function of s.
#     inputs = {'p_central_mev': nom_energy/1e6,
#                 'latticefile': latticefile,
#                 'inputbeamfile': inputbeamfile,
#                 'path_to_beam_centroid_file': path_beam_centroid_file,
#                 'path_to_uncoupled_Twiss_parameter_output_file': path_twiss_parameter_file}

#     runfile_template = os.path.join(os.path.dirname(abel.apis.elegant.elegant_api.__file__), 'templates', 'runscript_interstage.ele')
#     with open(runfile_template, 'r') as fin, open(tmpfile, 'w') as fout:
#         results = Template(fin.read()).substitute(inputs)
#         fout.write(results)

#     return tmpfile


# def make_lattice(beam, outputbeamfile, latticefile, evolutionfolder, tmpfolder=None):
        
#         # Make lattice file from template
#         lattice_template = './tests/data/elegant_templates/lattice_interstage.lte'

#         watch_disabled = False
            
#         inputs = {'charge': abs(beam.charge()),
#                   'spacer_length': 0.0,
#                   'output_filename': outputbeamfile,
#                   'watch_filename': evolutionfolder + 'output_%03ld.bun',
#                   'watch_disabled': int(watch_disabled)}
        
#         with open(lattice_template, 'r') as fin, open(latticefile, 'w') as fout:
#             fin.seek(0)  # make sure we're at the beginning

#             content = fin.read()
#             template = Template(content)
#             results = template.substitute(inputs)  # Substitute placeholders in template with inputs.

#             #print("RESULTS repr:", repr(results))#TODO: delete

#             fout.write(results)
            

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

    # Create a beam
    source = setup_basic_main_source()
    beam0 = source.track()

    # Write beam into an ELEGANT SDDS beam file
    elegant_write_beam(beam0, inputbeamfile, tmpfolder=tmpfolder)

    # Convert back to an ABEL beam
    beam = elegant_read_beam(inputbeamfile, tmpfolder=tmpfolder)
    beam.set_zs(-1*beam.zs()) # TODO: Seems like there is an inconsistency in the definition of z in the ELEGANT api. is this really correct?

    Beam.comp_beams(beam0, beam)

    if tmpfolder is not None:
        # Remove temporary files
        shutil.rmtree(tmpfolder)


# @pytest.mark.elegant_api_unit_test
# def test_elegant_run():
#     """
#     Test of ``elegant_run()``.
#     """

#     # make temporary folder and files
#     #parent_dir = '.' + os.sep + 'tests' + os.sep + 'run_data' + os.sep + 'temp' + os.sep
#     #parent_dir = os.path.join(os.path.dirname(abel.apis.elegant.elegant_api.__file__), 'templates', 'lattice_interstage.lte')

#     if not os.path.exists(parent_dir):
#         os.makedirs(parent_dir)
    
#     # create the temporary folder
#     tmpfolder = os.path.join(parent_dir, str(uuid.uuid4())) + os.sep
#     os.mkdir(tmpfolder)
#     inputbeamfile = tmpfolder + 'input_beam.bun'
#     outputbeamfile = tmpfolder + 'output_beam.bun'
#     latticefile = tmpfolder + 'interstage.lte'
#     evolution_folder = tmpfolder + 'evolution' + os.sep
#     os.mkdir(evolution_folder)

#      # Create a beam
#     source = setup_basic_main_source()
#     beam0 = source.track()
    
#     # Make lattice file
#     make_lattice(beam0, outputbeamfile, latticefile, evolution_folder, save_evolution=False, tmpfolder=tmpfolder)

#     # Make runfile
#     runfile = make_run_script(beam0.energy(), latticefile, inputbeamfile, tmpfolder)

#     # Run ELEGANT
#     beam = elegant_run(runfile, beam0, inputbeamfile, outputbeamfile, quiet=True, tmpfolder=tmpfolder) #This is not executed correctly yet...

