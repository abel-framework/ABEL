# -*- coding: utf-8 -*-
# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

"""
ABEL : unit tests for the ELEGANT wrapper
"""

import pytest, os
import numpy as np
import scipy.constants as SI
from abel.classes.source.impl.source_basic import SourceBasic
#from abel.classes.beam import Beam
from abel.wrappers.elegant.elegant_wrapper import *
#from abel.classes.interstage.impl.interstage_elegant import InterstageElegant
from string import Template


def elegant_missing():
    """
    Checks if the elegant executable exists and has executable permissions.
    Returns True if missing.

    Parameters
    ----------
    file_path: str
      The path to the file.

    Returns
    -------
    bool: False if the file exists and is executable, True otherwise.
    """
    file_path = CONFIG.elegant_exec
    if os.path.exists(file_path) and os.path.isfile(file_path):
        if os.access(file_path, os.X_OK):
            return False
    return True


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

#     runfile_template = os.path.join(os.path.dirname(abel.wrappers.elegant.elegant_wrapper.__file__), 'templates', 'runscript_interstage.ele')
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
            

@pytest.mark.skipif(elegant_missing(), reason="ELEGANT is not available")
@pytest.mark.elegant_wrapper_unit_test
def test_elegant_write_read_beam():
    """
    Test of ``elegant_write_beam()`` and ``elegant_read_beam()``.
    """

    # Create a temporary folder
    parent_dir = '.' + os.sep + 'tests' + os.sep + 'run_data' + os.sep + 'temp' + os.sep
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    tmpfolder = os.path.join(parent_dir, str(uuid.uuid4())) + os.sep
    os.mkdir(tmpfolder)

    # Set up the paths for the temporary ELEGANT beam files
    inputbeamfile = tmpfolder + 'input_beam.bun'

    # Create a beam
    source = setup_basic_main_source()
    beam0 = source.track()

    # Write beam into an ELEGANT SDDS beam file
    elegant_write_beam(beam0, inputbeamfile, tmpfolder=tmpfolder)

    # Convert back to an ABEL beam
    beam = elegant_read_beam(inputbeamfile, tmpfolder=tmpfolder)
    beam.set_zs(-1*beam.zs()) # TODO: Seems like there is an inconsistency in the definition of z in the ELEGANT wrapper. is this really correct? Does the ELEGANT beam perhaps need to be passed through an ELEGANT tracking to have the zs set correctly?

    assert np.isclose(beam0.location, beam.location, rtol=0.0, atol=1.0e-10)
    assert beam0.stage_number == beam.stage_number
    Beam.comp_beams(beam0, beam, comp_location=False)
    
    # Remove temporary files
    shutil.rmtree(tmpfolder)

# @pytest.mark.skipif(elegant_missing(), reason="ELEGANT is not available")
# @pytest.mark.elegant_wrapper_unit_test
# def test_elegant_run():
#     """
#     Test of ``elegant_run()``.
#     """

#     # make temporary folder and files
#     #parent_dir = '.' + os.sep + 'tests' + os.sep + 'run_data' + os.sep + 'temp' + os.sep
#     #parent_dir = os.path.join(os.path.dirname(abel.wrappers.elegant.elegant_wrapper.__file__), 'templates', 'lattice_interstage.lte')

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


@pytest.mark.skipif(elegant_missing(), reason="ELEGANT is not available")
@pytest.mark.elegant_wrapper_unit_test
def test_elegant_apl_fieldmap2D():
    """
    Test of ``elegant_apl_fieldmap2D()``.
    """

    from abel.utilities.other import find_closest_value_in_arr

    # Lens parameters
    tau_lens = 100                                                                  
    lensdim_x = 5e-3                                                                # [m]
    lensdim_y = 1e-3                                                                # [m]
    lens_x_offset = 0.0                                                             # [m]
    lens_y_offset = 0.0                                                             # [m]

    # Create the temporary folder
    parent_dir = '.' + os.sep + 'tests' + os.sep + 'run_data' + os.sep + 'temp' + os.sep
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    tmpfolder = os.path.join(parent_dir, str(uuid.uuid4())) + os.sep
    os.mkdir(tmpfolder)
    
    tmpfile =  tmpfolder + os.sep + 'stream_' + str(uuid.uuid4()) + '.tmp'

    sdds_file_name = elegant_apl_fieldmap2D(tau_lens=tau_lens,
                                            lensdim_x=lensdim_x, lensdim_y=lensdim_y, 
                                            lens_x_offset=lens_x_offset, lens_y_offset=lens_y_offset, 
                                            tmpfolder=tmpfolder)
    print(sdds_file_name)
    # Convert the output SDDS file to a CSV file for later comparison
    subprocess.run(CONFIG.elegant_exec + 'sdds2stream ' + sdds_file_name + ' -columns=x,y,Bx,By > ' + tmpfile, shell=True)

    # Load feildmap from CSV file
    fieldmap = np.loadtxt(open(tmpfile, "rb"), delimiter=' ')
    xs_test = fieldmap[:,0]
    ys_test = fieldmap[:,1]
    Bxs_test = fieldmap[:,2]
    Bys_test = fieldmap[:,3]

    # Create a fieldmap for comparison
    xs = np.linspace(-lensdim_x, lensdim_x, 501)
    ys = np.linspace(-lensdim_y, lensdim_y, 501)

    X, Y = np.meshgrid(xs, ys, indexing="xy")  # Shape: (len(ys), len(xs))
    Xo = X + lens_x_offset
    Yo = Y + lens_y_offset

    Bx = Yo + Xo * Yo * tau_lens
    By = -(Xo + ((Xo**2 + Yo**2) / 2) * tau_lens)

    xs_grid = X.ravel()
    ys_grid = Y.ravel()
    Bxs = Bx.ravel()
    Bys = By.ravel()
    
    # Compare the fieldmaps
    assert np.isclose(xs_test.min(), xs.min(), rtol=1e-5, atol=0.0)
    assert np.isclose(xs_test.max(), xs.max(), rtol=1e-5, atol=0.0)
    assert np.isclose(ys_test.min(), ys.min(), rtol=1e-5, atol=0.0)
    assert np.isclose(ys_test.max(), ys.max(), rtol=1e-5, atol=0.0)

    assert np.isclose(Bxs_test.min(), Bxs.min(), rtol=1e-5, atol=0.0)
    assert np.isclose(Bxs_test.max(), Bxs.max(), rtol=1e-5, atol=0.0)
    assert np.isclose(Bys_test.min(), Bys.min(), rtol=1e-5, atol=0.0)
    assert np.isclose(Bys_test.max(), Bys.max(), rtol=1e-5, atol=0.0)

    x0_idx, _ = find_closest_value_in_arr(Bxs, 0.0)  # Index for Bxs that is closest to Bxs=0.0.
    x0_idx_test, _ = find_closest_value_in_arr(Bxs_test, 0.0)
    assert np.isclose(Bxs_test[x0_idx_test], Bxs[x0_idx], rtol=1e-5, atol=0.0)
    assert np.isclose(xs_test[x0_idx_test], xs_grid[x0_idx], rtol=1e-5, atol=0.0)

    y0_idx, _ = find_closest_value_in_arr(Bys, 0.0)  # Index for Bys that is closest to Bys=0.0.
    y0_idx_test, _ = find_closest_value_in_arr(Bys_test, 0.0)
    assert np.isclose(Bys_test[y0_idx_test], Bys[y0_idx], rtol=1e-5, atol=0.0)
    assert np.isclose(ys_test[y0_idx_test], ys_grid[y0_idx], rtol=1e-5, atol=0.0)

    if len(Bxs) == len(Bxs_test):
        assert np.allclose(xs_test, xs_grid, rtol=1e-5, atol=0.0)
        assert np.allclose(ys_test, ys_grid, rtol=1e-5, atol=0.0)
        assert np.allclose(Bxs_test, Bxs, rtol=1e-5, atol=0.0)
        assert np.allclose(Bys_test, Bys, rtol=1e-5, atol=0.0)

    # Remove temporary files
    shutil.rmtree(tmpfolder)
