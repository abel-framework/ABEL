# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

import os, subprocess, time, sys
import numpy as np
import scipy.constants as SI
from string import Template
from pathlib import Path
from abel.CONFIG import CONFIG
from abel.classes.beam import Beam
from tqdm import tqdm
from abel.utilities.plasma_physics import k_p

# write the HiPACE++ input script to file
def hipace_write_inputs(filename_input, filename_beam, filename_driver, plasma_density, num_steps, time_step, box_range_z, box_size_xy, output_period=None, ion_motion=True, ion_species='H', radiation_reaction=False, beam_ionization=True, num_cell_xy=511, num_cell_z=512, driver_only=False, density_table_file=None, no_plasma=False, external_focusing_gradient=0, mesh_refinement=False, do_spin_tracking=False):
    """
    Write a HiPACE++ input script to file based on a provided template and 
    simulation parameters.
      
    It sets the longitudinal grid resolution, adjusts transverse grid resolution 
    for optimal performance (rounding the number of transfer cells 
    ``num_cell_xy`` to be 2^n − 1), configures the plasma, and optionally 
    enables mesh refinement and various physics effects. The main beam and drive 
    beam used in the simulation are extracted from ABEL ``Beam`` files. 
    
    The provided parameters and configurations are substituted into a template 
    file to produce the final HiPACE++ input file.
    

    Parameters
    ----------
    filename_input : str
        Output file path for the generated HiPACE++ input file.

    filename_beam : str
        Path to the HDF5 file containing the main beam (``Beam``).  

    filename_driver : str
        Path to the HDF5 file containing the drive beam (``Beam``). 

    plasma_density : [m^-3] float
        Plasma density.

    num_steps : int
        Maximum number of time steps to simulate.

    time_step : [s] float
        Time step duration used in the simulation.

    box_range_z : [m] float list
        Longitudinal simulation domain range ``[box_min_z, box_max_z]``.

    box_size_xy : [m] float
        Transverse box half-size in x and y that defines grid transverse extents.

    output_period : int, optional
        Interval (in number of time steps) for output diagnostics. Defaults to 
        ``num_steps`` if ``None``.

    ion_motion : bool, optional
        Flag for enabling ion motion in the plasma model. Defaults to ``True``.

    ion_species : str, optional
        Ion species name (e.g., ``'H'``, ``'He'``). Defaults to ``'H'``.

    radiation_reaction : bool, optional
        Flag for enabling radiation reaction. Defaults to ``False``.

    beam_ionization : bool, optional
        Flag for enabling beam ionization. Defaults to ``True``.

    num_cell_xy : int, optional
        Number of transverse grid cells. Automatically adjusted to 2^n - 1 if 
        necessary. Defaults to ``511``.

    num_cell_z : int, optional
        Number of longitudinal grid cells. Defauls to ``512``.

    driver_only : bool, optional
        If ``True``, only the driver beam is simulated. Defaults to ``False``.

    density_table_file : str, optional
        Path to a tabulated plasma density profile file. If ``None``, a uniform 
        plasma is assumed. Defaults to ``None``.

    no_plasma : bool, optional
        Disable plasma entirely (useful for vacuum beam transport). Defaults to ``False``.

    external_focusing_gradient : [T/m] float, optional
        Field gradient for an external magnetic field applied traversely across 
        the plasma. A value > 0 enables external focusing. Defaults to ``0``.

    mesh_refinement : bool, optional
        Enable mesh refinement (1 refinement level). Defaults to ``False``.

    do_spin_tracking : bool, optional
        Enable spin tracking for polarized beams. Defaults to ``False``.

    Returns
    -------
    None
        The function writes the completed HiPACE++ input script to ``filename_input`` and
        does not return any value.

    Notes
    -----
    - The plasma components and beam components are automatically determined based on
      ``ion_motion`` and ``driver_only``.
    - For example, if ``no_plasma`` is ``True``, all plasma-related inputs are disabled.
    - See the HiPACE++ documentation [1]_ for supported keywords.

    References
    ----------
    .. [1] HiPACE++ User Guide, https://hipace.readthedocs.io/
    """

    if output_period is None:
        output_period = int(num_steps/2)
        
    # locate template file
    filename_input_template = os.path.join(os.path.dirname(__file__), 'input_template')

    if no_plasma:
        plasma_components = 'no_plasma'
    else:
        # prepare plasma components (based on ion motion)
        if ion_motion:
            plasma_components = 'electrons ions'
        else:
            plasma_components = 'plasma'

    # driver-only mode
    if driver_only:
        beam_components = 'driver'
    else:
        beam_components = 'driver beam'
    
    # plasma-density profile from file
    if density_table_file is not None:
        density_comment1 = '#'
        density_comment2 = ''
    else:
        density_comment1 = ''
        density_comment2 = '#'
        density_table_file = ''
    
    # check that the number of transverse cells is 2^n-1
    num_cell_xy = num_cell_xy
    closest_exponent_xy = round(np.log2(num_cell_xy+1))
    new_num_cell_xy = 2**closest_exponent_xy - 1
    if not num_cell_xy == new_num_cell_xy:
        print('>> HiPACE++: Changing from', num_cell_xy, 'to', new_num_cell_xy, ' (i.e., 2^n-1) for better performance.')
        num_cell_xy = new_num_cell_xy

    # plasma-density profile from file
    if abs(external_focusing_gradient) > 0:
        external_focusing_comment = ''
    else:
        external_focusing_comment = '#'

    # mesh refinement (level 1)
    res_mr0 = box_size_xy/num_cell_xy
    box_size_xy_mr1 = 0.9/k_p(plasma_density)
    ref_ratio_xy = 2**np.ceil(np.log2(box_size_xy/box_size_xy_mr1))
    num_cell_xy_mr1 = 2**np.round(np.log2((num_cell_xy+1)/2))-1
    
    if not mesh_refinement:
        mesh_refinement_maxlevel = 0
        mesh_refinement_comment = '#'
    else:
        mesh_refinement_maxlevel = 1
        mesh_refinement_comment = ''
    
    # define inputs
    inputs = {'num_cell_x': int(num_cell_xy), 
              'num_cell_y': int(num_cell_xy), 
              'num_cell_z': int(num_cell_z),
              'plasma_density': plasma_density,
              'grid_low_x': -box_size_xy/2,
              'grid_high_x': box_size_xy/2,
              'grid_low_y': -box_size_xy/2,
              'grid_high_y': box_size_xy/2,
              'grid_low_z': min(box_range_z),
              'grid_high_z': max(box_range_z),
              'mesh_refinement_maxlevel': int(mesh_refinement_maxlevel),
              'num_cell_x_mr1': int(num_cell_xy_mr1), 
              'num_cell_y_mr1': int(num_cell_xy_mr1),
              'ref_ratio_x': int(ref_ratio_xy), 
              'ref_ratio_y': int(ref_ratio_xy),
              'mesh_refinement_comment': mesh_refinement_comment,
              'time_step': time_step,
              'max_step': int(num_steps),
              'output_period': output_period,
              'radiation_reaction': int(radiation_reaction),
              'do_spin_tracking': int(do_spin_tracking),
              'beam_components': beam_components,
              'plasma_components': plasma_components,
              'density_table_file': density_table_file,
              'density_comment1': density_comment1,
              'density_comment2': density_comment2,
              'ion_species': ion_species,
              'beam_ionization': int(beam_ionization),
              'external_focusing': abs(external_focusing_gradient),
              'external_focusing_comment': external_focusing_comment,
              'filename_beam': filename_beam,
              'filename_driver': filename_driver}

    # fill in template file
    with open(filename_input_template, 'r') as fin, open(filename_input, 'w') as fout:
        results = Template(fin.read()).substitute(inputs)
        fout.write(results)


# ==================================================
# write the HiPACE++ job script to file
def hipace_write_jobscript(filename_job_script, filename_input, num_nodes=1, num_tasks_per_node=8):
    """
    Write a HiPACE++ batch job script (for a Slurm-based cluster) to file based 
    on a template and system configuration.

    This function generates a batch job script for running HiPACE++ on an HPC 
    system by substituting values from the ABEL configuration class  
    :class:`CONFIG <abel.CONFIG>` into a template job script (the current job 
    script template follows the Slurm batch job syntax used on the LUMI 
    supercomputer [1]_). The compute partition is automatically chosen based on 
    requested resources. 
    
    The resulting script is made executable.

    Parameters
    ----------
    filename_job_script : str
        Output path for the generated HiPACE++ job script file. The resulting 
        file is given executable permissions.

    filename_input : str
        Path to the HiPACE++ input file (usually created using 
        :func:`hipace_write_inputs() <abel.wrappers.hipace.hipace_wrapper.hipace_write_inputs>`). 

    num_nodes : int, optional
        Number of compute nodes to allocate for the simulation job. Defaults to 
        1.

    num_tasks_per_node : int, optional
        Number of MPI tasks per compute node and sets the number of GPUs per 
        node to the same number. Defaults to 8.

    Returns
    -------
    None
        The function writes the completed job submission script to 
        ``filename_job_script`` and does not return any value. The script is 
        made executable (``chmod 0777``).

    Notes
    -----
    - The following fields from :class:`CONFIG <abel.CONFIG>` are required:
      
      * ``CONFIG.project_name``
      * ``CONFIG.hipace_binary``
      * ``CONFIG.partition_name_small``
      * ``CONFIG.partition_name_devel``
      * ``CONFIG.partition_name_standard``

    - The job partition [2]_ is selected based on the requested resources:
      
      * Partition given by ``CONFIG.partition_name_small`` for ≤2 nodes and ≤8 tasks per node.
      * Partition given by ``CONFIG.partition_name_devel`` for ≤32 nodes and ≤8 tasks per node.
      * Partition given by ``CONFIG.partition_name_standard`` otherwise.

    - Memory per GPU is assumed to be 60 GB and is scaled with ``num_tasks_per_node``.

    - The generated script uses placeholders defined in ``job_script_template``, typically found in the same directory as this function.

    References
    ----------
    .. [1] LUMI Supercomputer batch jobs documentation:
           https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/batch-job/

    .. [2] LUMI Supercomputer job scheduling documentation:
           https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/partitions/
    """
    
    # locate template file
    filename_job_script_template = os.path.join(os.path.dirname(__file__), 'job_script_template')
    
    # set the partition based on the number of nodes and tasks
    # based on LUMI (see https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/partitions/)
    if num_nodes <= 2 and num_tasks_per_node <= 8:
        partition_name = CONFIG.partition_name_small
    elif num_nodes <= 32 and num_tasks_per_node <= 8:
        partition_name = CONFIG.partition_name_devel
    else:
        partition_name = CONFIG.partition_name_standard
    
    # calculate the memory
    memory_per_gpu = 60 # [GB]
        
    # define inputs
    inputs = {'project_name': CONFIG.project_name,
              'partition_name': partition_name,
              'memory': str(int(memory_per_gpu*num_tasks_per_node))+'g',
              'num_nodes': num_nodes,
              'num_tasks_per_node': num_tasks_per_node,
              'hipace_binary_path': CONFIG.hipace_binary,
              'filename_input': filename_input}
    
    # fill in template file
    with open(filename_job_script_template, 'r') as fin, open(filename_job_script, 'w') as fout:
        results = Template(fin.read()).substitute(inputs)
        fout.write(results)
        
    # make executable
    Path(filename_job_script).chmod(0o0777)


# ==================================================
# run HiPACE++
def hipace_run(filename_job_script, num_steps, runfolder=None, quiet=False):
    """
    Run a HiPACE++ simulation locally or on a Slurm-based cluster using the
    provided job script, and load the resulting beam and driver data from the
    final output file.

    Parameters
    ----------
    filename_job_script : str
        Path to the batch job script used to submit the job. Typically generated
        by :func:`hipace_write_jobscript() <abel.wrappers.hipace.hipace_wrapper.hipace_write_jobscript>`.

    num_steps : int
        Determines which HDF5 file is loaded and used to extract the outputs
        ``beam`` and ``driver`` (e.g., ``num_steps=100`` loads the file
        ``openpmd_000100.h5``).

    runfolder : str, optional
        Path to the folder where the HiPACE++ job is executed and where results
        are written. If ``None``, derived from ``filename_job_script``.
        Defaults to ``None``.

    quiet : bool, optional
        If ``True``, suppresses terminal output during execution.
        Defaults to ``False``.

    Returns
    -------
    beam : ``Beam`` | None
        The loaded beam from the designated HiPACE++ output file.
        Set to ``None`` if the beam could not be loaded.

    driver : ``Beam``
        The loaded driver object from the designated HiPACE++ output file.
    """

    # TODO: need to check that num_steps is valid (positive int <= max_step in input_template).

    # extract runfolder from job script name
    if runfolder == None:
        runfolder = os.path.dirname(filename_job_script)

    # run HIPACE++
    if CONFIG.cluster_name == 'LOCAL':
        _hipace_run_local(filename_job_script, runfolder, quiet=quiet)
    else:
        _hipace_run_slurm(filename_job_script, num_steps, runfolder, quiet=quiet)
    
    # when finished, load the beam and driver
    filename = os.path.join(runfolder, "diags/hdf5/openpmd_{:06}.h5".format(int(num_steps)))
    
    try:
        beam = Beam.load(filename, beam_name='beam')
    except:
        beam = None
    driver = Beam.load(filename, beam_name='driver')
    
    return beam, driver


# ==================================================
def _hipace_run_local(filename_job_script, runfolder, quiet=False):
    "Helper for running HiPACE++ on the local machine. Returns when job is complete."

    if not quiet:
        print(f"Running HiPACE++ locally in folder '{runfolder}'...")

    #Get the input filename out of the job script
    # This could probably be solved in a better way,
    # but that would requiring reformulating the Slurm logic a bit.
    filename_input = None
    with open(filename_job_script, 'r') as fin:
        inLines = fin.readlines()
        for line in inLines:
            ls = line.strip()
            if ls.startswith('#'):
                lss = ls.split()
                if len(lss) < 3:
                    continue
                if lss[1] == 'filename_input':
                    if not ((lss[2].startswith('"') or lss[2].startswith("'")) and \
                            (lss[2].endswith('"') or lss[2].endswith("'"))):
                        raise ValueError('Unexpected format in job script, found line "'+line+'" but unable to parse')
                    if filename_input != None:
                        raise RuntimeError('Double find of input filename?')

                    filename_input = lss[2][1:-1]
    if filename_input == None:
        raise RuntimeError('Did not find the input filename in the HiPACE++ job file')

    import time
    start_time = time.time()
    #Run HiPACE++
    process = subprocess.Popen([CONFIG.hipace_binary,filename_input], cwd=runfolder,\
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, \
                               close_fds=True, bufsize=1, universal_newlines=True )
    for line in iter(process.stdout.readline, ''):
        #TODO: respect 'quiet' flag
        print(line.rstrip())
    process.stdout.close()
    returncode = process.wait()

    print(f'HiPACE++ complete, returncode = {returncode}, execution time={time.time()-start_time : .1f} [s]')
    if returncode != 0:
        raise RuntimeError('Errors during HiPACE++ simulation')


# ==================================================
def _hipace_run_slurm(filename_job_script, num_steps, runfolder, quiet=False):
    "Helper for running HiPACE++ on a batch system using Slurm. Returns when job is complete."

    # run system command
    cmd = 'cd ' + runfolder + ' && sbatch ' + os.path.basename(filename_job_script)
    if quiet:
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
    else:
        # run the command
        output = subprocess.check_output(cmd, shell=True)
        
        # get the job id
        words = str(output).replace('\\n','').replace("'",'').split()
        jobid = int(words[3].strip())

        # set up a progress bar
        desc = '>> Queueing HiPACE++ (job ' + str(jobid) + ')'
        pbar = tqdm(total=int(num_steps), desc=desc, unit=' steps', leave=True, file=sys.stdout, colour='green')
    
    # progress loop
    fail_counter = 0
    while True:

        try:
            # get the run queue
            cmd = 'squeue --job ' + str(jobid)
            output = subprocess.check_output(cmd, shell=True)
            lines = str(output).split('\\n')
            clean_line = " ".join(lines[1].split())
            keywords = clean_line.split()

            fail_counter = 0
            
            # extract progress
            if len(keywords) > 1:
                status = keywords[4]
                time_spent = keywords[5]
                if status == 'PD': # starting
                    if not quiet:
                        pbar.set_description('>> Starting HiPACE++ (job ' + str(jobid) + ')')
                        pbar.update(0)
                
                elif status == 'R': # running
                    if not quiet:
                        pbar.set_description('>>> Running HiPACE++ (job ' + str(jobid) + ')')
                        
                    # read progress from output file
                    outputfile = os.path.join(runfolder,'hipace-' + str(jobid) + '.out')
                    if os.path.exists(outputfile) and not quiet:
                        with open(outputfile, 'rb') as f:
                            try:  # catch OSError in case of a one line file 
                                f.seek(-2, os.SEEK_END)
                                while f.read(1) != b'\n':
                                    f.seek(-2, os.SEEK_CUR)
                            except OSError:
                                f.seek(0)
                            last_line = f.readline().decode()
                        stepnum, position = 0, 0
                        split_line = last_line.split(' step ', 1)
                        if len(split_line) > 1:
                            split_line2 = split_line[1].split(' at time = ', 1)
                            stepnum = int(split_line2[0])
                            position = float(split_line2[1].split(' with dt = ', 1)[0])*SI.c
                        pbar.update(int(stepnum-pbar.n))
                    
                elif status == 'CG': # closing
                    if not quiet:
                        pbar.set_description('>> Finished HiPACE++ (job ' + str(jobid) + ')')
                        pbar.update(int(num_steps-pbar.n))
                        pbar.close()
                    break
            else:
                break
        except:
            fail_counter += 1
            if fail_counter > 5:
                break
        
        # wait for some time
        wait_time = 3 # [s]
        time.sleep(wait_time)


# ==================================================
def hipaceHdf5_2_abelBeam(data_dir, hipace_iteration_idx, species='beam'):
    """
    Load an ABEL beam from a HiPACE++ HDF5 output file (OpenPMD format).

    Parameters
    ----------
    data_dir : str
        Path to the directory containing all HiPACE++ HDF5 output files.

    hipace_iteration_idx : int
        Specifies the simulation iteration number to be extracted out of all 
        available output files in ``data_dir``.

    species : str, optional
        Specifies the name of the beam to be extracted. Defaults to ``'beam'``.


    Returns
    ----------
    beam : ``Beam``
        The extracted beam.
    """

    from openpmd_viewer import OpenPMDTimeSeries

    opmd_time_series = OpenPMDTimeSeries(data_dir)
    hipace_iteration = opmd_time_series.iterations[hipace_iteration_idx]
    
    xs_hipace, uxs_hipace, ys_hipace, uys_hipace, zs_hipace, uzs_hipace, weights_hipace, masses_hipace, charges_hipace = opmd_time_series.get_particle( ['x', 'ux', 'y', 'uy', 'z', 'uz', 'w', 'mass', 'charge'], species=species, iteration=hipace_iteration, plot=False )
    beam = Beam()
    beam.set_phase_space(xs=xs_hipace, ys=ys_hipace, zs=zs_hipace, 
                            uxs=uxs_hipace*SI.c, uys=uys_hipace*SI.c, 
                            uzs=uzs_hipace*SI.c, 
                            Q=np.sum(charges_hipace*weights_hipace), weightings=weights_hipace)
    
    beam.location = opmd_time_series.t[hipace_iteration_idx]*SI.c
    beam.particle_mass = masses_hipace[0]

    return beam

