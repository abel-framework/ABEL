import os, subprocess, time, sys
import numpy as np
import scipy.constants as SI
from string import Template
from pathlib import Path
from abel import CONFIG, Beam
from tqdm import tqdm
from abel.utilities.plasma_physics import k_p

# write the HiPACE++ input script to file
def hipace_write_inputs(filename_input, filename_beam, filename_driver, plasma_density, num_steps, time_step, box_range_z, box_size, output_period=None, ion_motion=True, ion_species='H', radiation_reaction=False, beam_ionization=True, num_cell_xy=511, num_cell_z=512, driver_only=False, density_table_file=None, no_plasma=False, external_focusing_radial=0, filename_test_particle='empty.h5'):

    if output_period is None:
        output_period = int(num_steps)
        
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

    if not filename_test_particle == 'empty.h5':
        beam_components += ' test_particle'
    
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
    if abs(external_focusing_radial) > 0:
        external_focusing_radial_comment = ''
    else:
        external_focusing_radial_comment = '#'
    
    # define inputs
    inputs = {'num_cell_x': int(num_cell_xy), 
              'num_cell_y': int(num_cell_xy), 
              'num_cell_z': int(num_cell_z),
              'plasma_density': plasma_density,
              'grid_low_x': -box_size/2,
              'grid_high_x': box_size/2,
              'grid_low_y': -box_size/2,
              'grid_high_y': box_size/2,
              'grid_low_z': min(box_range_z),
              'grid_high_z': max(box_range_z),
              'time_step': time_step,
              'max_step': int(num_steps),
              'output_period': output_period,
              'radiation_reaction': int(radiation_reaction),
              'beam_components': beam_components,
              'plasma_components': plasma_components,
              'ion_species': ion_species,
              'beam_ionization': int(beam_ionization),
              'external_focusing_radial': abs(external_focusing_radial),
              'external_focusing_radial_comment': external_focusing_radial_comment,
              'filename_beam': filename_beam,
              'filename_driver': filename_driver,
              'filename_test_particle': filename_test_particle}

    # fill in template file
    with open(filename_input_template, 'r') as fin, open(filename_input, 'w') as fout:
        results = Template(fin.read()).substitute(inputs)
        fout.write(results)


# write the HiPACE++ job script to file
def hipace_write_jobscript(filename_job_script, filename_input, num_nodes=1, num_tasks_per_node=8):
    
    # locate template file
    filename_job_script_template = os.path.join(os.path.dirname(__file__), 'job_script_template')
    
    # set the partition based on the number of nodes and tasks
    if num_nodes == 1 and num_tasks_per_node < 8:
        partition_name = CONFIG.partition_name_small
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


# run HiPACE++
def hipace_run(filename_job_script, num_steps, runfolder=None, quiet=False):

    # extract runfolder from job script name
    if runfolder == None:
        runfolder = os.path.dirname(filename_job_script)

    # run HIPACE++
    if CONFIG.cluster_name == 'LOCAL':
        _hipace_run_local(filename_job_script, runfolder, quiet=False)
    else:
        _hipace_run_slurm(filename_job_script, num_steps, runfolder, quiet=False)
    
    # when finished, load the beam and driver
    filename = os.path.join(runfolder, "diags/hdf5/openpmd_{:06}.h5".format(int(num_steps)))
    
    try:
        beam = Beam.load(filename, beam_name='beam')
    except:
        beam = None
    driver = Beam.load(filename, beam_name='driver')
    
    return beam, driver


def _hipace_run_local(filename_job_script, runfolder, quiet=False):
    "Helper for running HiPACE++ on the local machine. Returns when job is complete."

    if not quiet:
        print(f"Running HiPACE locally in folder '{runfolder}'...")

    #Get the input filename out of the job script
    # This could probably be solved in a better way,
    # but that would requiring reformulating the SLURM logic a bit.
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
        raise RuntimeError('Did not find the input filename in the hipace job file')

    import time
    start_time = time.time()
    #Run HIPACE
    process = subprocess.Popen([CONFIG.hipace_binary,filename_input], cwd=runfolder,\
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, \
                               close_fds=True, bufsize=1, universal_newlines=True )
    for line in iter(process.stdout.readline, ''):
        #TODO: respect 'quiet' flag
        print(line.rstrip())
    process.stdout.close()
    returncode = process.wait()

    print(f'HiPace complete, returncode = {returncode}, execution time={time.time()-start_time : .1f} [s]')
    if returncode != 0:
        raise RuntimeError('Errors during HiPace simulation')


def _hipace_run_slurm(filename_job_script, num_steps, runfolder, quiet=False):
    "Helper for running HiPACE++ on a batch system using SLURM. Returns when job is complete."

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
    while True:
        
        # get the run queue
        cmd = 'squeue --job ' + str(jobid)
        output = subprocess.check_output(cmd, shell=True)
        lines = str(output).split('\\n')
        clean_line = " ".join(lines[1].split())
        keywords = clean_line.split()
        
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

        # wait for some time
        wait_time = 3 # [s]
        time.sleep(wait_time)
    
    # when finished, load the beam and driver
    filename = runfolder + "diags/hdf5/openpmd_{:06}.h5".format(int(num_steps))
    try:
        beam = Beam.load(filename, beam_name='beam')
    except:
        beam = None
    driver = Beam.load(filename, beam_name='driver')
    
    try:
        test_particle = Beam.load(filename, beam_name='test_particle')
    except:
        test_particle = None
    
    return beam, driver, test_particle


