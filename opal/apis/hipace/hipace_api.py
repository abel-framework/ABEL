import os, subprocess, time
import numpy as np
from string import Template
from pathlib import Path
from opal import CONFIG, Beam
from opal.utilities.plasma_physics import k_p


# write the HiPACE++ input script to file
def hipace_write_inputs(filename_input, filename_beam, filename_driver, plasma_density, num_steps, time_step, box_min_z, box_max_z, box_size_x=None, box_size_y=None, n_cell_x=255, n_cell_y=255, n_cell_z=256):
    
    # create temporary stream file and folder
    box_size_default = 10/k_p(plasma_density) # TODO: calculate blowout radius based on beam outside here
    if box_size_x is None:
        box_size_x = box_size_default    
    if box_size_y is None:
        box_size_y = box_size_default
    
    # locate template file
    filename_input_template = CONFIG.opal_path + 'opal/apis/hipace/input_template'
    
    # define inputs
    inputs = {'n_cell_x': int(n_cell_x), 
              'n_cell_y': int(n_cell_y), 
              'n_cell_z': int(n_cell_z),
              'plasma_density': plasma_density,
              'grid_low_x': -box_size_x/2,
              'grid_high_x': box_size_x/2,
              'grid_low_y': -box_size_y/2,
              'grid_high_y': box_size_y/2,
              'grid_low_z': box_min_z,
              'grid_high_z': box_max_z,
              'time_step': time_step,
              'max_step': int(num_steps),
              'output_period': int(num_steps),
              'filename_beam': filename_beam,
              'filename_driver': filename_driver}

    # fill in template file
    with open(filename_input_template, 'r') as fin, open(filename_input, 'w') as fout:
        results = Template(fin.read()).substitute(inputs)
        fout.write(results)


# write the HiPACE++ job script to file
def hipace_write_jobscript(filename_job_script, filename_input):
    
    # locate template file
    filename_job_script_template = CONFIG.opal_path + 'opal/apis/hipace/job_script_template'
    
    # define inputs
    inputs = {'nodes': 1,
              'filename_input': filename_input}
    
    # fill in template file
    with open(filename_job_script_template, 'r') as fin, open(filename_job_script, 'w') as fout:
        results = Template(fin.read()).substitute(inputs)
        fout.write(results)
        
    # make executable
    Path(filename_job_script).chmod(0o0777)


# run HiPACE++
def hipace_run(filename_job_script, num_steps, runfolder=None, quiet=False):
    
    # make run folder automatically from job script folder
    if runfolder is None:
        runfolder = str.replace(filename_job_script, os.path.basename(filename_job_script), '')
    
    # run system command
    cmd = 'cd ' + runfolder + ' && sbatch ' + filename_job_script
    if quiet:
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
    else:
        #subprocess.call(cmd, shell=True, stdout=None)
        output = subprocess.check_output(cmd, shell=True)
        words = str(output).replace('\\n','').replace("'",'').split()
        jobid = int(words[3].strip())
        print("Running job " + str(jobid))
    
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
            if status == 'PD':
                print('>> Starting (' + time_spent + ')')
            elif status == 'R':
                print('>> Running (' + time_spent + ')')
            elif status == 'CG':
                print('>> Ending (' + time_spent + ')')
        else:
            print('>> Done!')
            # the simulation is done
            break
        
        # wait for some time
        time.sleep(5)
    
    # when finished, load the beam
    filename = runfolder + "diags/hdf5/openpmd_{:06}.h5".format(int(num_steps))
    beam = Beam.load(filename)
    
    return beam