import os, subprocess, time
import numpy as np
from string import Template
from pathlib import Path
from abel import CONFIG, Beam
from abel.utilities.plasma_physics import k_p

# write the HiPACE++ input script to file
def hipace_write_inputs(filename_input, filename_beam, filename_driver, plasma_density, num_steps, time_step, box_range_z, box_size, output_period=None, ion_motion=True, ion_species='H', radiation_reaction=False, beam_ionization=True, num_cell_xy=511, num_cell_z=424, driver_only=False, filename_test_particle=''):

    if output_period is None:
        output_period = int(num_steps)
        
    # locate template file
    filename_input_template = CONFIG.abel_path + 'abel/apis/hipace/input_template'

    # prepare plasma components (based on ion motion)
    if ion_motion:
        plasma_components = 'electrons ions'
    else:
        plasma_components = 'plasma'
        
    beam_components = 'driver'
    if not driver_only:
        beam_components += ' beam'

    if filename_test_particle:
        beam_components += ' test_particle'
        
    
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
              'filename_beam': filename_beam,
              'filename_driver': filename_driver,
              'filename_test_particle': filename_test_particle}

    # fill in template file
    with open(filename_input_template, 'r') as fin, open(filename_input, 'w') as fout:
        results = Template(fin.read()).substitute(inputs)
        fout.write(results)


# write the HiPACE++ job script to file
def hipace_write_jobscript(filename_job_script, filename_input, num_nodes=1):
    
    # locate template file
    filename_job_script_template = CONFIG.abel_path + 'abel/apis/hipace/job_script_template'
    
    # define inputs
    inputs = {'num_nodes': num_nodes,
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
        filename_job_script = os.path.basename(filename_job_script)
    
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
                break
        else:
            print('>> Done!')
            # the simulation is done
            break
        
        # wait for some time
        wait_time = 10 # [s]
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