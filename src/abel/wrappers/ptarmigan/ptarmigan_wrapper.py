import numpy as np
import os, subprocess
from string import Template

def ptarmigan_write_inputs(filename_input, beam, laser_a0, laser_wavelength, laser_duration, laser_waist_size, laser_polarization, collision_angle_deg):

    # define inputs
    inputs = {'dt_multiplier': float(0.5), 
              'radiation_reaction': 'true', 
              'pair_creation': 'true',
              'increase_pair_rate_by': 1.0e4,
              'laser_a0': float(laser_a0),
              'laser_wavelength_um': laser_wavelength*1e6,
              'laser_duration_fs': laser_duration*1e15,
              'laser_waist_size_um': laser_waist_size*1e6,
              'laser_polarization': laser_polarization,
              'num_particles': len(beam),
              'bunch_population': beam.population(),
              'gamma': beam.gamma(),
              'rel_energy_spread': beam.rel_energy_spread(),
              'beam_size_um': np.sqrt(beam.beam_size_x()*beam.beam_size_y()) * 1e6, 
              'bunch_length_um': beam.bunch_length()*1e6,
              'divergence_urad': np.sqrt(beam.divergence_x()*beam.divergence_x())*1e6,
              'collision_angle_deg': collision_angle_deg}

    filename_input_template = os.path.join(os.path.dirname(__file__), 'input_template_gaussian.yml')
    
    # fill in template file
    with open(filename_input_template, 'r') as fin, open(filename_input, 'w') as fout:
        results = Template(fin.read()).substitute(inputs)
        fout.write(results)


def ptarmigan_run(filename_input, runfolder=None, quiet=False):

    # extract runfolder from job script name
    if runfolder == None:
        runfolder = os.path.dirname(filename_input)

    # executable
    ptarmigan_binary_loc = '/Users/carlal/UiO/Code/software/ptarmigan/target/release/'

    # run system command
    cmd = ptarmigan_binary_loc + 'ptarmigan ' + filename_input
    if not quiet:
        stdout = subprocess.DEVNULL
    else:
        stdout = None
    subprocess.call(cmd, shell=True, stdout=stdout)
    
    # run process
    #process = subprocess.Popen([ptarmigan_binary, filename_input], cwd=runfolder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True, bufsize=1, universal_newlines=True)
    # process.stdout.close()

    

