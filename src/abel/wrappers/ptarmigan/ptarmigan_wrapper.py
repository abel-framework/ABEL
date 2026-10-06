import numpy as np
import os, subprocess
from string import Template
from abel.CONFIG import CONFIG

def ptarmigan_write_inputs(filename_input, beam, laser_a0, laser_wavelength, laser_duration, laser_waist_size, laser_polarization, collision_angle_deg, increase_pair_rate_by):

    # write beam to file
    filename_beam = 'beam.h5'
    filepath_beam = os.path.join(os.path.dirname(filename_input), 'beam.h5')
    beam2ptarmigan_h5(beam, filepath_beam)
    
    # define inputs
    inputs = {'dt_multiplier': float(0.5), 
              'radiation_reaction': 'true', 
              'pair_creation': 'true',
              'increase_pair_rate_by': increase_pair_rate_by,
              'laser_a0': float(laser_a0),
              'laser_wavelength_um': laser_wavelength*1e6,
              'laser_duration_fs': laser_duration*1e15,
              'laser_waist_size_um': laser_waist_size*1e6,
              'laser_polarization': laser_polarization,
              'beam_file': filename_beam,
              'collision_angle_deg': collision_angle_deg}

    filename_input_template = os.path.join(os.path.dirname(__file__), 'input_template_beam.yml')

    # fill in template file
    with open(filename_input_template, 'r') as fin, open(filename_input, 'w') as fout:
        results = Template(fin.read()).substitute(inputs)
        fout.write(results)


def ptarmigan_run(filename_input, runfolder=None, quiet=False):

    # extract runfolder from job script name
    if runfolder == None:
        runfolder = os.path.dirname(filename_input)
    
    # run system command
    cmd = CONFIG.ptarmigan_binary + ' ' + filename_input
    if not quiet:
        stdout = subprocess.DEVNULL
    else:
        stdout = None
    subprocess.call(cmd, shell=True, stdout=stdout)


def ptarmigan_extract_outputs(filename_output):

    from types import SimpleNamespace
    import h5py

    # declare the output structure
    output = SimpleNamespace()
    
    with h5py.File(filename_output, 'r') as f:

        # extract configuration info
        config = f['config']

        # find the units
        unit_pos = config['unit/position'][()].decode('utf-8')
        if str(unit_pos) == 'mm':
            scale_pos = 1e-3 # convert to [m]
        else:
            raise Exception('Unknown position unit')
        
        unit_mom = config['unit/momentum'][()].decode('utf-8')
        if str(unit_mom) == 'GeV/c':
            scale_mom = 1e9 # convert to [eV/m]
        else:
            raise Exception('Unknown momentum unit')

        # get the initial beam particle number (for sorting)
        num_particles_input = config['beam/n'][()]
        
        # get the final-state dataset (all particles)
        dataset = f['final-state']
        
        # get the electron data
        output.electron = SimpleNamespace()
        output.electron.x = dataset['electron/position'][()][:,1]*scale_pos
        output.electron.y = dataset['electron/position'][()][:,2]*scale_pos
        output.electron.z = dataset['electron/position'][()][:,0]*scale_pos
        output.electron.px = dataset['electron/momentum'][()][:,1]*scale_mom
        output.electron.py = dataset['electron/momentum'][()][:,2]*scale_mom
        output.electron.pz = dataset['electron/momentum'][()][:,0]*scale_mom
        output.electron.weights = dataset['electron/weight'][()]
        output.electron.n_gamma = dataset['electron/n_gamma'][()]
        output.electron.ids = dataset['electron/id'][()]
        output.electron.parent_ids = dataset['electron/parent_id'][()]
        output.electron.input_beam_mask = output.electron.ids < num_particles_input

        # get the positron data
        output.positron = SimpleNamespace()
        output.positron.x = dataset['positron/position'][()][:,1]*scale_pos
        output.positron.y = dataset['positron/position'][()][:,2]*scale_pos
        output.positron.z = dataset['positron/position'][()][:,0]*scale_pos
        output.positron.px = dataset['positron/momentum'][()][:,1]*scale_mom
        output.positron.py = dataset['positron/momentum'][()][:,2]*scale_mom
        output.positron.pz = dataset['positron/momentum'][()][:,0]*scale_mom
        output.positron.weights = dataset['positron/weight'][()]
        output.positron.n_gamma = dataset['positron/n_gamma'][()]
        output.positron.ids = dataset['positron/id'][()]
        output.positron.parent_ids = dataset['positron/parent_id'][()]
        output.positron.input_beam_mask = output.positron.ids < num_particles_input

        # get the laser data
        output.laser = SimpleNamespace()
        output.laser.a0 = config['laser/a0'][()]
        output.laser.fwhm_duration = config['laser/fwhm_duration'][()]
        output.laser.waist = config['laser/waist'][()]
        output.laser.polarization = config['laser/polarization'][()]
        output.laser.wavelength = config['laser/wavelength'][()]
        output.laser.absorption = dataset['laser/absorption'][()]
        output.laser.energy = dataset['laser/energy'][()]

        # get the beam data
        output.beam = SimpleNamespace()
        output.beam.num_particles = config['beam/n'][()]

        # get the photon data
        output.photon = SimpleNamespace()
        output.photon.a0_at_creation = dataset['photon/a0_at_creation'][()]
        output.photon.parent_chi = dataset['photon/parent_chi'][()]
        output.photon.weights = dataset['photon/weight'][()]

    return output


def beam2ptarmigan_h5(beam, filename):
    """
    Write a Ptarmigan-compatible HDF5 particle-beam file.
    """
    
    import h5py
    import scipy.constants as SI

    with h5py.File(filename, 'w') as f:

        # set the axis
        f.create_dataset('beam_axis', data=np.bytes_('+z'))

        # set the units
        units = f.create_group('config/unit')
        units.create_dataset('momentum',data=np.bytes_('GeV/c'))
        units.create_dataset('position', data=np.bytes_('mm'))
        scale_mom = 1e9
        scale_pos = 1e-3
        
        # prepare momentum and position 4-vectors
        scale_E = scale_mom
        scale_p = scale_mom*SI.e/SI.c
        scale_m = scale_mom*SI.e/SI.c**2

        # declare the arrays
        momentum = np.zeros(len(beam), dtype=np.dtype((np.float64, (4,))))
        position = np.zeros(len(beam), dtype=np.dtype((np.float64, (4,))))
        weight = np.ones(len(beam), dtype=np.float64)

        # fill the momentum array
        momentum[:, 0] = np.sqrt(SI.m_e**2/scale_m**2 + beam.pxs()**2/scale_p**2 + beam.pys()**2/scale_p**2 + beam.pzs()**2/scale_p**2)
        momentum[:, 1] = beam.pxs()/scale_p
        momentum[:, 2] = beam.pys()/scale_p
        momentum[:, 3] = beam.pzs()/scale_p

        # fill the position array
        position[:, 0] = beam.zs()/scale_pos
        position[:, 1] = beam.xs()/scale_pos
        position[:, 2] = beam.ys()/scale_pos
        position[:, 3] = beam.zs()/scale_pos

        # fill the weight array
        weight[:] = beam.weightings()
        
        # fill the polarization array
        polarization = np.zeros(len(beam), dtype=np.dtype((np.float64, (4,))))
        if beam.spin_polarization() > 0:
            polarization[:, 0] = np.ones_like(beam.spxs())
            polarization[:, 1] = beam.spxs()
            polarization[:, 2] = beam.spys()
            polarization[:, 3] = beam.spzs()
            
        # set the particle data
        particle_group = f.create_group('final-state/electron')
        particle_group.create_dataset('weight', data=weight)
        particle_group.create_dataset('momentum', data=momentum)
        particle_group.create_dataset('position', data=position)
        particle_group.create_dataset('polarization', data=polarization)

    

    