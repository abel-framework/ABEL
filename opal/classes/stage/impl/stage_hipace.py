from opal import Stage, CONFIG
from opal.apis.hipace.hipace_api import hipace_write_inputs, hipace_run, hipace_write_jobscript
from opal.utilities.plasma_physics import k_p, beta_matched
import scipy.constants as SI
import numpy as np
import os, shutil, uuid


class StageHipace(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None, add_driver_to_beam=False, keep_data=False):
        
        super().__init__(length, nom_energy_gain, plasma_density)
        
        self.driver_source = driver_source
        self.add_driver_to_beam = add_driver_to_beam
        
        self.keep_data = keep_data

        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # make temp folder
        if not os.path.exists(CONFIG.temp_path):
            os.mkdir(CONFIG.temp_path)
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
        
        # make directory
        if not os.path.exists(tmpfolder):
            os.mkdir(tmpfolder)
        
        #saving beam to temporary folder
        filename_beam = tmpfolder + 'beam.h5'
        beam0.save(filename = filename_beam, hipace_units=True)
        
        #producing and saving drive beam
        filename_driver = tmpfolder + 'driver.h5'
        driver = self.driver_source.track()
        driver.save(filename = filename_driver, beam_name = 'driver', hipace_units=True)
        
        # making box length
        box_min_z = min(min(beam0.zs()),min(driver.zs()))
        box_max_z = max(max(beam0.zs()),max(driver.zs()))
        
        gamma_min = min(beam0.gamma(),driver.gamma())
        k_beta = k_p(self.plasma_density)/np.sqrt(2*gamma_min)
        T_betatron = (2*np.pi/k_beta)/SI.c
        time_step0 = T_betatron/40
        
        num_steps = np.ceil(self.length/(time_step0*SI.c))
        time_step = self.length/(num_steps*SI.c)
        
        # input file
        filename_input = tmpfolder + 'input_file'
        hipace_write_inputs(filename_input, filename_beam, filename_driver, self.plasma_density, num_steps, time_step, box_min_z, box_max_z)
        
        #job script
        filename_job_script = tmpfolder + 'run.sh'
        hipace_write_jobscript(filename_job_script, filename_input)
        
        # run hipace
        beam = hipace_run(filename_job_script, num_steps)
        
        # copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        # remove nan particles
        beam.remove_nans()
        
        # delete temp files 
        if self.keep_data or (savedepth > 0 and runnable is not None):
            source_folder = tmpfolder + 'diags/hdf5/'
            destination_folder = runnable.shot_path() + '/stage_' + str(beam0.stage_number)
            shutil.move(source_folder, destination_folder)
        
        if os.path.exists(tmpfolder):
            shutil.rmtree(tmpfolder)

        return super().track(beam, savedepth, runnable, verbose)
    
        
    def get_length(self):
        return self.length
    
    def get_energy_gain(self):
        return self.nom_energy_gain
    
    def get_energy_efficiency(self):
        return None # TODO
    
    def get_energy_usage(self):
        return None # TODO
    
    def get_matched_beta_function(self, energy):
        return beta_matched(self.plasma_density, energy)
    
    def plot_wakefield(self):
        pass # TODO
    
    
    
    
  