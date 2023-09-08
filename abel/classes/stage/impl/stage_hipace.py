from abel import Stage, CONFIG
from abel.apis.hipace.hipace_api import hipace_write_inputs, hipace_run, hipace_write_jobscript
from abel.utilities.plasma_physics import *
import scipy.constants as SI
from matplotlib import pyplot as plt
import numpy as np
import os, shutil, uuid, copy
from openpmd_viewer import OpenPMDTimeSeries


class StageHipace(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None, add_driver_to_beam=False, keep_data=False, full_output=False):
        
        super().__init__(length, nom_energy_gain, plasma_density)
        
        self.driver_source = driver_source
        self.add_driver_to_beam = add_driver_to_beam
        
        self.keep_data = keep_data
        self.full_output = full_output
        
        self.__initial_wakefield = None
        self.__final_wakefield = None
        
        self.__initial_driver = None
        self.__final_driver = None

        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        ## PREPARE TEMPORARY FOLDER
        
        # make temp folder
        if not os.path.exists(CONFIG.temp_path):
            os.mkdir(CONFIG.temp_path)
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
        
        # make directory
        if not os.path.exists(tmpfolder):
            os.mkdir(tmpfolder)
        
        
        # SAVE BEAMS
        
        # saving beam to temporary folder
        filename_beam = tmpfolder + 'beam.h5'
        beam0.save(filename = filename_beam)
        
        # produce and save drive beam
        filename_driver = tmpfolder + 'driver.h5'
        driver = self.__get_initial_driver()
        driver.save(filename = filename_driver, beam_name = 'driver')
        
        
        # MAKE INPUT FILE
        
        # make longitudinal box range
        num_sigmas = 6
        box_min_z = beam0.z_offset() - num_sigmas * beam0.bunch_length()
        box_max_z = driver.z_offset() + num_sigmas * driver.bunch_length()
        box_range_z = [box_min_z, box_max_z]
        
        # making transverse box size
        box_size_xy = 5 * blowout_radius(self.plasma_density, driver.peak_current())
        
        # calculate the time step
        gamma_min = min(beam0.gamma(),driver.gamma()/2)
        k_beta = k_p(self.plasma_density)/np.sqrt(2*gamma_min)
        T_betatron = (2*np.pi/k_beta)/SI.c
        time_step0 = T_betatron/20
        
        # convert to number of steps (and re-adjust timestep to be divisible)
        self.num_steps = np.ceil(self.length/(time_step0*SI.c))
        time_step = self.length/(self.num_steps*SI.c)

        # overwrite output period
        if self.full_output:
            output_period = 1
        else:
            output_period = None
        
        # input file
        filename_input = tmpfolder + 'input_file'
        hipace_write_inputs(filename_input, filename_beam, filename_driver, self.plasma_density, self.num_steps, time_step, box_range_z, box_size_xy, output_period=output_period)
        
        
        ## RUN SIMULATION
        
        # make job script
        filename_job_script = tmpfolder + 'run.sh'
        hipace_write_jobscript(filename_job_script, filename_input)
        
        # run HiPACE++
        beam = hipace_run(filename_job_script, self.num_steps)
        
        
        ## ADD METADATA
        
        # copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        # remove nan particles
        beam.remove_nans()
        
        # extract wakefield data
        source_folder = tmpfolder + 'diags/hdf5/'
        self.__extract_wakefield(source_folder)
        
        # save drivers
        self.__initial_driver = driver
        
        
        ## MOVE AND DELETE TEMPORARY DATA
        
        # delete temp files 
        if self.keep_data or (savedepth > 0 and runnable is not None):
            destination_folder = runnable.shot_path() + '/stage_' + str(beam0.stage_number)
            shutil.move(source_folder, destination_folder)
        
        if os.path.exists(tmpfolder):
            shutil.rmtree(tmpfolder)

        return super().track(beam, savedepth, runnable, verbose)
    
        
    def get_length(self):
        return self.length
    
    def get_nom_energy_gain(self):
        return self.nom_energy_gain
    
    def energy_efficiency(self):
        return None # TODO
    
    def energy_usage(self):
        return None # TODO
    
    def matched_beta_function(self, energy):
        return beta_matched(self.plasma_density, energy)
    
    def __extract_wakefield(self, path):
        
        # prepare to read simulation data
        ts = OpenPMDTimeSeries(path)

        # save initial on-axis wakefield
        Ez0, metadata0 = ts.get_field(field='Ez', iteration=0)
        zs0 = metadata0.z
        Ez0_onaxis = Ez0[:,round(len(metadata0.x)/2)]
        self.__initial_wakefield = (zs0, Ez0_onaxis)
        
        # save final on-axis wakefield
        Ez, metadata = ts.get_field(field='Ez', iteration=self.num_steps)
        zs = metadata.z
        Ez_onaxis = Ez[:,round(len(metadata.x)/2)]
        self.__final_wakefield = (zs, Ez_onaxis)
        
    def __get_initial_driver(self):
        if self.__initial_driver is not None:
            return self.__initial_driver
        else:
            return self.driver_source.track()
        
        
    def plot_wakefield(self, beam=None):
        
        # extract wakefield if not already existing
        if (self.__initial_wakefield is None) or (self.__final_wakefield is None):
            return

        # assign to variables
        zs0, Ezs0 = self.__initial_wakefield
        zs, Ezs = self.__final_wakefield
        
        # get current profile
        driver = copy.deepcopy(self.__get_initial_driver())
        driver += beam
        Is, ts = driver.current_profile(bins=np.linspace(min(zs/SI.c), max(zs/SI.c), int(np.sqrt(len(driver))/2)))
        zs_I = ts*SI.c
        
        # plot it
        fig, axs = plt.subplots(2, 1)
        fig.set_figwidth(CONFIG.plot_width_default)
        fig.set_figheight(9)
        col0 = "xkcd:light gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        af = 0.1
        zlims = [min(zs)*1e6, max(zs)*1e6]
        
        axs[0].plot(zs*1e6, np.zeros(zs.shape), '-', color=col0)
        if self.nom_energy_gain is not None:
            axs[0].plot(zs*1e6, -self.nom_energy_gain/self.get_length()*np.ones(zs.shape)/1e9, ':', color=col0)
        axs[0].plot(zs0*1e6, Ezs0/1e9, '-', color=col1)
        axs[0].plot(zs*1e6, Ezs/1e9, ':', color=col2)
        axs[0].set_xlabel('z (um)')
        axs[0].set_ylabel('Longitudinal electric field (GV/m)')
        axs[0].set_xlim(zlims)
        axs[0].set_ylim(bottom=-wave_breaking_field(self.plasma_density)/1e9, top=1.3*max(Ezs)/1e9)
        
        axs[1].fill(np.concatenate((zs_I, np.flip(zs_I)))*1e6, np.concatenate((-Is, np.zeros(Is.shape)))/1e3, color=col1, alpha=af)
        axs[1].plot(zs_I*1e6, -Is/1e3, '-', color=col1)
        axs[1].set_xlabel('z (um)')
        axs[1].set_ylabel('Beam current (kA)')
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=0, top=1.2*max(-Is)/1e3)
        