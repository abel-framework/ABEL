from abel import Stage, CONFIG
from abel.apis.hipace.hipace_api import hipace_write_inputs, hipace_run, hipace_write_jobscript
from abel.utilities.plasma_physics import *
from abel.utilities.relativity import energy2gamma
import scipy.constants as SI
from matplotlib import pyplot as plt
import numpy as np
import os, shutil, uuid, copy
from openpmd_viewer import OpenPMDTimeSeries
from abel.utilities.plasma_physics import k_p
from matplotlib.colors import LogNorm
from types import SimpleNamespace

import sys
sys.path.append(CONFIG.hipace_path + '/tools')
import read_insitu_diagnostics

class StageHipace(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None, add_driver_to_beam=False, keep_data=False, output=None, ion_motion=True, ion_species='H', beam_ionization=True, radiation_reaction=False, analytical = False):
        
        super().__init__(length, nom_energy_gain, plasma_density)
        
        self.driver_source = driver_source
        self.add_driver_to_beam = add_driver_to_beam
        
        self.keep_data = keep_data
        self.output = output
        
        self.ion_motion = ion_motion
        self.ion_species = ion_species
        self.beam_ionization = beam_ionization
        self.radiation_reaction = radiation_reaction

        self.evolution = SimpleNamespace()
        
        self.analytical = analytical
        
        self.__initial_wakefield = None
        self.__final_wakefield = None
        self.__initial_driver = None
        self.__final_driver = None
        self.__initial_transverse = None
        self.__final_transverse = None
        self.__bubble_size = None
        self.__initial_witness = None
        self.__final_witness = None
        self.__initial_rho = None
        self.__final_rho = None
        self.__final_focusing = None
        self.__initial_focusing = None
        self.__amplitude_evol = None
        self.__phase_advances = None
        

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
        filename_beam = 'beam.h5'
        path_beam = tmpfolder + filename_beam
        beam0.save(filename = path_beam)
        
        # produce and save drive beam
        filename_driver = 'driver.h5'
        path_driver = tmpfolder + filename_driver
        driver = self.__get_initial_driver()
        driver.save(filename = path_driver, beam_name = 'driver')
        
        
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
        
        if self.output is not None:
            remainder = self.num_steps % self.output
            if remainder >= self.output/2:  # If remainder is 10 or greater, round up
                self.num_steps = self.num_steps + (self.output - remainder)
            else:  # If remainder is less than 10, round down
                self.num_steps = self.num_steps - remainder
        
        time_step = self.length/(self.num_steps*SI.c)

        # overwrite output period
        if self.output is not None:
            output_period = self.output
        else:
            output_period = None
        
        # input file
        filename_input = 'input_file'
        path_input = tmpfolder + filename_input
        hipace_write_inputs(path_input, filename_beam, filename_driver, self.plasma_density, self.num_steps, time_step, box_range_z, box_size_xy, ion_motion=self.ion_motion, ion_species=self.ion_species, beam_ionization=self.beam_ionization, radiation_reaction=self.radiation_reaction, output_period=output_period)
        
        
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
        
        # clean extreme outliers
        beam.remove_halo_particles()

        # extract insitu diagnostics (beam)
        insitu_files = tmpfolder + 'diags/insitu/reduced_beam.*.txt'
        all_data = read_insitu_diagnostics.read_file(insitu_files)
        average_data = all_data['average']
        self.evolution.location = beam0.location + all_data['time']*SI.c
        self.evolution.charge = read_insitu_diagnostics.total_charge(all_data)
        self.evolution.energy = read_insitu_diagnostics.energy_mean_eV(all_data)
        self.evolution.x = average_data['[x]']
        self.evolution.y = average_data['[y]']
        self.evolution.xp = average_data['[ux]']/average_data['[uz]']
        self.evolution.yp = average_data['[uy]']/average_data['[uz]']
        self.evolution.energy_spread = read_insitu_diagnostics.energy_spread_eV(all_data)
        self.evolution.rel_energy_spread = self.evolution.energy_spread/self.evolution.energy
        self.evolution.beam_size_x = read_insitu_diagnostics.position_std(average_data, direction='x')
        self.evolution.beam_size_y = read_insitu_diagnostics.position_std(average_data, direction='y')
        self.evolution.bunch_length = read_insitu_diagnostics.position_std(average_data, direction='z')
        self.evolution.emit_nx = read_insitu_diagnostics.emittance_x(average_data)
        self.evolution.emit_ny = read_insitu_diagnostics.emittance_y(average_data)
        self.evolution.beta_x = np.sqrt(self.evolution.beam_size_x)/(self.evolution.emit_nx*energy2gamma(self.evolution.energy))
        self.evolution.beta_y = np.sqrt(self.evolution.beam_size_y)/(self.evolution.emit_ny*energy2gamma(self.evolution.energy))
        
        
        
        # extract wakefield data
        source_folder = tmpfolder + 'diags/hdf5/'
        self.__extract_wakefield(source_folder)
        self.__extract_witness(source_folder)
        self.__extract_final_driver(source_folder)
        self.__extract_transverse(source_folder)
        self.__extract_focusing(source_folder)
        self.__extract_rho(source_folder)
        if self.output is not None:
            self.__extract_amplitude_evol(source_folder)
        
        # save drivers
        self.__initial_driver = driver
        
        
        ## MOVE AND DELETE TEMPORARY DATA
        
        # delete temp files 
        if self.keep_data or (savedepth > 0 and runnable is not None):
            destination_folder = runnable.shot_path() + '/stage_' + str(beam0.stage_number)
            shutil.move(source_folder, destination_folder)
            source_folder2 = tmpfolder + 'diags/insitu/'
            destination_folder2 = runnable.shot_path() + '/stage_' + str(beam0.stage_number) + '_insitu'
            shutil.move(source_folder2, destination_folder2)
        
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
    
    def analytical_focusing(self):
        focusing = self.plasma_density * SI.e /(2 * SI.epsilon_0)
        return focusing

    def plot_evolution(self):
        
        # line format
        fmt = "-"
        col0 = "tab:gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        
        # plot evolution
        fig, axs = plt.subplots(3,2)
        fig.set_figwidth(CONFIG.plot_width_default*1.2)
        fig.set_figheight(CONFIG.plot_width_default*1.1)
        long_label = 'Location (m)'
        long_axis = self.evolution.location

        # plot energy
        axs[0,0].plot(long_axis, self.evolution.energy / 1e9, color=col1)
        axs[0,0].set_ylabel('Energy (GeV)')

        # plot energy spread
        axs[1,0].plot(long_axis, self.evolution.rel_energy_spread * 100, color=col1)
        axs[1,0].set_ylabel('Energy spread (%)')

        # plot charge
        axs[2,0].plot(long_axis, self.evolution.charge[0] * np.ones(long_axis.shape) * 1e9, ':', color=col0)
        axs[2,0].plot(long_axis, self.evolution.charge * 1e9, color=col1)
        axs[2,0].set_xlabel(long_label)
        axs[2,0].set_ylabel('Charge (nC)')

        # plot transverse offset
        axs[0,1].plot(long_axis, np.zeros(long_axis.shape), ':', color=col0)
        axs[0,1].plot(long_axis, self.evolution.x*1e6, color=col1)
        axs[0,1].plot(long_axis, self.evolution.y*1e6, color=col2)
        axs[0,1].set_ylabel('Transverse offset (um)')

        # plot beta function
        axs[1,1].plot(long_axis, self.evolution.beta_x*1e3, color=col1)
        axs[1,1].plot(long_axis, self.evolution.beta_y*1e3, color=col2)
        axs[1,1].set_ylabel('Beta function (mm)')

        # plot normalized emittance
        axs[2,1].plot(long_axis, self.evolution.emit_nx*1e6, color=col1)
        axs[2,1].plot(long_axis, self.evolution.emit_ny*1e6, color=col2)
        axs[2,1].set_xlabel(long_label)
        axs[2,1].set_ylabel('Emittance, rms (mm mrad)')
        
        
        plt.show()
        
        
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
        
        
    def __extract_rho(self, path):
        
        # prepare to read simulation data
        ts = OpenPMDTimeSeries(path)

        # Extract plasma and beam density
        Ez0, metadata0 = ts.get_field(field='rho', iteration=0)
        jz_beam0, metadataj0 = ts.get_field(field='jz_beam', iteration=0)
        old_extent = metadata0.imshow_extent
        old_extent_j0 = metadataj0.imshow_extent
        self.__initial_rho = (old_extent,  Ez0, jz_beam0, old_extent_j0)
        
        Ez0, metadata0 = ts.get_field(field='rho', iteration=self.num_steps)
        jz_beam, metadataj = ts.get_field(field='jz_beam', iteration=self.num_steps)
        old_extent = metadata0.imshow_extent
        old_extent_j = metadataj.imshow_extent
        self.__final_rho = (old_extent,  Ez0, jz_beam, old_extent_j)

        
    def __extract_transverse(self, path):
        
        # prepare to read simulation data
        ts = OpenPMDTimeSeries(path)
        x_w, z_w = self.__initial_witness
        x_d, z_d = ts.get_particle(species='driver', iteration=0, var_list=['x', 'z'])
        
        tail = np.mean(z_w) - 3*np.std(z_w)
        head = np.mean(z_w) + 3*np.std(z_w)
        
        # save initial on-axis wakefield
        E0, metadata0 = ts.get_field(field='ExmBy', iteration=0)
        zs0 = metadata0.z
        
        # interpolation in x for initial
        x_index = int(np.floor((np.mean(x_d)-metadata0.x[0])/metadata0.dx))
        frac = ((np.mean(x_d) - metadata0.x[0])/metadata0.dx) - x_index
        
        index_witness_head = int( np.ceil( (head -metadata0.z[0])/metadata0.dz) ) 
        index_witness_tail = int( np.floor( (tail -metadata0.z[0])/metadata0.dz) )
        
        F_trans_sliced_head = E0[index_witness_head, x_index]*(1-frac) + E0[index_witness_head, x_index+1]*frac
        
        zs0 = zs0[index_witness_tail:index_witness_head]
        E0_onaxis = E0[index_witness_tail:index_witness_head, x_index]*(1-frac) + E0[index_witness_tail:index_witness_head, x_index+1]*(frac) -  F_trans_sliced_head
        
        self.__initial_transverse = (zs0, E0_onaxis)
        
        # get final particle information
        x_w_f, z_w_f = ts.get_particle(species='beam', iteration=self.num_steps, var_list=['x', 'z'])
        x_d_f, z_d_f = ts.get_particle(species='driver', iteration=self.num_steps, var_list=['x', 'z'])
        
        # get tail and head for final beam
        tail_f = np.mean(z_w_f) - 3*np.std(z_w_f)
        head_f = np.mean(z_w_f) + 3*np.std(z_w_f)
        
        # save final on-axis wakefield
        E, metadata = ts.get_field(field='ExmBy', iteration=self.num_steps)
        zs = metadata.z
        
        # interpolation in x for final
        x_index_f = int(np.floor((np.mean(x_d_f)-metadata.x[0])/metadata.dx))
        frac_f = ((np.mean(x_d_f) - metadata.x[0])/metadata.dx) - x_index_f
        
        # slice the fields
        index_witness_head_f = int( np.ceil( (head_f -metadata.z[0])/metadata.dz) ) 
        index_witness_tail_f = int( np.floor( (tail_f -metadata.z[0])/metadata.dz) )
        
        zs = zs[index_witness_tail_f:index_witness_head_f]
        E_onaxis = E[index_witness_tail_f:index_witness_head_f, x_index_f]*(1-frac_f) + E[index_witness_tail_f:index_witness_head_f, x_index_f+1]*(frac_f)
        self.__final_transverse = (zs, E_onaxis)
        
    def __extract_witness(self, path):
        ts = OpenPMDTimeSeries(path)

        # get x and z coordinates of beam
        x_i, z_i = ts.get_particle(species='beam', iteration=0, var_list=['x', 'z'])
        x_f, z_f = ts.get_particle(species='beam', iteration=self.num_steps, var_list=['x', 'z'])
        
        # set initial and final witness
        self.__initial_witness = x_i, z_i
        self.__final_witness = x_f, z_f

        
    def __extract_final_driver(self, path):
        ts = OpenPMDTimeSeries(path)
        # get x and z coordinates of final driver beam
        x_f, z_f = ts.get_particle(species='driver', iteration=self.num_steps, var_list=['x', 'z']) 
        self.__final_driver = x_f, z_f
        
    def __extract_amplitude_evol(self, path):
        ts = OpenPMDTimeSeries(path)
        # array for amplitudes and phase_advances per output
        amplitudes = np.zeros(1+int(self.num_steps/self.output))
        phase_advances = np.zeros(1+int(self.num_steps/self.output))
        # get wavebreaking field
        wave_breaking = wave_breaking_field(self.plasma_density)
        # get normalised momentum in each direction
        puz_initial = np.mean(np.array(ts.get_particle(species = 'beam', iteration = 0, var_list = ['uz']))[0,:])
        pux_initial = np.mean(np.array(ts.get_particle(species = 'beam', iteration = 0, var_list = ['ux']))[0,:])
        puy_initial = np.mean(np.array(ts.get_particle(species = 'beam', iteration = 0, var_list = ['uy']))[0,:])
        # get gamma
        gamma_initial = np.mean(np.sqrt(1+ pux_initial**2 + puz_initial**2 + puy_initial**2))
        # iterate through steps and calculate at each output
        for i in range(int(self.num_steps+1)): 
            if i % self.output == 0:
                pux =  np.array(ts.get_particle(species = 'beam', iteration = i, var_list = ['ux']))[0,:]
                puz =  np.array(ts.get_particle(species = 'beam', iteration = i, var_list = ['uz']))[0,:]        
                puy =  np.array(ts.get_particle(species = 'beam', iteration = i, var_list = ['uy']))[0,:]    
                gamma = np.mean(np.sqrt(1+ pux**2 + puz**2 + puy**2))             
                pux = pux/puz
                phase_advances[int(i/self.output)] = np.sqrt(2)*(np.sqrt(gamma) - np.sqrt(gamma_initial)) * wave_breaking
                x = np.array(ts.get_particle(species='beam', iteration=i, var_list=['x']))[0,:]
                beta_matched = np.sqrt(2*gamma)/k_p(self.plasma_density)
                amplitudes[int(i/self.output)] = np.sqrt(np.mean(puz*((x**2)/beta_matched + (pux**2)*beta_matched)))
        self.__amplitude_evol = phase_advances, amplitudes
        
        
    
    def get_amplitudes(self):
        if self.__amplitude_evol is None:
            print('No amplitudes registered')
        advs, amplitudes = self.__amplitude_evol
        return advs, amplitudes
            
    def __get_initial_driver(self):
        if self.__initial_driver is not None:
            return self.__initial_driver
        else:
            return self.driver_source.track()
        
    def plot_wakefield(self, beam=None):
        
        # extract wakefield if not already existing
        if (self.__initial_wakefield is None) or (self.__final_wakefield is None):
            print('No wakefield')
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
        
    
    def __extract_focusing(self, path):
        # prepare to read simulation data
        ts = OpenPMDTimeSeries(path)

        x_w, z_w = self.__initial_witness
        x_d, z_d =  ts.get_particle(species = 'driver', iteration = 0, var_list = ['x', 'z'])
        
        x_w_f, z_w_f = self.__final_witness
        x_d_f, z_d_f =  self.__final_driver

        # Get initial tail and head 
        tail = np.mean(z_w) - 3*np.std(z_w)
        head = np.mean(z_w) + 3*np.std(z_w)
        
        # Get final tail and head
        tail_f = np.mean(z_w_f) - 3*np.std(z_w_f)
        head_f = np.mean(z_w_f) + 3*np.std(z_w_f)

        # Extract field data
        F_trans, m_trans = ts.get_field('ExmBy', iteration = 0)
        
        # Extract final field data
        F_trans_f, m_trans_f = ts.get_field('ExmBy', iteration = self.num_steps)
        
        # Get head and tail index 
        index_witness_tail= int(np.round((tail - m_trans.z[0])/m_trans.dz))
        index_witness_head= int(np.round((head - m_trans.z[0])/m_trans.dz))
        F_trans = F_trans[index_witness_tail:index_witness_head, :]
        
        # Get final head and tail index 
        index_witness_tail_f= int(np.round((tail_f - m_trans_f.z[0])/m_trans_f.dz))
        index_witness_head_f= int(np.round((head_f - m_trans_f.z[0])/m_trans_f.dz))
        F_trans_f = F_trans_f[index_witness_tail_f:index_witness_head_f, :]
        
        # Set focusing fields
        self.__initial_focusing = F_trans, m_trans.x
        self.__final_focusing = F_trans_f, m_trans_f.x

    def plot_transverse(self, beam=None):

        # extract wakefield if not already existing
        if (self.__initial_transverse is None) or (self.__final_transverse is None):
            return

        # assign to variables
        z0, E0 = self.__initial_transverse
        z, E = self.__final_transverse

        # plot it
        fig, ax1 = plt.subplots(figsize = (7,6))
        fig.set_figwidth(CONFIG.plot_width_default)
        fig.set_figheight(9)

        initial_w = ax1.plot(z0*1e6, E0, label = 'Initial Transverse Wakefield', color = 'orange')
        final_w = ax1.plot(z*1e6, E, label = 'Final Transverse Wakefield')
        plt.xlabel('z (um)')
        plt.ylabel(r'$E_{x} - cB_{y}'' (um)$')
        
        plt.legend()
        
        return 
    
    def plot_initial_bubble(self, beam=None, savefig = None):
        # extract density if not already existing
        if (self.__initial_rho is None):
            print('Charge density not extracted')
            return
        
        # extract wakefield if not already existing
        if (self.__initial_wakefield is None):
            print('No wakefield available')
            return 

        # assign to variables
        zs0, Ezs0 = self.__initial_wakefield
        extent, rho0, jz0, j0extent = self.__initial_rho
        Is0 = self.__initial_driver.peak_current()
        
        # calculate densities and extents
        bubble_radius = blowout_radius(self.plasma_density, Is0)
        extent = np.array([extent[2], extent[3], extent[0], extent[1]])*1e6
        j0extent = np.array([j0extent[2], j0extent[3], j0extent[0], j0extent[1]])*1e6
        charge_density0 = -jz0/(SI.c * SI.e)
        rho0 = -(rho0/(SI.e) -self.plasma_density)
        
        # make figures
        fig, ax = plt.subplots(figsize = (8,7))
        ax2 = ax.twinx()
        ax2.plot(zs0*1e6, Ezs0/1e9, color = 'black')
        ax2.set_ylabel(r'$E_{z}$' ' (GV/m)')
        ax2.set_ylim(bottom=-1.3*max(Ezs0)/1e9, top=1.3*max(Ezs0)/1e9)
        axpos = ax.get_position()
        pad_fraction = 0.1  # Fraction of the figure width to use as padding between the ax and colorbar
        cbar_width_fraction = 0.03  # Fraction of the figure width for the colorbar width

        # Create colorbar axes based on the relative position and size
        cax1 = fig.add_axes([axpos.x1 + pad_fraction, axpos.y0, cbar_width_fraction, axpos.height])
        cax2 = fig.add_axes([axpos.x1 + pad_fraction + cbar_width_fraction, axpos.y0, cbar_width_fraction, axpos.height])
        clims = np.array([1e-2, 1e3])*self.plasma_density
        
        # plasma electrons
        initial = ax.imshow(rho0.T/1e6, extent=extent, norm=LogNorm(), origin='lower', cmap = 'Blues', alpha = np.array(rho0.T>clims.min()*2, dtype = float))
        cb = fig.colorbar(initial, cax=cax1)
        initial.set_clim(clims/1e6)
        cb.ax.tick_params(axis='y',which='both', direction='in')
        cb.set_ticklabels([])
        
        # beam electrons
        charge_density_plot0 = ax.imshow(charge_density0.T/1e6, extent=j0extent, norm=LogNorm(), origin='lower', cmap='Oranges', alpha = np.array(charge_density0.T>clims.min()*2, dtype = float))
        cb2 = fig.colorbar(charge_density_plot0, cax = cax2)
        cb2.set_label(label=r'Electron density ' + r'$\mathrm{cm^{-3}}$',size=10)
        cb2.ax.tick_params(axis='y',which='both', direction='in')
        charge_density_plot0.set_clim(clims/1e6)

        # Set labels
        ax.set_xlabel('z (um)')
        ax.set_ylabel('x (um)')
        
        ax.grid(False)
        ax2.grid(False)
        ylims = np.array([-1, 1])*bubble_radius*1.4
        ax.set_ylim(ylims*1e6)
        
        left = axpos.x0  # Left spacing is the x start of the main axes
        right = 1 - (cax2.get_position().x1 / fig.get_size_inches()[0])  # Right spacing is the end of the second colorbar axis, adjusted for figure size
        top = axpos.y1  # Top spacing is the y end of the main axes
        bottom = axpos.y0  # Bottom spacing is the y start of the main axes

        # Apply the calculated spacings
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
        if savefig is not None:
            fig.savefig(str(savefig), bbox_inches='tight', dpi = 1000)
        return 
    
    
    def plot_final_bubble(self, beam=None):
        # extract density if not already existing
        if (self.__final_rho is None):
            print('Charge density not extracted')
            return
        
        # extract wakefield if not already existing
        if (self.__final_wakefield is None):
            print('No wakefield available')
            return 

        # assign to variables
        zs0, Ezs0 = self.__final_wakefield
        extent, rho0, jz0, j0extent = self.__final_rho
        Is0 = self.__initial_driver.peak_current()
        
        # calculate densities and extents
        bubble_radius = blowout_radius(self.plasma_density, Is0)
        extent = np.array([extent[2], extent[3], extent[0], extent[1]])*1e6
        j0extent = np.array([j0extent[2], j0extent[3], j0extent[0], j0extent[1]])*1e6
        charge_density0 = -jz0/(SI.c * SI.e)
        rho0 = -(rho0/(SI.e) -self.plasma_density)
        
        # make figures
        fig, ax = plt.subplots(figsize = (8,8))
        ax2 = ax.twinx()
        ax2.plot(zs0*1e6, Ezs0/1e9, color = 'black')
        ax2.set_ylabel(r'$E_{z}$' ' (GV/m)')
        ax2.set_ylim(bottom=-wave_breaking_field(self.plasma_density)/1e9, top=1.3*max(Ezs0)/1e9)
        axpos = ax.get_position()
        pad_fraction = 0.1  # Fraction of the figure width to use as padding between the ax and colorbar
        cbar_width_fraction = 0.03  # Fraction of the figure width for the colorbar width

        # Create colorbar axes based on the relative position and size
        cax1 = fig.add_axes([axpos.x1 + pad_fraction, axpos.y0, cbar_width_fraction, axpos.height])
        cax2 = fig.add_axes([axpos.x1 + pad_fraction + cbar_width_fraction, axpos.y0, cbar_width_fraction, axpos.height])
        clims = np.array([1e-2, 1e3])*self.plasma_density
        
        # plasma electrons
        initial = ax.imshow(rho0.T/1e6, extent=extent, norm=LogNorm(), origin='lower', cmap = 'Blues', alpha = np.array(rho0.T>clims.min()*2, dtype = float))
        cb = plt.colorbar(initial, cax=cax1)
        initial.set_clim(clims/1e6)
        cb.ax.tick_params(axis='y',which='both', direction='in')
        cb.set_ticklabels([])
        
        # beam electrons
        charge_density_plot0 = ax.imshow(charge_density0.T/1e6, extent=j0extent, norm=LogNorm(), origin='lower', cmap='Oranges', alpha = np.array(charge_density0.T>clims.min()*2, dtype = float))
        cb2 = plt.colorbar(charge_density_plot0, cax = cax2)
        cb2.set_label(label=r'Electron density ' + r'$\mathrm{cm^{-3}}$',size=10)
        cb2.ax.tick_params(axis='y',which='both', direction='in')
        charge_density_plot0.set_clim(clims/1e6)

        # Set labels
        ax.set_xlabel('z (um)')
        ax.set_ylabel('x (um)')
        
        ax.grid(False)
        ax2.grid(False)
        ylims = np.array([-1, 1])*bubble_radius*1.4
        ax.set_ylim(ylims*1e6)
        return 
    
    def get_transverse_sliced(self):
        zs0, E0 = self.__initial_transverse
        E_tail = E0[0]
        return E_tail

    def get_normalised_initial_focusing(self):
        focusing, m_trans_init = self.__initial_focusing
        fractional_bubble = int(len(m_trans_init)/20)
        focusing_normalised = (focusing[-1, 11*fractional_bubble]-focusing[-1,9*fractional_bubble])/(m_trans_init[11*fractional_bubble]- m_trans_init[9*fractional_bubble]) 
        return focusing_normalised
    
    def get_rms_accel_initial(self):
        x_w, z_w = self.__initial_witness
        z, Ez0 = self.__initial_wakefield
        dz = np.median(np.diff(z))
        tail = np.mean(z_w) - 3*np.std(z_w)
        head = np.mean(z_w) + 3*np.std(z_w)
        
        index_witness_tail= int(np.round((tail - z[0])/dz))
        index_witness_head= int(np.round((head - z[0])/dz))
        
        E_z_rms = np.mean(Ez0[index_witness_tail:index_witness_head])
        return E_z_rms
    
