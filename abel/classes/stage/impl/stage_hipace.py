from abel import Stage, CONFIG
from abel.apis.hipace.hipace_api import hipace_write_inputs, hipace_run, hipace_write_jobscript
from abel.utilities.plasma_physics import *
from abel.utilities.relativity import energy2gamma
import scipy.constants as SI
import scipy.stats as spstats
from matplotlib import pyplot as plt
import numpy as np
import os, shutil, uuid, copy
from openpmd_viewer import OpenPMDTimeSeries
from abel.utilities.plasma_physics import k_p
from matplotlib.colors import LogNorm
from types import SimpleNamespace

try:
    import sys
    sys.path.append(os.path.join(CONFIG.hipace_path, 'tools'))
    import read_insitu_diagnostics
except:
    print('Could not import HiPACE++ tools')

class StageHipace(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=None, keep_data=False, save_drivers=False, output=None, ion_motion=True, ion_species='H', beam_ionization=True, radiation_reaction=False, num_nodes=1, num_cell_xy=511, driver_only=False, plasma_density_from_file=None, no_plasma=False):
        
        super().__init__(length, nom_energy_gain, plasma_density, driver_source, ramp_beta_mag)
        
        # simulation specifics
        self.keep_data = keep_data
        self.output = output
        self.num_nodes = num_nodes
        self.num_cell_xy = num_cell_xy
        self.driver_only = driver_only
        self.plasma_density_from_file = plasma_density_from_file
        self.save_drivers = save_drivers
        self.no_plasma = no_plasma

        # physics flags
        self.ion_motion = ion_motion
        self.ion_species = ion_species
        self.beam_ionization = beam_ionization
        self.radiation_reaction = radiation_reaction
        
        self.__initial_driver = None
        self.__final_driver = None
        self.__initial_transverse = None
        self.__final_transverse = None
        self.__bubble_size = None
        self.__initial_witness = None
        self.__final_witness = None
        self.__final_focusing = None
        self.__initial_focusing = None
        self.__amplitude_evol = None
        self.__phase_advances = None
        

    def track(self, beam_incoming, savedepth=0, runnable=None, verbose=False):
        
        ## PREPARE TEMPORARY FOLDER
        
        # make temp folder
        if not os.path.exists(CONFIG.temp_path):
            os.mkdir(CONFIG.temp_path)
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
        
        # make directory
        if not os.path.exists(tmpfolder):
            os.mkdir(tmpfolder)
        
        # generate driver
        driver_incoming = self.driver_source.track()

        # plasma-density ramps (de-magnify beta function)
        location_flattop_start = 0
        if self.upramp is not None:
            beam0, driver0 = self.track_upramp(beam_incoming, driver_incoming)
            location_flattop_start = beam0.location
        else:
            # apply plasma-density up ramp (demagnify beta function)
            driver0 = copy.deepcopy(driver_incoming)
            beam0 = copy.deepcopy(beam_incoming)
            if self.ramp_beta_mag is not None:
                driver0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver_incoming)
                beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver_incoming)
        
        beam0.location = 0.0
        driver0.location = 0.0
        
        # SAVE BEAMS TO BE USED IN HiPACE++ SIMULATION
        
        # saving beam to temporary folder
        filename_beam = 'beam.h5'
        path_beam = tmpfolder + filename_beam
        beam0.save(filename = path_beam)
        
        # produce and save drive beam
        filename_driver = 'driver.h5'
        path_driver = tmpfolder + filename_driver
        driver0.save(filename = path_driver, beam_name = 'driver')

        # make directory
        if self.plasma_density_from_file is not None:
            density_table_file = os.path.basename(self.plasma_density_from_file)
            shutil.copyfile(self.plasma_density_from_file, tmpfolder + density_table_file)

            self.length = self.get_length()
            self.plasma_density = self.get_plasma_density()
        else:
            density_table_file = None
        
        
        # MAKE INPUT FILE
        
        # make longitudinal box range
        num_sigmas = 6
        if self.driver_only:
            box_min_z = driver0.z_offset() + driver0.bunch_length() - np.max([2*np.pi/k_p(self.plasma_density), 2.1*blowout_radius(self.plasma_density, driver0.peak_current())])
        else:
            box_min_z = beam0.z_offset() - num_sigmas * beam0.bunch_length() 
        box_min_z = box_min_z - 1.5/k_p(self.plasma_density)
        box_max_z = min(driver0.z_offset() + num_sigmas * driver0.bunch_length(), np.max(driver0.zs())+0.5/k_p(self.plasma_density))
        box_range_z = [box_min_z, box_max_z]
        
        # making transverse box size
        box_size_r = 2*np.max([4/k_p(self.plasma_density), 2*blowout_radius(self.plasma_density, driver0.peak_current())])
        
        # calculate number of cells in x to get similar resolution
        dr = box_size_r/self.num_cell_xy
        num_cell_z = round((box_max_z-box_min_z)/dr)
        
        # calculate the time step
        beta_matched = np.sqrt(2*min(beam0.gamma(),driver0.gamma()/2))/k_p(self.plasma_density)
        dz = beta_matched/20
        
        # convert to number of steps (and re-adjust timestep to be divisible)
        self.num_steps = np.ceil(self.get_length()/dz)
        
        if self.output is not None:
            remainder = self.num_steps % self.output
            if remainder >= self.output/2:  # If remainder is 10 or greater, round up
                self.num_steps = self.num_steps + (self.output - remainder)
            else:  # If remainder is less than 10, round down
                self.num_steps = self.num_steps - remainder
        
        time_step = self.get_length()/(self.num_steps*SI.c)

        # overwrite output period
        if self.output is not None:
            output_period = self.output
        else:
            output_period = None
        
        # input file
        filename_input = 'input_file'
        path_input = tmpfolder + filename_input
        hipace_write_inputs(path_input, filename_beam, filename_driver, self.plasma_density, self.num_steps, time_step, box_range_z, box_size_r, ion_motion=self.ion_motion, ion_species=self.ion_species, beam_ionization=self.beam_ionization, radiation_reaction=self.radiation_reaction, output_period=output_period, num_cell_xy=self.num_cell_xy, num_cell_z=num_cell_z, driver_only=self.driver_only, density_table_file=density_table_file, no_plasma=self.no_plasma)
        
        
        ## RUN SIMULATION
        
        # make job script
        filename_job_script = tmpfolder + 'run.sh'
        hipace_write_jobscript(filename_job_script, filename_input, num_nodes=self.num_nodes)
        
        # run HiPACE++
        beam, driver = hipace_run(filename_job_script, self.num_steps)
        if self.driver_only:
            beam = beam0
        
        ## ADD METADATA
        
        # copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam_incoming.trackable_number
        beam.stage_number = beam_incoming.stage_number
        beam.location = location_flattop_start + beam0.location
        driver.trackable_number = beam_incoming.trackable_number
        driver.stage_number = beam_incoming.stage_number
        driver.location = beam.location
        
        # apply plasma-density down ramp (magnify beta function)
        if self.downramp is not None:
            beam_outgoing, driver_outgoing = self.track_downramp(beam, driver)
        else:
            beam_outgoing = copy.deepcopy(beam)
            driver_outgoing = copy.deepcopy(driver)
            if self.ramp_beta_mag is not None:
                beam_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver_incoming)
                driver_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver_incoming)
        
        ## SAVE DRIVERS TO FILE
        if self.save_drivers:
            driver_incoming.stage_number = beam_incoming.stage_number
            driver_incoming.location = beam_incoming.location
            driver_incoming.trackable_number = 0
            self.save_driver_to_file(driver_incoming, runnable)
            driver_outgoing.stage_number = beam_outgoing.stage_number
            driver_outgoing.location = beam_outgoing.location
            driver_outgoing.trackable_number = 1
            self.save_driver_to_file(driver_outgoing, runnable)
        
        # clean nan particles and extreme outliers
        beam_outgoing.remove_nans()
        beam_outgoing.remove_halo_particles()

        # extract insitu diagnostics and wakefield data
        self.__extract_evolution(tmpfolder, beam0, runnable)
        self.__extract_initial_and_final_step(tmpfolder, beam0, runnable)
        
        # delete temp folder
        shutil.rmtree(tmpfolder)
        
        # calculate efficiency
        self.calculate_efficiency(beam_incoming, driver_incoming, beam_outgoing, driver_outgoing)
        
        # save current profile
        self.calculate_beam_current(beam_incoming, driver_incoming, beam_outgoing, driver_outgoing)

        # return the beam (and optionally the driver)
        if self._return_tracked_driver:
            return super().track(beam_outgoing, savedepth, runnable, verbose), driver_outgoing
        else:
            return super().track(beam_outgoing, savedepth, runnable, verbose)
    
    
    def __extract_evolution(self, tmpfolder, beam0, runnable):

        insitu_path = tmpfolder + 'diags/insitu/'

        bunches = ['beam','driver']

        for bunch in bunches:

            # skip for beam if driver only
            if bunch == 'beam' and self.driver_only:
                continue
            
            # extract in-situ data
            insitu_file = insitu_path + 'reduced_' + bunch + '.*.txt'
            all_data = read_insitu_diagnostics.read_file(insitu_file)
            average_data = all_data['average']
            
            # store variables
            evol = SimpleNamespace()
            evol.slices = SimpleNamespace()
            
            evol.location = beam0.location + all_data['time']*SI.c
            evol.charge = read_insitu_diagnostics.total_charge(all_data)
            evol.energy = read_insitu_diagnostics.energy_mean_eV(all_data)
            evol.z = average_data['[z]']
            evol.x = average_data['[x]']
            evol.y = average_data['[y]']
            evol.xp = average_data['[ux]']/average_data['[uz]']
            evol.yp = average_data['[uy]']/average_data['[uz]']
            evol.energy_spread = read_insitu_diagnostics.energy_spread_eV(all_data)
            evol.rel_energy_spread = evol.energy_spread/evol.energy
            evol.beam_size_x = read_insitu_diagnostics.position_std(average_data, direction='x')
            evol.beam_size_y = read_insitu_diagnostics.position_std(average_data, direction='y')
            evol.bunch_length = read_insitu_diagnostics.position_std(average_data, direction='z')
            evol.emit_nx = read_insitu_diagnostics.emittance_x(average_data)
            evol.emit_ny = read_insitu_diagnostics.emittance_y(average_data)
            evol.plasma_density = self.get_plasma_density(evol.location)
            
            # beta functions (temporary fix)
            evol.beta_x = evol.beam_size_x**2*(evol.energy/0.5109989461e6)/evol.emit_nx # TODO: improve with x-x' correlations instead of x-px
            evol.beta_y = evol.beam_size_y**2*(evol.energy/0.5109989461e6)/evol.emit_ny # TODO: improve with y-y' correlations instead of y-py

            # slice parameters
            slice_mask = abs(read_insitu_diagnostics.per_slice_charge(all_data)[0,:]) > SI.e
            evol.slices.charge = read_insitu_diagnostics.per_slice_charge(all_data)[:,slice_mask]
            evol.slices.energy = read_insitu_diagnostics.energy_mean_eV(all_data, per_slice=True)[:,slice_mask]
            evol.slices.z = all_data['[z]'][:,slice_mask]
            evol.slices.x = all_data['[x]'][:,slice_mask]
            evol.slices.y = all_data['[y]'][:,slice_mask]
            evol.slices.xp = all_data['[ux]'][:,slice_mask]/all_data['[uz]'][:,slice_mask]
            evol.slices.yp = all_data['[uy]'][:,slice_mask]/all_data['[uz]'][:,slice_mask]
            evol.slices.energy_spread = read_insitu_diagnostics.energy_spread_eV(all_data, per_slice=True)[:,slice_mask]
            evol.slices.rel_energy_spread = evol.slices.energy_spread/evol.slices.energy
            evol.slices.beam_size_x = read_insitu_diagnostics.position_std(all_data, direction='x')[:,slice_mask]
            evol.slices.beam_size_y = read_insitu_diagnostics.position_std(all_data, direction='y')[:,slice_mask]
            evol.slices.bunch_length = read_insitu_diagnostics.position_std(all_data, direction='z')[:,slice_mask]
            evol.slices.emit_nx = read_insitu_diagnostics.emittance_x(all_data)[:,slice_mask]
            evol.slices.emit_ny = read_insitu_diagnostics.emittance_y(all_data)[:,slice_mask]

            # energy spectrum
            calculate_spectral_info = False
            evol.peak_spectral_density = np.empty_like(evol.location)
            evol.energy_spread_fwhm = np.empty_like(evol.location)
            evol.rel_energy_spread_fwhm = np.empty_like(evol.location)
            if calculate_spectral_info:
                for step in range(len(evol.location)):
                    Es = np.linspace(np.min(evol.energy[step]-5*evol.energy_spread[step]), np.max(evol.energy[step]+5*evol.energy_spread[step]), 500)
                    dQ_dE = np.zeros_like(Es)
                    for i in range(len(evol.slices.charge[step,:])):
                        dQ_dE_slice = spstats.norm.pdf(Es, loc=evol.slices.energy[step,i], scale=evol.slices.energy_spread[step,i])
                        Q_slice = np.trapz(dQ_dE_slice, x=Es)
                        if abs(Q_slice) > 0:
                            dQ_dE_slice = dQ_dE_slice*evol.slices.charge[step,i]/np.trapz(dQ_dE_slice, x=Es)
                            dQ_dE += dQ_dE_slice
                                    
                    dQ_dE_max = np.max(abs(dQ_dE))
                    evol.peak_spectral_density[step] = dQ_dE_max
                    evol.energy_spread_fwhm[step] = np.max(Es[abs(dQ_dE) > dQ_dE_max*0.5]) - np.min(Es[abs(dQ_dE) > dQ_dE_max*0.5])
            else:
                evol.peak_spectral_density[:] = np.nan
                evol.energy_spread_fwhm[:] = np.nan
            
            evol.rel_energy_spread_fwhm = evol.energy_spread_fwhm/evol.energy
            
            # assign it
            if bunch == 'beam':
                self.evolution.beam = evol
            elif bunch == 'driver':
                self.evolution.driver = evol
            
            # TODO: add divergences
            # TODO: add beta functions, alpha functions
            # TODO: add angular momentum 
            # TODO: add normalized amplitude

        # delete or move data
        if self.keep_data:
            destination_path = runnable.shot_path() + 'stage_' + str(beam0.stage_number) + '/insitu'
            shutil.move(insitu_path, destination_path)
        
        
    def __extract_initial_and_final_step(self, tmpfolder, beam0, runnable):
        
        # prepare to read simulation data
        source_path = tmpfolder + 'diags/hdf5/'
        ts = OpenPMDTimeSeries(source_path)

        # extract initial on-axis wakefield
        Ez0, metadata0 = ts.get_field(field='Ez', slice_across=['x'], iteration=min(ts.iterations))
        self.initial.plasma.wakefield.onaxis.zs = metadata0.z
        self.initial.plasma.wakefield.onaxis.Ezs = Ez0
        
        # extract final on-axis wakefield
        Ez, metadata = ts.get_field(field='Ez', slice_across=['x'], iteration=max(ts.iterations))
        self.final.plasma.wakefield.onaxis.zs = metadata.z
        self.final.plasma.wakefield.onaxis.Ezs = Ez
        
        # extract initial plasma density
        rho0_plasma, metadata0_plasma = ts.get_field(field='rho', iteration=min(ts.iterations))
        self.initial.plasma.density.extent = metadata0_plasma.imshow_extent[[2,3,0,1]]
        self.initial.plasma.density.rho = -(rho0_plasma.T/SI.e-self.plasma_density)
 
        # extract final beam density
        jz0_beam, metadata0_beam = ts.get_field(field='jz_beam', iteration=min(ts.iterations))
        self.initial.beam.density.extent = metadata0_beam.imshow_extent[[2,3,0,1]]
        self.initial.beam.density.rho = -jz0_beam.T/(SI.c*SI.e)

        # extract initial plasma density
        rho_plasma, metadata_plasma = ts.get_field(field='rho', iteration=max(ts.iterations))
        self.final.plasma.density.extent = metadata_plasma.imshow_extent[[2,3,0,1]]
        self.final.plasma.density.rho = -(rho_plasma.T/SI.e-self.plasma_density)

        # extract final beam density
        jz_beam, metadata_beam = ts.get_field(field='jz_beam', iteration=max(ts.iterations))
        self.final.beam.density.extent = metadata_beam.imshow_extent[[2,3,0,1]]
        self.final.beam.density.rho = -jz_beam.T/(SI.c*SI.e)
        
        # delete or move data
        if self.keep_data:
            destination_path = runnable.shot_path() + 'stage_' + str(beam0.stage_number)
            shutil.move(source_path, destination_path)

        
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

    
    def get_plasma_density(self, locations=None):
        if self.plasma_density_from_file is not None:
            density_table = np.loadtxt(self.plasma_density_from_file, delimiter=" ", dtype=float)
            ns = density_table[:,1]
            self.plasma_density = ns.max()
            if locations is not None:
                ss = density_table[:,0]
                return np.interp(locations, ss, ns)
            else:
                return self.plasma_density
        else:
            if locations is not None:
                return self.plasma_density*np.ones(locations.shape)
            else:
                return self.plasma_density

    def get_length(self):
        if self.plasma_density_from_file is not None:
            density_table = np.loadtxt(self.plasma_density_from_file, delimiter=" ", dtype=float)
            ss = density_table[:,0]
            self.length = ss.max()-ss.min()
        return super().get_length()
    
    def get_amplitudes(self):
        if self.__amplitude_evol is None:
            print('No amplitudes registered')
        advs, amplitudes = self.__amplitude_evol
        return advs, amplitudes
    
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
    
