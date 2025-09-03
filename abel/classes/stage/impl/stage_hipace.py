from abel.CONFIG import CONFIG
from abel.classes.stage.stage import Stage
from abel.utilities.plasma_physics import *
from abel.utilities.relativity import energy2gamma
import scipy.constants as SI
import numpy as np
import os, shutil, uuid, copy
from abel.utilities.plasma_physics import k_p
from types import SimpleNamespace

#try:
#    import sys
#    sys.path.append(os.path.join(CONFIG.hipace_path, 'tools'))
#    import read_insitu_diagnostics
#except:
#    print(f"Could not import HiPACE++ tools from {os.path.join(CONFIG.hipace_path, 'tools')}")

class StageHipace(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=None, keep_data=False, save_drivers=False, output=None, ion_motion=True, ion_species='H', beam_ionization=True, radiation_reaction=False, num_nodes=1, num_cell_xy=511, driver_only=False, plasma_density_from_file=None, no_plasma=False, external_focusing=False, mesh_refinement=True, do_spin_tracking=False):
        
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

        # external focusing (APL-like) [T/m]
        self.external_focusing = external_focusing
        self._external_focusing_gradient = None

        # plasma profile
        self.plasma_profile = SimpleNamespace()
        self.plasma_profile.ss = None
        self.plasma_profile.ns = None

        # physics flags
        self.ion_motion = ion_motion
        self.ion_species = ion_species
        self.mesh_refinement = mesh_refinement
        self.beam_ionization = beam_ionization
        self.radiation_reaction = radiation_reaction
        self.do_spin_tracking = do_spin_tracking
        

    def track(self, beam_incoming, savedepth=0, runnable=None, verbose=False):

        from abel.apis.hipace.hipace_api import hipace_write_inputs, hipace_run, hipace_write_jobscript
        
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

        # Set ramp lengths, nominal energies, nominal energy gains
        # and flattop nominal energy if not already done
        self._prepare_ramps()
        self._make_ramp_profile(tmpfolder)
        
        # set external focusing
        if self.external_focusing == False:
            self.external_focusing_gradient = 0
        if self.external_focusing == True and self._external_focusing_gradient is None:
            num_half_oscillations = 1
            self._external_focusing_gradient = self.driver_source.energy/SI.c*(num_half_oscillations*np.pi/self.get_length())**2
        
        beam0 = beam_incoming
        driver0 = driver_incoming
        
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
            if not os.path.exists(tmpfolder + density_table_file):
                shutil.copyfile(self.plasma_density_from_file, tmpfolder + density_table_file)

            #self.length = self.get_length() # TODO: ensure that the length from a density profile is correct
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
        box_size_xy = 2*np.max([4/k_p(self.plasma_density), 2*blowout_radius(self.plasma_density, driver0.peak_current())])
        
        # calculate number of cells in x to get similar resolution
        dr = box_size_xy/self.num_cell_xy
        if self.mesh_refinement:
            num_cell_z = 2*round((box_max_z-box_min_z)/dr)
        else:
            num_cell_z = round((box_max_z-box_min_z)/dr)
        
        # calculate the time step
        beta_matched = np.sqrt(2*min(beam0.gamma(),driver0.gamma()/2))/k_p(self.plasma_density)
        dz = beta_matched/20
        
        # convert to number of steps (and re-adjust timestep to be divisible)
        self.num_steps = np.ceil(self.length_flattop/dz)
        
        if self.output is not None:
            remainder = self.num_steps % self.output
            if remainder >= self.output/2:  # If remainder is 10 or greater, round up
                self.num_steps = self.num_steps + (self.output - remainder)
            else:  # If remainder is less than 10, round down
                self.num_steps = self.num_steps - remainder
        
        #time_step = self.length_flattop/(self.num_steps*SI.c)
        time_step = self.length/(self.num_steps*SI.c)

        # overwrite output period
        if self.output is not None:
            output_period = self.output
        else:
            output_period = None
        
        # input file
        filename_input = 'input_file'
        path_input = tmpfolder + filename_input
        hipace_write_inputs(path_input, filename_beam, filename_driver, self.plasma_density, self.num_steps, time_step, box_range_z, box_size_xy, ion_motion=self.ion_motion, ion_species=self.ion_species, beam_ionization=self.beam_ionization, radiation_reaction=self.radiation_reaction, output_period=output_period, num_cell_xy=self.num_cell_xy, num_cell_z=num_cell_z, driver_only=self.driver_only, density_table_file=density_table_file, no_plasma=self.no_plasma, external_focusing_gradient=self._external_focusing_gradient, mesh_refinement=self.mesh_refinement, do_spin_tracking=self.do_spin_tracking)
        
        
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
        beam.location = beam_incoming.location
        driver.trackable_number = beam_incoming.trackable_number
        driver.stage_number = beam_incoming.stage_number
        driver.location = beam_incoming.location

        beam_outgoing = beam
        driver_outgoing = driver
        
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

        # reset location 
        beam_outgoing.location = beam_incoming.location
        
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

        from abel.apis.hipace import read_insitu_diagnostics
        
        # suppress divide-by-zero errors
        np.seterr(divide='ignore', invalid='ignore')

        # in-situ data path
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

            # add spin information
            if '[sx]' in np.dtype(average_data.dtype).names:
                evol.spin_x = average_data['[sx]']
                evol.spin_y = average_data['[sy]']
                evol.spin_z = average_data['[sz]']
            else:
                evol.spin_x = None
                evol.spin_y = None
                evol.spin_z = None
            
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
                        import scipy.stats as spstats
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

            # peak current
            slice_charges_tuple = (evol.slices.charge[:, :-1] + evol.slices.charge[:, 1:])/2
            slice_thicknesses_tuple = np.diff(evol.slices.z)
            slice_zs_tuple = (evol.slices.z[:, :-1] + evol.slices.z[:, 1:])/2
            slice_currents_tuple = np.sign(slice_charges_tuple)*slice_charges_tuple*SI.c/slice_thicknesses_tuple
            peak_currents = np.max(np.stack(slice_currents_tuple), axis=1)
            evol.peak_current = peak_currents
            
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

        from openpmd_viewer import OpenPMDTimeSeries
        
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

    
    def _make_ramp_profile(self, tmpfolder):
        """Prepare the ramps (local to HiPACE)."""
        
        # check that there is not already a plasma density profile set
        assert self.plasma_density_from_file is None

        # make the plasma ramp profile
        if self.has_ramp():

            ss_upramp = np.linspace(0, self.upramp.length, 100)
            if self.upramp.ramp_shape == 'uniform':
                ns_upramp = np.ones_like(ss_upramp)*self.upramp.plasma_density
            
            ss_flattop = max(ss_upramp)+np.linspace(0, self.length_flattop, 100)
            ns_flattop = np.ones_like(ss_flattop)*self.plasma_density
                
            ss_downramp = max(ss_flattop)+np.linspace(0, self.downramp.length, 100)
            if self.downramp.ramp_shape == 'uniform':
                ns_downramp = np.ones_like(ss_downramp)*self.downramp.plasma_density

            ss = np.concatenate((ss_upramp, ss_flattop, ss_downramp), axis=0)
            ns = np.concatenate((ns_upramp, ns_flattop, ns_downramp), axis=0)

            # save to file
            self.plasma_profile.ss = ss
            self.plasma_profile.ns = ns
            
            # save to file
            density_table = np.column_stack((ss, ns))
            filename = os.path.join(tmpfolder, 'plasma_profile.txt')
            np.savetxt(filename, density_table, delimiter=" ")
            self.plasma_density_from_file = filename
            
        
    def plot_plasma_density_profile(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1,1)
        ax.plot(self.plasma_profile.ss, self.plasma_profile.ns/1e6, '-')
        ax.set_xlim(min(self.plasma_profile.ss), max(self.plasma_profile.ss))
        ax.set_yscale('log')
        ax.set_xlabel('Longitudinal position (m)')
        ax.set_ylabel(r'Plasma density (cm$^{-3}$)')
        

    def get_plasma_density(self, locations=None):
        if self.plasma_density_from_file is not None:
            ns = self.plasma_profile.ns
            self.plasma_density = ns.max()
            if locations is not None:
                ss = self.plasma_profile.ss
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
            #density_table = np.loadtxt(self.plasma_density_from_file, delimiter=" ", dtype=float)
            ss = self.plasma_profile.ss
            #ss = density_table[:,0]
            return ss.max()-ss.min()
        return super().get_length()
