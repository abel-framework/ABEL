from abel import Stage, Beam, CONFIG
import scipy.constants as SI
from matplotlib import pyplot as plt
from abel.utilities.plasma_physics import blowout_radius, k_p, beta_matched
from abel.apis.wake_t.wake_t_api import beam2wake_t_bunch, wake_t_bunch2beam
import os, shutil, uuid
import numpy as np
import wake_t
from openpmd_viewer import OpenPMDTimeSeries

class StageWakeT(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None, drive_beam=None, ramp_beta_mag=1, num_cell_xy=256, keep_data=False, run_path=None):
        
        super().__init__(length, nom_energy_gain, plasma_density, driver_source)
        
        self.ramp_beta_mag = ramp_beta_mag
        self.num_cell_xy = num_cell_xy
        self.keep_data = keep_data
        self.drive_beam = drive_beam
        self.run_path = run_path


    # ==================================================
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        self.stage_number = beam0.stage_number
        
        # make temp folder
        if not os.path.exists(CONFIG.temp_path):
            os.mkdir(CONFIG.temp_path)
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
        if not os.path.exists(tmpfolder):
            os.mkdir(tmpfolder)

        # make driver (and convert to WakeT bunch)
        if self.drive_beam is None:
            driver0 = self.driver_source.track()
            self.drive_beam = driver0 
        else:
            driver0 = self.drive_beam  # Allows the drive beam to be passed from outside.
        
        # apply plasma-density up ramp (demagnify beta function)
        driver0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver0)
        beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver0)
        
        # convert beams to WakeT bunches
        driver0_wake_t = beam2wake_t_bunch(driver0, name='driver')
        beam0_wake_t = beam2wake_t_bunch(beam0, name='beam')
        
        # make longitudinal box range
        num_sigmas = 6
        #box_min_z = beam0.z_offset() - num_sigmas * beam0.bunch_length()
        R_blowout = blowout_radius(self.plasma_density, driver0.peak_current())
        box_min_z = driver0.z_offset() - 4.0 * R_blowout
        #box_min_z = driver0.z_offset() - 3.3 * R_blowout
        #box_max_z = min(driver0.z_offset() + num_sigmas * driver0.bunch_length(), np.max(driver0.zs())+0.25/k_p(self.plasma_density))
        box_max_z = min(driver0.z_offset() + num_sigmas * driver0.bunch_length(), np.max(driver0.zs()) + 0.5*R_blowout)

        if box_min_z > beam0.zs().min() or box_max_z < driver0.zs().max():
            raise SimulationDomainSizeError('The simulation domain is too small along z.')
        #box_range_z = [box_min_z, box_max_z]
        
        
        # making transverse box size
        box_size_r = np.max([4/k_p(self.plasma_density), 3*blowout_radius(self.plasma_density, driver0.peak_current())])
        
        # calculate number of cells in x to get similar resolution
        dr = box_size_r/self.num_cell_xy
        num_cell_z = round((box_max_z-box_min_z)/dr)
        
        # find stepsize
        matched_beta = np.sqrt(2*min(beam0.gamma(),driver0.gamma()/2))/k_p(self.plasma_density)
        dz = matched_beta/10
        
        n_out = round(self.length/dz/8)
        plasma = wake_t.PlasmaStage(length=self.length, density=self.plasma_density, wakefield_model='quasistatic_2d',
                                    r_max=box_size_r, r_max_plasma=box_size_r, xi_min=box_min_z, xi_max=box_max_z, 
                                    n_out=n_out, n_r=int(self.num_cell_xy), n_xi=int(num_cell_z), dz_fields=dz, ppc=1)
        
        # do tracking
        bunches = plasma.track([driver0_wake_t, beam0_wake_t], opmd_diag=True, diag_dir=tmpfolder, show_progress_bar=verbose)  # v0.8.0
        #bunches = plasma.track([driver0_wake_t, beam0_wake_t], opmd_diag=True, diag_dir=tmpfolder)
        
        # save evolution of the beam and driver
        self.__extract_evolution(bunches)
        self.__extract_initial_and_final_step(tmpfolder)

        # delete or move data
        #if self.keep_data:
        #    shot_path = runnable.shot_path()  # TODO: this does not work yet
        #    destination_path = runnable.shot_path() + 'stage_' + str(bunches[1].stage_number) + '/insitu'
        #    shutil.move(tmpfolder, destination_path)
        
        # remove temporary directory
        shutil.rmtree(tmpfolder)
        
        # extract beams
        beam = wake_t_bunch2beam(bunches[1][-1])
        driver = wake_t_bunch2beam(bunches[0][-1])
        
        # copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        # apply plasma-density down ramp (magnify beta function)
        beam.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver0)
        driver.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver0)
        
        # calculate efficiency
        self.calculate_efficiency(beam0, driver0, beam, driver)
        
        # save current profile
        self.calculate_beam_current(beam0, driver0, beam, driver)
        
        return super().track(beam, savedepth, runnable, verbose)


    # ==================================================
    def __extract_evolution(self, bunches):
    #def __extract_evolution(self, tmpfolder, bunches, runnable):

        # get beam
        beam_evol = wake_t.diagnostics.analyze_bunch_list(bunches[1])
        
        # store variables
        self.evolution.location = beam_evol['prop_dist']
        self.evolution.charge = beam_evol['q_tot']
        self.evolution.energy = beam_evol['avg_ene']*SI.m_e*SI.c**2/SI.e
        #self.evolution.z = beam_evol['z_avg']
        self.evolution.x = beam_evol['x_avg']
        self.evolution.y = beam_evol['y_avg']
        self.evolution.xp = beam_evol['theta_x']
        self.evolution.yp = beam_evol['theta_y']
        self.evolution.rel_energy_spread = beam_evol['rel_ene_spread']
        self.evolution.energy_spread = self.evolution.rel_energy_spread*self.evolution.energy
        self.evolution.beam_size_x = beam_evol['sigma_x']
        self.evolution.beam_size_y = beam_evol['sigma_y']
        self.evolution.bunch_length = beam_evol['sigma_z']
        self.evolution.emit_nx = beam_evol['emitt_x']
        self.evolution.emit_ny = beam_evol['emitt_y']
        self.evolution.beta_x = beam_evol['beta_x']
        self.evolution.beta_y = beam_evol['beta_y']


    # ==================================================
    def __extract_initial_and_final_step(self, tmpfolder):
        
        # prepare to read simulation data
        source_path = tmpfolder + 'hdf5/'
        ts = OpenPMDTimeSeries(source_path)

        # extract initial on-axis wakefield
        Ez0, metadata0_Ez = ts.get_field(field='E', coord='z', slice_across=['r'], iteration=min(ts.iterations))
        self.initial.plasma.wakefield.onaxis.zs = metadata0_Ez.z
        self.initial.plasma.wakefield.onaxis.Ezs = Ez0
        
        # extract final on-axis wakefield
        Ez, metadata_Ez = ts.get_field(field='E', coord='z', slice_across=['r'], iteration=max(ts.iterations))
        self.final.plasma.wakefield.onaxis.zs = metadata_Ez.z
        self.final.plasma.wakefield.onaxis.Ezs = Ez

        # extract initial fields
        Ez0, metadata0_Ez = ts.get_field(field='E', coord='z', iteration=min(ts.iterations), plot=False)
        self.initial.plasma.wakefield.Ezs = Ez0
        self.initial.plasma.wakefield.Ezs_metadata = metadata0_Ez
        Bz0, metadata0_Bz = ts.get_field(field='B', coord='z', iteration=min(ts.iterations), plot=False)
        self.initial.plasma.wakefield.Bzs = Bz0
        self.initial.plasma.wakefield.Bzs_metadata = metadata0_Bz
        Ex0, metadata0_Ex = ts.get_field(field='E', coord='x', iteration=min(ts.iterations), plot=False)
        self.initial.plasma.wakefield.Exs = Ex0
        self.initial.plasma.wakefield.Exs_metadata = metadata0_Ex
        Bx0, metadata0_Bx = ts.get_field(field='B', coord='x', iteration=min(ts.iterations), plot=False)
        self.initial.plasma.wakefield.Bxs = Bx0
        self.initial.plasma.wakefield.Bxs_metadata = metadata0_Bx
        Ey0, metadata0_Ey = ts.get_field(field='E', coord='y', iteration=min(ts.iterations), plot=False)
        self.initial.plasma.wakefield.Eys = Ey0
        self.initial.plasma.wakefield.Eys_metadata = metadata0_Ey
        By0, metadata0_By = ts.get_field(field='B', coord='y', iteration=min(ts.iterations), plot=False)
        self.initial.plasma.wakefield.Bys = By0
        self.initial.plasma.wakefield.Bys_metadata = metadata0_By

        # extract final fields
        Ez, metadata_Ez = ts.get_field(field='E', coord='z', iteration=max(ts.iterations), plot=False)
        self.final.plasma.wakefield.Ezs = Ez
        self.final.plasma.wakefield.Ezs_metadata = metadata_Ez
        Bz, metadata_Bz = ts.get_field(field='B', coord='z', iteration=max(ts.iterations), plot=False)
        self.final.plasma.wakefield.Bzs = Bz
        self.final.plasma.wakefield.Bzs_metadata = metadata_Bz
        Ex, metadata_Ex = ts.get_field(field='E', coord='x', iteration=max(ts.iterations), plot=False)
        self.final.plasma.wakefield.Exs = Ex
        self.final.plasma.wakefield.Exs_metadata = metadata_Ex
        Bx, metadata_Bx = ts.get_field(field='B', coord='x', iteration=max(ts.iterations), plot=False)
        self.final.plasma.wakefield.Bxs = Bx
        self.final.plasma.wakefield.Bxs_metadata = metadata_Bx
        Ey, metadata_Ey = ts.get_field(field='E', coord='y', iteration=max(ts.iterations), plot=False)
        self.final.plasma.wakefield.Eys = Ey
        self.final.plasma.wakefield.Eys_metadata = metadata_Ey
        By, metadata_By = ts.get_field(field='B', coord='y', iteration=max(ts.iterations), plot=False)
        self.final.plasma.wakefield.Bys = By
        self.final.plasma.wakefield.Bys_metadata = metadata_By
        
        # extract initial plasma density
        rho0_plasma, metadata0_plasma = ts.get_field(field='rho', iteration=min(ts.iterations))
        self.initial.plasma.density.extent = metadata0_plasma.imshow_extent
        self.initial.plasma.density.rho = -(rho0_plasma/SI.e)
        self.initial.plasma.density.metadata = metadata0_plasma

        # extract final plasma density
        rho_plasma, metadata_plasma = ts.get_field(field='rho', iteration=max(ts.iterations))
        self.final.plasma.density.extent = metadata_plasma.imshow_extent
        self.final.plasma.density.rho = -(rho_plasma/SI.e)
        self.final.plasma.density.metadata = metadata_plasma
        
        # extract initial beam density
        data0_beam = ts.get_particle(species='beam', var_list=['x','y','z','w'], iteration=min(ts.iterations))
        data0_driver = ts.get_particle(species='driver', var_list=['x','y','z','w'], iteration=min(ts.iterations))
        extent0 = metadata0_plasma.imshow_extent
        Nbins0 = self.initial.plasma.density.rho.shape
        dr0 = (extent0[3]-extent0[2])/Nbins0[0]
        dz0 = (extent0[1]-extent0[0])/Nbins0[1]
        mask0_beam = np.logical_and(data0_beam[1] < dr0/2, data0_beam[1] > -dr0/2)
        jz0_beam, _, _ = np.histogram2d(data0_beam[0][mask0_beam], data0_beam[2][mask0_beam], weights=data0_beam[3][mask0_beam], bins=Nbins0, range=[extent0[2:4],extent0[0:2]])
        mask0_driver = np.logical_and(data0_driver[1] < dr0/2, data0_driver[1] > -dr0/2)
        jz0_driver, _, _ = np.histogram2d(data0_driver[0][mask0_driver], data0_driver[2][mask0_driver], weights=data0_driver[3][mask0_driver], bins=Nbins0, range=[extent0[2:4],extent0[0:2]])
        self.initial.beam.density.extent = metadata0_plasma.imshow_extent
        self.initial.beam.density.rho = (jz0_beam+jz0_driver)/(dr0*dr0*dz0)

        # extract final beam density
        data_beam = ts.get_particle(species='beam', var_list=['x','y','z','w'], iteration=max(ts.iterations))
        data_driver = ts.get_particle(species='driver', var_list=['x','y','z','w'], iteration=max(ts.iterations))
        extent = metadata_plasma.imshow_extent
        Nbins = self.final.plasma.density.rho.shape
        dr = (extent[3]-extent[2])/Nbins[0]
        dz = (extent[1]-extent[0])/Nbins[1]
        mask_beam = np.logical_and(data_beam[1] < dr/2, data_beam[1] > -dr/2)
        jz_beam, _, _ = np.histogram2d(data_beam[0][mask_beam], data_beam[2][mask_beam], weights=data_beam[3][mask_beam], bins=Nbins, range=[extent[2:4],extent[0:2]])
        mask_driver = np.logical_and(data_driver[1] < dr/2, data_driver[1] > -dr/2)
        jz_driver, _, _ = np.histogram2d(data_driver[0][mask_driver], data_driver[2][mask_driver], weights=data_driver[3][mask_driver], bins=Nbins, range=[extent[2:4],extent[0:2]])
        self.final.beam.density.extent = metadata_plasma.imshow_extent
        self.final.beam.density.rho = (jz_beam+jz_driver)/(dr*dr*dz)
        

    # ==================================================
    def energy_usage(self):
        return None # TODO


    # ==================================================
    def matched_beta_function(self, energy):
        return beta_matched(self.plasma_density, energy) * self.ramp_beta_mag

    
    # ==================================================
    # Apply waterfall function to all beam dump files
    def __waterfall_fcn(self, fcns, edges, data_dir, species='beam', clean=False, remove_halo_nsigma=20, args=None):
        """
        Applies waterfall function to all Wake-T HDF5 output files in ``data_dir``.

         Parameters
        ----------
        fcns: A list of Beam class methods
            Beam class profile methods such as Beam.current_profile, Beam.rel_energy_spectrum, Beam.transverse_profile_x, Beam.transverse_profile_y.

        edges: float list
            Specifies the bins to be used to create the histogram(s) in the waterfall plot(s).

        data_dir: string
            Path to the directory containing all Wake-T HDF5 output files.

        species: string, optional
            Specifies the name of the beam to be extracted.

        clean: bool, optional
            Determines whether the extracted beams from the Wake-T HDF5 output files should be cleaned before further processing.

        remove_halo_nsigma: float, optional
            Defines a threshold for identifying and removing "halo" particles based on their deviation from the core of the particle beam.

        args: float list, optional
            Allows passing additional arguments to the functions in ``fcns``.
            
            
        Returns
        ----------
        waterfalls: list of 2D float NumPy arrays
            Each element in ``waterfalls`` corresponds to the output of one function in ``fcns`` applied across all files (i.e., simulation outputs). The dimension of element i is determined by the length of ``edges`` and the number of simulation outputs.
        
        locations: [m] 1D float NumPy array
            Stores the location for each slice of ``waterfalls``.
        
        bins: list of 1D float NumPy arrays
            Each element contains the bins used for the slices/histograms in ``waterfalls``.
        """

        from abel.apis.wake_t.wake_t_api import wake_t_hdf5_load
        
        # find number of beam outputs to plot
        files = sorted(os.listdir(data_dir))
        num_outputs = len(files)
        
        # prepare to read simulation data
        file_path = data_dir + files[0]
        
        # declare data structure
        bins = [None] * len(fcns)
        waterfalls = [None] * len(fcns)
        for j in range(len(fcns)):
            waterfalls[j] = np.empty((len(edges[j])-1, num_outputs))
        
        locations = np.empty(num_outputs)
        
        # go through files
        for index in range(num_outputs):
            # load phase space
            file_path = data_dir + files[index]
            beam = wake_t_hdf5_load(file_path=file_path, species=species)

            if clean:
                beam.remove_halo_particles(nsigma=remove_halo_nsigma)
            
            # find beam location
            locations[index] = beam.location
            
            # get all waterfalls (apply argument if it exists)
            for j in range(len(fcns)):
                if args[j] is None:
                    waterfalls[j][:,index], bins[j] = fcns[j](beam, bins=edges[j])
                else:
                    waterfalls[j][:,index], bins[j] = fcns[j](beam, args[j][index], bins=edges[j])
                
        return waterfalls, locations, bins


    # ==================================================
    def extract_waterfalls(self, data_dir, species='beam', clean=False, remove_halo_nsigma=20, args=None):
        '''
        Extracts data for waterfall plots for current profile, relative energy spectrum, horizontal transverse profile and vertical transverse profile.

        Parameters
        ----------
        data_dir: string
            Path to the directory containing all Wake-T HDF5 output files.

        species: string, optional
            Specifies the name of the beam to be extracted.

        clean: bool, optional
            Determines whether the extracted beams from the Wake-T HDF5 output files should be cleaned before further processing.

        remove_halo_nsigma: float, optional
            Defines a threshold for identifying and removing "halo" particles based on their deviation from the core of the particle beam.

        args: float list, optional
            Allows passing additional arguments to the functions in fcns.

            
        Returns
        ----------
        waterfalls: list of 2D float NumPy arrays
            Each element in ``waterfalls`` corresponds to the output of one function in fcns applied across all files (i.e., simulation outputs). The dimension of element i is determined by the length of edges and the number of simulation outputs.
        
        locations: [m] 1D float NumPy array
            Stores the location for each slice of the ``waterfalls``.
        
        bins: list of 1D float NumPy arrays
            Each element contains the bins used for the slices/histograms in ``waterfalls``.
        '''

        from abel.apis.wake_t.wake_t_api import wake_t_hdf5_load
        
        files = sorted(os.listdir(data_dir))
        file_path = data_dir + files[0]
        beam0 = wake_t_hdf5_load(file_path=file_path, species=species)
        num_bins = int(np.sqrt(len(beam0)*2))
        nsig = 5
        
        if species == 'driver':
            deltaedges = np.linspace(-0.5, 0.5, num_bins)
        else:
            deltaedges = np.linspace(-0.05, 0.05, num_bins)
        tedges = (beam0.z_offset(clean=True) + nsig*beam0.bunch_length(clean=True)*np.linspace(-1, 1, num_bins)) / SI.c
        xedges = (nsig*beam0.beam_size_x() + abs(beam0.x_offset()))*np.linspace(-1, 1, num_bins)
        yedges = (nsig*beam0.beam_size_y() + abs(beam0.y_offset()))*np.linspace(-1, 1, num_bins)
        
        waterfalls, locations, bins = self.__waterfall_fcn([Beam.current_profile, Beam.rel_energy_spectrum, Beam.transverse_profile_x, Beam.transverse_profile_y], [tedges, deltaedges, xedges, yedges], data_dir, species=species, clean=clean, remove_halo_nsigma=remove_halo_nsigma, args=[None, None, None, None])

        return waterfalls, locations, bins


    # ==================================================
    def plot_waterfalls(self, waterfalls, locations, bins, save_fig=False):
        '''
        Makes waterfall plots for current profile, relative energy spectrum, horizontal transverse profile and vertical transverse profile.

        Parameters
        ----------
        waterfalls: list of 2D float NumPy arrays
            Each element in ``waterfalls`` corresponds to the output of one function in fcns applied across all files (i.e., simulation outputs). The dimension of element i is determined by the length of edges and the number of simulation outputs.
        
        locations: [m] 1D float NumPy array
            Stores the location for each slice of the ``waterfalls``.
        
        bins: list of 1D float NumPy arrays
            Each element contains the bins used for the slices/histograms in ``waterfalls``.

        save_fig: bool, optional
            Flag for saving the output figure.
        '''

        # prepare figure
        fig, axs = plt.subplots(4,1)
        fig.set_figwidth(8)
        fig.set_figheight(2.8*4)
        
        # current profile
        Is = waterfalls[0]
        ts = bins[0]
        c0 = axs[0].pcolor(locations, ts*SI.c*1e6, -Is/1e3, cmap=CONFIG.default_cmap, shading='auto')
        cbar0 = fig.colorbar(c0, ax=axs[0])
        axs[0].set_ylabel(r'Longitudinal position [$\mathrm{\mu}$m]')
        cbar0.ax.set_ylabel('Beam current [kA]')
        #axs[0].set_title('Shot ' + str(shot+1))
        
        # energy profile
        dQddeltas = waterfalls[1]
        deltas = bins[1]
        c1 = axs[1].pcolor(locations, deltas*1e2, -dQddeltas*1e7, cmap=CONFIG.default_cmap, shading='auto')
        cbar1 = fig.colorbar(c1, ax=axs[1])
        axs[1].set_ylabel('Relative energy spread [%]')
        cbar1.ax.set_ylabel('Spectral density [nC/%]')
        
        densityX = waterfalls[2]
        xs = bins[2]
        c2 = axs[2].pcolor(locations, xs*1e6, -densityX*1e3, cmap=CONFIG.default_cmap, shading='auto')
        cbar2 = fig.colorbar(c2, ax=axs[2])
        axs[2].set_ylabel(r'Horizontal position [$\mathrm{\mu}$m]')
        cbar2.ax.set_ylabel(r'Charge density [nC/$\mathrm{\mu}$m]')
        
        densityY = waterfalls[3]
        ys = bins[3]
        c3 = axs[3].pcolor(locations, ys*1e6, -densityY*1e3, cmap=CONFIG.default_cmap, shading='auto')
        cbar3 = fig.colorbar(c3, ax=axs[3])
        axs[3].set_ylabel(r'Vertical position [$\mathrm{\mu}$m]')
        cbar3.ax.set_ylabel(r'Charge density [nC/$\mathrm{\mu}$m]')
        axs[3].set_xlabel('Location along the stage [m]')
        
        plt.show()
        if save_fig:
            plot_path = self.run_path + 'plots' + os.sep
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            filename = plot_path + 'waterfalls' + '.png'
            fig.savefig(filename, format='png', dpi=600, bbox_inches='tight', transparent=False)

            