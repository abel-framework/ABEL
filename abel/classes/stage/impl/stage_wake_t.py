from abel.classes.stage.stage import Stage
from abel.classes.beam import Beam
from abel.CONFIG import CONFIG
import scipy.constants as SI
import os, shutil, uuid
import numpy as np
from types import SimpleNamespace

class StageWakeT(Stage):
    
    def __init__(self, nom_accel_gradient=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=None, num_cell_xy=256, keep_data=False, ion_motion=False):
        
        super().__init__(nom_accel_gradient=nom_accel_gradient, nom_energy_gain=nom_energy_gain, plasma_density=plasma_density, driver_source=driver_source, ramp_beta_mag=ramp_beta_mag)
        
        self.num_cell_xy = num_cell_xy
        self.keep_data = keep_data
        
        # physics flags
        self.ion_motion = ion_motion

        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        from abel.utilities.plasma_physics import blowout_radius, k_p, beta_matched
        
        # make temp folder
        if not os.path.exists(CONFIG.temp_path):
            os.mkdir(CONFIG.temp_path)
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
        if not os.path.exists(tmpfolder):
            os.mkdir(tmpfolder)

        # Set ramp lengths, nominal energies, nominal energy gains
        # and flattop nominal energy if not already done
        self._prepare_ramps()
        plasma_profile = self.get_plasma_profile()
        
        # make driver (and convert to WakeT bunch)
        driver0 = self.driver_source.track()
        
        # convert beams to WakeT bunches
        from abel.apis.wake_t.wake_t_api import beam2wake_t_bunch
        driver0_wake_t = beam2wake_t_bunch(driver0, name='driver')
        beam0_wake_t = beam2wake_t_bunch(beam0, name='beam')
        
        # make longitudinal box range
        num_sigmas = 6
        #box_min_z = beam0.z_offset() - num_sigmas * beam0.bunch_length()
        R_blowout = blowout_radius(self.plasma_density, driver0.peak_current())
        box_min_z = driver0.z_offset() - 3.3 * R_blowout
        #box_max_z = min(driver0.z_offset() + num_sigmas * driver0.bunch_length(), np.max(driver0.zs())+0.25/k_p(self.plasma_density))
        box_max_z = min(driver0.z_offset() + num_sigmas * driver0.bunch_length(), np.max(driver0.zs()) + 0.5*R_blowout)
        box_range_z = [box_min_z, box_max_z]
        
        # making transverse box size
        box_size_r = np.max([4/k_p(self.plasma_density), 3*blowout_radius(self.plasma_density, driver0.peak_current())])
        
        # calculate number of cells in x to get similar resolution
        dr = box_size_r/self.num_cell_xy
        num_cell_z = round((box_max_z-box_min_z)/dr)
        
        # find stepsize
        beta_matched = np.sqrt(2*min(beam0.gamma(),driver0.gamma()/2))/k_p(self.plasma_density)
        dz = beta_matched/10

        # select the wakefield model (ion motion or not)
        if self.ion_motion:
            wakefield_model='quasistatic_2d_ion'
        else:
            wakefield_model='quasistatic_2d'
            
        n_out = round(self.length/dz/8)
        import wake_t
        plasma = wake_t.PlasmaStage(length=self.length, density=plasma_profile, wakefield_model=wakefield_model,
                                    r_max=box_size_r, r_max_plasma=box_size_r, xi_min=box_min_z, xi_max=box_max_z, 
                                    n_out=n_out, n_r=int(self.num_cell_xy), n_xi=int(num_cell_z), dz_fields=dz, ppc=1)
        
        # do tracking
        bunches = plasma.track([driver0_wake_t, beam0_wake_t], opmd_diag=True, diag_dir=tmpfolder)
        
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
        from abel.apis.wake_t.wake_t_api import wake_t_bunch2beam
        beam = wake_t_bunch2beam(bunches[1][-1])
        driver = wake_t_bunch2beam(bunches[0][-1])
        
        # copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        # calculate efficiency
        self.calculate_efficiency(beam0, driver0, beam, driver)
        
        # save current profile
        self.calculate_beam_current(beam0, driver0, beam, driver)
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def __extract_evolution(self, bunches):

        # get beam
        from wake_t.diagnostics import analyze_bunch_list

        # cycle through bunches (driver and trailing bunch)
        for i, bunch in enumerate(bunches):

            # extract data
            import warnings
            warnings.filterwarnings("ignore")
            beam_evol = analyze_bunch_list(bunch)
            
            # make container
            evol = SimpleNamespace()
            
            # store variables
            evol.location = beam_evol['prop_dist']
            evol.charge = beam_evol['q_tot']
            evol.energy = beam_evol['avg_ene']*SI.m_e*SI.c**2/SI.e
            evol.z = np.empty_like(evol.location)*np.nan
            evol.x = beam_evol['x_avg']
            evol.y = beam_evol['y_avg']
            evol.xp = beam_evol['theta_x']
            evol.yp = beam_evol['theta_y']
            evol.rel_energy_spread = beam_evol['rel_ene_spread']
            evol.energy_spread = evol.rel_energy_spread*evol.energy
            evol.beam_size_x = beam_evol['sigma_x']
            evol.beam_size_y = beam_evol['sigma_y']
            evol.bunch_length = beam_evol['sigma_z']
            evol.emit_nx = beam_evol['emitt_x']
            evol.emit_ny = beam_evol['emitt_y']
            evol.beta_x = beam_evol['beta_x']
            evol.beta_y = beam_evol['beta_y']
            evol.plasma_density = np.ones_like(evol.location)*self.plasma_density
            
            # assign it to the right beam
            if i == 0:
                self.evolution.driver = evol
            elif i == 1:
                self.evolution.beam = evol
        

    
    def __extract_initial_and_final_step(self, tmpfolder):

        from openpmd_viewer import OpenPMDTimeSeries
        
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

    
    def get_plasma_profile(self):
        """Prepare the ramps (local to WakeT)."""
        
        # make the plasma ramp profile
        if self.has_ramp():

            # assert uniform ramps (for now)
            assert(self.upramp.ramp_shape == 'uniform')
            assert(self.downramp.ramp_shape == 'uniform')

            # define the density levels
            n_upramp = self.upramp.plasma_density
            n_flattop = self.plasma_density
            n_downramp = self.downramp.plasma_density

            # define the ramp transition locations
            z_up = self.upramp.length
            z_down = self.upramp.length + self.length_flattop
            
            # define the (uniform) profile function            
            profile_fcn = lambda z: np.piecewise(z, [z<z_up,np.logical_and(z>=z_up,z<=z_down),z>z_down], [n_upramp, n_flattop, n_downramp])

            return profile_fcn
            
        else:

            # otherwise return a constant density
            return self.plasma_density
    