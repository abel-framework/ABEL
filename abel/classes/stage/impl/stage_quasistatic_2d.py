from abel.CONFIG import CONFIG
from abel.classes.stage.stage import Stage
import numpy as np
import scipy.constants as SI
from abel.utilities.plasma_physics import *
import os, shutil, uuid, copy, sys
from abel.physics_models.particles_transverse_wake_instability import transverse_wake_instability_particles

class StageQuasistatic2d(Stage):
    
    def __init__(self, nom_accel_gradient=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=None, enable_transverse_instability=False, enable_radiation_reaction=False, calculate_evolution=False):
        
        super().__init__(nom_accel_gradient, nom_energy_gain, plasma_density, driver_source, ramp_beta_mag)
        
        # physics flags
        self.enable_transverse_instability = enable_transverse_instability
        self.enable_radiation_reaction = enable_radiation_reaction

        # simulation flags
        self.calculate_evolution = calculate_evolution
    
        
    # track the particles through
    def track(self, beam_incoming, savedepth=0, runnable=None, verbose=False):

        import wake_t
        from abel.apis.wake_t.wake_t_api import beam2wake_t_bunch, wake_t_bunch2beam
        from openpmd_viewer import OpenPMDTimeSeries
        import warnings
        
        # suppress numba warnings from Ocelot
        warnings.simplefilter('ignore', category=RuntimeWarning)
        
        # make driver (and convert to WakeT bunch)
        driver_incoming = self.driver_source.track()

        # Set ramp lengths, nominal energies, nominal energy gains
        # and flattop nominal energy if not already done
        self._prepare_ramps()

        # plasma-density ramps (de-magnify beta function)
        if self.upramp is not None:
            self.upramp.calculate_evolution = self.calculate_evolution
            beam0, driver0 = self.track_upramp(beam_incoming, driver_incoming)
        else:
            beam0 = copy.deepcopy(beam_incoming)
            driver0 = copy.deepcopy(driver_incoming)
            if self.ramp_beta_mag is not None:
                beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver_incoming)
                driver0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver_incoming)

        # make copy of the beam to update later
        beam = copy.deepcopy(beam0)
        
        # convert beams to WakeT bunches
        driver0_wake_t = beam2wake_t_bunch(driver0, name='driver')
        beam0_wake_t = beam2wake_t_bunch(beam0, name='beam')
        
        # create plasma stage
        box_min_z = beam0.z_offset(clean=True) - 5 * beam0.bunch_length(clean=True) - 0.25/k_p(self.plasma_density)
        box_max_z = driver0.z_offset(clean=True) + 4 * driver0.bunch_length(clean=True)
        box_size_r = 3 * blowout_radius(self.plasma_density, driver0.peak_current())
        k_beta_driver = k_p(self.plasma_density)/np.sqrt(2*driver0.gamma())
        k_beta_beam = k_p(self.plasma_density)/np.sqrt(2*beam0.gamma())
        lambda_betatron_min = 2*np.pi/max(k_beta_beam, k_beta_driver)
        lambda_betatron_max = 2*np.pi/min(k_beta_beam, k_beta_driver)
        dz = lambda_betatron_min/10
        
        # need to make sufficiently many steps
        n_out = max(1, round(lambda_betatron_max/lambda_betatron_min/2))
        plasma = wake_t.PlasmaStage(length=dz, density=self.plasma_density, wakefield_model='quasistatic_2d',
                                    r_max=box_size_r, r_max_plasma=box_size_r, xi_min=box_min_z, xi_max=box_max_z, 
                                    n_out=n_out, n_r=512, n_xi=512, dz_fields=dz, ppc=4)
        
        # make temp folder
        if not os.path.exists(CONFIG.temp_path):
            os.mkdir(CONFIG.temp_path)
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
        if not os.path.exists(tmpfolder):
            os.mkdir(tmpfolder)

        # function to quiet the tracking output
        from contextlib import contextmanager
        @contextmanager
        def suppress_stdout():
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:  
                    yield
                finally:
                    sys.stdout = old_stdout
                    
        # perform tracking
        with suppress_stdout():
            bunches = plasma.track([driver0_wake_t, beam0_wake_t], opmd_diag=True, diag_dir=tmpfolder)
        
        # convert back to ABEL beams
        beam_waket = wake_t_bunch2beam(bunches[1][-1])
        driver_waket = wake_t_bunch2beam(bunches[0][-1])
        
        # save evolution of the beam and driver
        self._driver_evolution = wake_t.diagnostics.analyze_bunch_list(bunches[0])
        self._beam_evolution = wake_t.diagnostics.analyze_bunch_list(bunches[1])

        # extract wakefield info
        ts = OpenPMDTimeSeries(tmpfolder+'hdf5/')
        Ez, metadata = ts.get_field(field='E', coord='z', iteration=min(ts.iterations))
        self.initial.plasma.wakefield.onaxis.zs = metadata.z
        self.initial.plasma.wakefield.onaxis.Ezs = Ez[round(len(metadata.r)/2),:].flatten()
        
        # extract initial plasma density
        rho0_plasma, metadata0_plasma = ts.get_field(field='rho', iteration=min(ts.iterations))
        self.initial.plasma.density.extent = metadata0_plasma.imshow_extent
        self.initial.plasma.density.rho = -(rho0_plasma/SI.e)
        
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

        # remove temporary directory
        shutil.rmtree(tmpfolder)
        
        if self.enable_transverse_instability:

            # TODO: make sure driver offset is correctly handled
            
            # find maximum blowout radius
            rs = np.linspace(self.initial.plasma.density.extent[2], self.initial.plasma.density.extent[3], self.initial.plasma.density.rho.shape[0])
            rbs = np.zeros(self.initial.plasma.density.rho.shape[1])
            threshold = 0.9*self.plasma_density
            for i in range(len(rbs)):
                rbs_upper = rs[np.logical_and(rs > 0, self.initial.plasma.density.rho[:,i] > threshold)].min()
                rbs_lower = rs[np.logical_and(rs < 0, self.initial.plasma.density.rho[:,i] > threshold)].max()
                rbs[i] = (rbs_upper-rbs_lower)/2

            # interpolate for each particle
            import scipy.interpolate.interp1d as interp1d
            rbs_interp = interp1d(self.initial.plasma.wakefield.onaxis.zs, rbs)
            Ezs_interp = interp1d(self.initial.plasma.wakefield.onaxis.zs, self.initial.plasma.wakefield.onaxis.Ezs)
            
            # perform tracking  # TODO: transverse_wake_instability_particles() signa has changed.
            beam = transverse_wake_instability_particles(beam, self.plasma_density, Ezs_interp, rbs_interp, self.length_flattop, show_prog_bar=True)
            
        else:
            
            # calculate energy gain
            delta_Es = self.length_flattop*(beam_waket.Es() - beam.Es())/dz

            # find driver offset (to shift the beam relative) and apply betatron motion
            output = beam.apply_betatron_motion(self.length_flattop, self.plasma_density, delta_Es, x0_driver=driver0.x_offset(), y0_driver=driver0.y_offset(), radiation_reaction=self.enable_radiation_reaction, calc_evolution=self.calculate_evolution)
            if self.calculate_evolution:
                Es_final, self.evolution.beam = output
            else:
                Es_final = output
            
            # accelerate beam (and remove nans)
            beam.set_Es(Es_final)
            
        # decelerate driver (and remove nans)
        delta_Es_driver = self.length_flattop*(driver0.Es()-driver_waket.Es())/dz
        driver = copy.deepcopy(driver0)
        driver.apply_betatron_damping(delta_Es_driver)
        driver.set_Es(driver0.Es() + delta_Es_driver)
        
        # apply plasma-density down ramp (magnify beta function)
        if self.downramp is not None:
            self.downramp.calculate_evolution = self.calculate_evolution
            beam_outgoing, driver_outgoing = self.track_downramp(beam, driver)
        else:
            beam_outgoing = copy.deepcopy(beam)
            driver_outgoing = copy.deepcopy(driver)
            if self.ramp_beta_mag is not None:
                beam_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)
                driver_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)
        
        # copy meta data from input beam (will be iterated by super)
        beam_outgoing.trackable_number = beam_incoming.trackable_number
        beam_outgoing.stage_number = beam_incoming.stage_number
        beam_outgoing.location = beam_incoming.location
        
        # clean nan particles and extreme outliers
        beam.remove_nans()
        beam.remove_halo_particles()
        
        # calculate efficiency
        self.calculate_efficiency(beam_incoming, driver_incoming, beam_outgoing, driver_outgoing)
        
        # save current profile
        self.calculate_beam_current(beam_incoming, driver_incoming, beam_outgoing, driver_outgoing)
        
        # return the beam (and optionally the driver)
        if self._return_tracked_driver:
            return super().track(beam_outgoing, savedepth, runnable, verbose), driver_outgoing
        else:
            return super().track(beam_outgoing, savedepth, runnable, verbose)
        