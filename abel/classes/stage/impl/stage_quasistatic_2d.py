from abel import Stage, CONFIG
from matplotlib import pyplot as plt
import numpy as np
import scipy.constants as SI
from abel.utilities.plasma_physics import *
import wake_t
import os, shutil, uuid, copy, sys
from openpmd_viewer import OpenPMDTimeSeries
from abel.apis.wake_t.wake_t_api import beam2wake_t_bunch, wake_t_bunch2beam
from contextlib import contextmanager

class StageQuasistatic2d(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None, add_driver_to_beam=False):
        
        super().__init__(length, nom_energy_gain, plasma_density)
        
        self.add_driver_to_beam = add_driver_to_beam
        self.driver_source = driver_source
        
        self.ramp_beta_mag = 1
        
        self._driver_initial = None
        self._driver_final = None
        self._initial_wakefield = None
        self._current_profile = None
        
    # track the particles through
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        # suppress numba warnings from Ocelot
        import warnings
        warnings.simplefilter('ignore', category=RuntimeWarning)
        
        # make driver (and convert to WakeT bunch)
        driver0 = self.driver_source.track()
        
        # apply plasma-density up ramp (demagnify beta function)
        driver0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver0)
        beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver0)
        
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
        beam = wake_t_bunch2beam(bunches[1][-1])
        driver = wake_t_bunch2beam(bunches[0][-1])
        
        # save evolution of the beam and driver
        self._driver_evolution = wake_t.diagnostics.analyze_bunch_list(bunches[0])
        self._beam_evolution = wake_t.diagnostics.analyze_bunch_list(bunches[1])

        # extract wakefield info
        tseries = OpenPMDTimeSeries(tmpfolder+'hdf5/')
        Ez, metadata = tseries.get_field(field='E', coord='z', iteration=0)
        self.initial.plasma.wakefield.onaxis.zs = metadata.z
        self.initial.plasma.wakefield.onaxis.Ezs = Ez[round(len(metadata.r)/2),:].flatten()
        
        # remove temporary directory
        shutil.rmtree(tmpfolder)
        
        # calculate energy gain
        delta_Es = self.length*(beam.Es() - beam0.Es())/dz
        
        # find driver offset (to shift the beam relative) and apply betatron motion
        beam.apply_betatron_motion(self.length, self.plasma_density, delta_Es, x0_driver=driver0.x_offset(), y0_driver=driver0.y_offset())
        
        # accelerate beam (and remove nans)
        beam.set_Es(beam0.Es() + delta_Es)
        
        # decelerate driver (and remove nans)
        delta_Es_driver = self.length*(driver0.Es()-driver.Es())/dz
        driver.apply_betatron_damping(delta_Es_driver)
        driver.flip_transverse_phase_spaces()
        driver.set_Es(driver0.Es() + delta_Es_driver)
        
        # apply plasma-density down ramp (magnify beta function)
        beam.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver0)
        driver.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver0)
        
        # copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        # clean nan particles and extreme outliers
        beam.remove_nans()
        beam.remove_halo_particles()
        
        # calculate efficiency
        self.calculate_efficiency(beam0, driver0, beam, driver)
        
        # save current profile
        self.calculate_beam_current(beam0, driver0, beam, driver)
        
        return super().track(beam, savedepth, runnable, verbose)
    
    
    # matched beta function of the stage (for a given energy)
    def matched_beta_function(self, energy):
        return beta_matched(self.plasma_density, energy) * self.ramp_beta_mag
        
    def energy_usage(self):
        return self.driver_source.energy_usage()
    