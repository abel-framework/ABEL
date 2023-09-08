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
        
        self.driver_to_wake_efficiency = None
        self.wake_to_beam_efficiency = None
        self.driver_to_beam_efficiency = None
        
        self.ramp_beta_mag = 1
        
        self._driver_initial = None
        self._driver_final = None
        self._initial_wakefield = None
        self._current_profile = None
        
    # track the particles through
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        # make driver (and convert to WakeT bunch)
        driver0 = self.driver_source.track()
        
        # apply plasma-density down ramp (demagnify beta function)
        driver0.magnify_beta_function(1/self.ramp_beta_mag)
        beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver0)
        
        # convert beams to WakeT bunches
        driver0_wake_t = beam2wake_t_bunch(driver0, name='driver')
        beam0_wake_t = beam2wake_t_bunch(beam0, name='beam')

        # create plasma stage
        box_min_z = beam0.z_offset() - 5 * beam0.bunch_length() - 0.25/k_p(self.plasma_density)
        box_max_z = driver0.z_offset() + 4 * driver0.bunch_length()
        box_size_r = 3 * blowout_radius(self.plasma_density, driver0.peak_current())
        k_beta = k_p(self.plasma_density)/np.sqrt(2*min(beam0.gamma(),driver0.gamma()/2))
        lambda_betatron = (2*np.pi/k_beta)
        dz = lambda_betatron/10
        plasma = wake_t.PlasmaStage(length=dz, density=self.plasma_density, wakefield_model='quasistatic_2d',
                                    r_max=box_size_r, r_max_plasma=box_size_r, xi_min=box_min_z, xi_max=box_max_z, 
                                    n_out=1, n_r=512, n_xi=512, dz_fields=dz, ppc=4)
        
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
        self._initial_wakefield = (metadata.z, Ez[round(len(metadata.r)/2),:].flatten())

        # remove temporary directory
        if os.path.exists(tmpfolder):
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
        
        # apply plasma-density up ramp (magnify beta function)
        beam.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver0)
        driver.magnify_beta_function(self.ramp_beta_mag)
        
        # calculate efficiency
        Etot0_beam = beam0.total_energy()
        Etot_beam = beam.total_energy()
        Etot0_driver = driver0.total_energy()
        Etot_driver = driver.total_energy()
        self.driver_to_wake_efficiency = (Etot0_driver-Etot_driver)/Etot0_driver
        self.wake_to_beam_efficiency = (Etot_beam-Etot0_beam)/(Etot0_driver-Etot_driver)
        self.driver_to_beam_efficiency = self.driver_to_wake_efficiency*self.wake_to_beam_efficiency

        # save drivers
        self._driver_initial = driver0
        self._driver_final = driver
        
        # save current profile
        dt = 40*np.mean([driver0.bunch_length(clean=True)/np.sqrt(len(driver0)), beam0.bunch_length(clean=True)/np.sqrt(len(beam0))])
        tbins = np.arange(metadata.z.min(), metadata.z.max(), dt)/SI.c
        Is, ts = (driver0 + beam0).current_profile(bins=tbins)
        self._current_profile = (ts*SI.c, Is)
        
        # copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        return super().track(beam, savedepth, runnable, verbose)
        
    
       
    def plot_wakefield(self, beam=None, save_to_file=None):
        
        # check if wakefield exists
        assert self._initial_wakefield is not None, 'Wakefield not calculated yet'
        
        # extract wakefield and current profileif not already existing
        zs, Ezs = self._initial_wakefield
        zs_I, Is = self._current_profile
        
        # plot it
        fig, axs = plt.subplots(2, 1)
        fig.set_figwidth(CONFIG.plot_width_default*0.7)
        fig.set_figheight(8)
        col0 = "xkcd:light gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        af = 0.1
        zlims = [min(zs)*1e6, max(zs)*1e6]
        
        axs[0].plot(zs*1e6, np.zeros(zs.shape), '-', color=col0)
        if self.nom_energy_gain is not None:
            axs[0].plot(zs*1e6, -self.nom_energy_gain/self.get_length()*np.ones(zs.shape)/1e9, ':', color=col2)
        axs[0].plot(zs*1e6, Ezs/1e9, '-', color=col1)
        axs[0].set_xlabel('z (um)')
        axs[0].set_ylabel('Longitudinal electric field (GV/m)')
        axs[0].set_xlim(zlims)
        axs[0].set_ylim(bottom=-max(1.1*min(Ezs), wave_breaking_field(self.plasma_density))/1e9, top=1.1*max(Ezs)/1e9)
        
        axs[1].fill(np.concatenate((zs_I, np.flip(zs_I)))*1e6, np.concatenate((-Is, np.zeros(Is.shape)))/1e3, color=col1, alpha=af)
        axs[1].plot(zs_I*1e6, -Is/1e3, '-', color=col1)
        axs[1].set_xlabel('z (um)')
        axs[1].set_ylabel('Beam current (kA)')
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=0, top=1.2*max(-Is)/1e3)
        
        # save to file
        if save_to_file is not None:
            plt.savefig(save_to_file, format="pdf", bbox_inches="tight")
        
       
    
    # matched beta function of the stage (for a given energy)
    def matched_beta_function(self, energy):
        return beta_matched(self.plasma_density, energy) * self.ramp_beta_mag
        
    def get_length(self):
        return self.length
    
    def get_nom_energy_gain(self):
        return self.nom_energy_gain
    
    def energy_efficiency(self):
        return self.driver_to_beam_efficiency, self.driver_to_wake_efficiency, self.wake_to_beam_efficiency
    
    def energy_usage(self):
        return self.driver_source.energy_usage()
    