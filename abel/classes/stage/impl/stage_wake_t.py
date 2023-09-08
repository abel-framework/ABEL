from abel import Stage, Beam, CONFIG
import scipy.constants as SI
from matplotlib import pyplot as plt
from abel.utilities.plasma_physics import blowout_radius, k_p, beta_matched
from abel.apis.wake_t.wake_t_api import beam2wake_t_bunch, wake_t_bunch2beam
import numpy as np
import wake_t
from openpmd_viewer import OpenPMDTimeSeries


class StageWakeT(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=1):
        
        super().__init__(length, nom_energy_gain, plasma_density)
        
        self.driver_source = driver_source
        self.ramp_beta_mag = ramp_beta_mag

        self._beam_evolution = None
        self._driver_evolution = None
        
    
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # make driver (and convert to WakeT bunch)
        driver0 = self.driver_source.track()
        
        # apply plasma-density down ramp (demagnify beta function)
        if self.ramp_beta_mag is not None:
            beam0.magnify_beta_function(1/self.ramp_beta_mag)
            driver0.magnify_beta_function(1/self.ramp_beta_mag)
        
        # convert beams to WakeT bunches
        driver0_wake_t = beam2wake_t_bunch(driver0, name='driver')
        beam0_wake_t = beam2wake_t_bunch(beam0, name='beam')

        # create plasma stage
        box_min_z = beam0.z_offset() - 5 * beam0.bunch_length()
        box_max_z = driver0.z_offset() + 5 * driver0.bunch_length()
        box_size_r = 3 * blowout_radius(self.plasma_density, driver0.peak_current())
        
        k_beta = k_p(self.plasma_density)/np.sqrt(2*min(beam0.gamma(),driver0.gamma()/2))
        lambda_betatron = (2*np.pi/k_beta)
        dz = lambda_betatron/20
        n_out = round(self.length/dz/2)
        plasma = wake_t.PlasmaStage(length=self.length, density=self.plasma_density, wakefield_model='quasistatic_2d',
                                    r_max=box_size_r, r_max_plasma=box_size_r, xi_min=box_min_z, xi_max=box_max_z, 
                                    n_out=n_out, n_r=256, n_xi=256, dz_fields=dz, ppc=4)
        
        # do tracking
        bunches = plasma.track([driver0_wake_t, beam0_wake_t], opmd_diag=False)
        
        # save evolution of the beam and driver
        self._driver_evolution = wake_t.diagnostics.analyze_bunch_list(bunches[0])
        self._beam_evolution = wake_t.diagnostics.analyze_bunch_list(bunches[1])
        
        # convert back to ABEL beam
        beam = wake_t_bunch2beam(bunches[1][-1])
        
        # copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        # apply plasma-density up ramp (magnify beta function)
        if self.ramp_beta_mag is not None:
            beam.magnify_beta_function(self.ramp_beta_mag)
            driver.magnify_beta_function(self.ramp_beta_mag)
            
        return super().track(beam, savedepth, runnable, verbose)
    
    
    def plot_evolution(self):
        
        if self._beam_evolution is not None and self._driver_evolution is not None:
            
            # prepare the figure
            fig, axs = plt.subplots(5, 2)
            fig.set_figwidth(CONFIG.plot_fullwidth_default)
            fig.set_figheight(CONFIG.plot_fullwidth_default*0.8)

            # beam plots
            axs[0,0].plot(self._beam_evolution['prop_dist'], self._beam_evolution['avg_ene']*SI.m_e*SI.c**2/SI.e/1e9)
            axs[0,0].set_xlabel('s (m)')
            axs[0,0].set_ylabel('Energy, mean (GeV)')
            axs[0,0].set_title('Beam')
            
            axs[1,0].plot(self._beam_evolution['prop_dist'], self._beam_evolution['rel_ene_spread']*1e2)
            axs[1,0].set_xlabel('s (m)')
            axs[1,0].set_ylabel('Rel. energy spread, rms (%)')
            
            axs[2,0].plot(self._beam_evolution['prop_dist'], self._beam_evolution['beta_x']*1e3)
            axs[2,0].plot(self._beam_evolution['prop_dist'], self._beam_evolution['beta_y']*1e3)
            axs[2,0].set_xlabel('s (m)')
            axs[2,0].set_ylabel('Beta function (mm)')
            axs[2,0].set_yscale('log')
            
            axs[3,0].plot(self._beam_evolution['prop_dist'], self._beam_evolution['emitt_x']*1e6)
            axs[3,0].plot(self._beam_evolution['prop_dist'], self._beam_evolution['emitt_y']*1e6)
            axs[3,0].set_xlabel('s (m)')
            axs[3,0].set_ylabel('Norm. emittance (mm mrad)')
            axs[3,0].set_yscale('log')

            axs[4,0].plot(self._beam_evolution['prop_dist'], self._beam_evolution['x_avg']*1e6)
            axs[4,0].plot(self._beam_evolution['prop_dist'], self._beam_evolution['y_avg']*1e6)
            axs[4,0].set_xlabel('s (m)')
            axs[4,0].set_ylabel('Offset, mean (um)')
            
            # driver plots
            axs[0,1].plot(self._driver_evolution['prop_dist'], self._driver_evolution['avg_ene']*SI.m_e*SI.c**2/SI.e/1e9)
            axs[0,1].set_xlabel('s (m)')
            axs[0,1].set_ylabel('Energy, mean (GeV)')
            axs[0,1].set_title('Driver')
            
            axs[1,1].plot(self._beam_evolution['prop_dist'], self._beam_evolution['rel_ene_spread']*1e2)
            axs[1,1].set_xlabel('s (m)')
            axs[1,1].set_ylabel('Rel. energy spread, rms (%)')
            
            axs[2,1].plot(self._driver_evolution['prop_dist'], self._driver_evolution['beta_x'])
            axs[2,1].plot(self._driver_evolution['prop_dist'], self._driver_evolution['beta_y'])
            axs[2,1].set_xlabel('s (m)')
            axs[2,1].set_ylabel('Beta function (m)')
            axs[2,1].set_yscale('log')
            
            axs[3,1].plot(self._driver_evolution['prop_dist'], self._driver_evolution['emitt_x']*1e6)
            axs[3,1].plot(self._driver_evolution['prop_dist'], self._driver_evolution['emitt_y']*1e6)
            axs[3,1].set_xlabel('s (m)')
            axs[3,1].set_ylabel('Norm. emittance (mm mrad)')
            axs[3,1].set_yscale('log')
            
            axs[4,1].plot(self._driver_evolution['prop_dist'], self._driver_evolution['x_avg']*1e6)
            axs[4,1].plot(self._driver_evolution['prop_dist'], self._driver_evolution['y_avg']*1e6)
            axs[4,1].set_xlabel('s (m)')
            axs[4,1].set_ylabel('Offset, mean (um)')

        
    def get_length(self):
        return self.length
    
    def get_nom_energy_gain(self):
        return self.nom_energy_gain
    
    def energy_efficiency(self):
        return None # TODO
    
    def energy_usage(self):
        return None # TODO
    
    def matched_beta_function(self, energy):
        return beta_matched(self.plasma_density, energy) * self.ramp_beta_mag
        
    def plot_wakefield(self, beam=None):
        pass
        