from opal import Stage, Beam, CONFIG
import scipy.constants as SI
from matplotlib import pyplot as plt
from opal.utilities.plasma_physics import blowout_radius, k_p, beta_matched
from opal.apis.wake_t.wake_t_api import beam2wake_t_bunch, wake_t_bunch2beam
import numpy as np
import wake_t
from openpmd_viewer import OpenPMDTimeSeries


class StageWakeT(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None):
        
        super().__init__(length, nom_energy_gain, plasma_density)
        
        self.driver_source = driver_source

        self._beam_evolution = None
        self._driver_evolution = None


        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # make driver (and convert to WakeT bunch)
        driver0 = self.driver_source.track()
        driver0_wake_t = beam2wake_t_bunch(driver0, name='driver')
        
        # convert beam to WakeT bunch
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
        
        # convert back to OPAL beam
        beam = wake_t_bunch2beam(bunches[1][-1])
        
        return super().track(beam, savedepth, runnable, verbose)
    
    
    def plot_evolution(self):
        
        if self._beam_evolution is not None and self._driver_evolution is not None:
            
            # prepare the figure
            fig, axs = plt.subplots(2, 6)
            fig.set_figwidth(CONFIG.plot_width_default)
            fig.set_figheight(9)

            # beam plots
            axs[0,0].plot(self._beam_evolution['prop_dist'], self._beam_evolution['beta_x'])
            axs[0,0].plot(self._beam_evolution['prop_dist'], self._beam_evolution['beta_y'])
            axs[0,0].set_xlabel('s (m)')
            axs[0,0].set_ylabel('Beta function (m)')
            axs[0,0].set_yscale('log')
            axs[0,0].set_title('Beam')

            axs[0,0].plot(self._beam_evolution['prop_dist'], self._beam_evolution['emit_nx'])
            axs[0,0].plot(self._beam_evolution['prop_dist'], self._beam_evolution['emit_ny'])
            axs[0,0].set_xlabel('s (m)')
            axs[0,0].set_ylabel('Beta function (m)')
            axs[0,0].set_yscale('log')
            axs[0,0].set_title('Beam')

            # driver plots
            axs[1].plot(self._beam_evolution['prop_dist'], self._beam_evolution['avg_ene']*SI.m_e*SI.c**2/SI.e/1e9)
            axs[1].plot(self._driver_evolution['prop_dist'], self._driver_evolution['avg_ene']*SI.m_e*SI.c**2/SI.e/1e9)
            axs[1].set_xlabel('s (m)')
            axs[1].set_ylabel('Mean energy (GeV)')
            axs[1].set_title('Driver')

        
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
        
    def plot_wakefield(self, beam=None):
        pass
        