from abel.classes.stage.stage import Stage
from abel.classes.source.source import Source
from abel.physics_models.spin_tracking import evolve_hills_numeric, compute_fields, tbmt_boris_spin_update, track_spin, plot_spin_tracking
from abel.utilities.plasma_physics import k_p
import numpy as np
import scipy.constants as SI
import copy
import warnings

SI.r_e = SI.physical_constants['classical electron radius'][0]
     
        
class StageSpin(Stage):
    
    def __init__(self, nom_accel_gradient=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=None, transformer_ratio=1, calc_evolution=False):
        
        super().__init__(nom_accel_gradient=nom_accel_gradient, nom_energy_gain=nom_energy_gain, plasma_density=plasma_density, driver_source=driver_source, ramp_beta_mag=ramp_beta_mag)
        
        self.transformer_ratio = transformer_ratio
        self.calc_evolution = calc_evolution
        

        
    
    def track(self, beam_incoming, savedepth=0, runnable=None, verbose=False):

        driver_incoming = self.driver_source.track()

        if self.plasma_density is None:
            self.optimize_plasma_density(beam_incoming)

        beam0 = beam_incoming  # TODO: check this later...

        self._prepare_ramps()

        if self.upramp is not None:
            beam0, driver0 = self.track_upramp(beam_incoming, driver_incoming)
        else:
            beam0 = copy.deepcopy(beam_incoming)
            driver0 = copy.deepcopy(driver_incoming)
            if self.ramp_beta_mag is not None:
                beam0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver_incoming)
                driver0.magnify_beta_function(1/self.ramp_beta_mag, axis_defining_beam=driver_incoming)

        beam = copy.deepcopy(beam0)
        driver = copy.deepcopy(driver0)

        #New part, track spin for each particle
        final_spins = [] 
        all_spins = [] 

        sx = beam.spxs()
        sy = beam.spys()
        sz = beam.spzs()
        x0 = beam.xs()
        ux0 = beam.uxs()
        
        for i in range(len(beam)): #loop over all particles in the beam
            
            S0 = np.array([sx[i], sy[i], sz[i]]) #get the initial spin vector
            
            ss, x, ux = evolve_hills_numeric(x0[i], ux0[i], self.length_flattop, driver0.gammas(), k_p(self.plasma_density)) #simulate transverese motion using Hill's eq
            Es, Bs = compute_fields(ss, x, ux, self.driver_source) #compute electric and magnetic fields along the path
            
            S_hist = track_spin(ss, x, ux, Es, Bs, self.driver_source, S0) #track spin using the T-BMT equation and boris roatation
            all_spins.append(S_hist) 
        
            final_spins.append(S_hist[-1]) #save the full spin history and final spin
        
        beam.set_final_spins(final_spins) #set final spin vectors to the beam

        # ========== Betatron oscillations ==========
        deltaEs = np.full(len(beam.Es()), self.nom_energy_gain_flattop)
        if self.calc_evolution:
            _, evol = beam.apply_betatron_motion(self.length_flattop, self.plasma_density, deltaEs, x0_driver=driver0.x_offset(), y0_driver=driver0.y_offset(), calc_evolution=self.calc_evolution)
            self.evolution.beam = evol
        else:
            beam.apply_betatron_motion(self.length_flattop, self.plasma_density, deltaEs, x0_driver=driver0.x_offset(), y0_driver=driver0.y_offset())

        beam.set_Es(beam.Es() + self.nom_energy_gain_flattop)

        if isinstance(self.driver_source, Source) and (self.driver_source.jitter.xp != 0 or self.driver_source.x_angle != 0 or self.driver_source.jitter.yp != 0 or self.driver_source.y_angle != 0):
            self._rotate_beams_back_to_original(beam, driver0)

        if self.downramp is not None:
            beam_outgoing, driver_outgoing = self.track_downramp(beam, driver)
        else:
            beam_outgoing = copy.deepcopy(beam)
            driver_outgoing = copy.deepcopy(driver)
            if self.ramp_beta_mag is not None:
                beam_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)
                driver_outgoing.magnify_beta_function(self.ramp_beta_mag, axis_defining_beam=driver)

        if self._return_tracked_driver:
            return super().track(beam_outgoing, savedepth, runnable, verbose), driver_outgoing
        else:
            return super().track(beam_outgoing, savedepth, runnable, verbose), all_spins
    
    def optimize_plasma_density(self, source):
        extraction_efficiency = (self.transformer_ratio / 0.75) * abs(source.abs_charge() / self.driver_source.get_charge())


        energy_density_z_extracted = abs(source.get_charge()*self.nom_accel_gradient)
        energy_density_z_wake = energy_density_z_extracted/extraction_efficiency
        norm_blowout_radius = ((32*SI.r_e/(SI.m_e*SI.c**2))*energy_density_z_wake)**(1/4)
        norm_accel_gradient = 1/3 * (norm_blowout_radius)**1.15
        wavebreaking_field = self.nom_accel_gradient / norm_accel_gradient
        plasma_wavenumber = wavebreaking_field/(SI.m_e*SI.c**2/SI.e)
        self.plasma_density = plasma_wavenumber**2*SI.m_e*SI.c**2*SI.epsilon_0/SI.e**2
