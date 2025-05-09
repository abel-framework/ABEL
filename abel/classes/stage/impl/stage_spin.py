from abel.classes.stage.stage import Stage
from abel.classes.source.source import Source
from abel.physics_models.spin_tracking import *
from abel.utilities.plasma_physics import k_p
from abel.utilities.relativity import energy2proper_velocity, proper_velocity2energy, momentum2proper_velocity, proper_velocity2momentum, proper_velocity2gamma, energy2gamma, gamma2momentum
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
    
        final_spins = [] 
        all_spins = [] 

        sx = beam.spxs()
        sy = beam.spys()
        sz = beam.spzs()
        x0 = beam.xs()
        y0 = beam.ys()
        ux0 = beam.uxs()
        uy0 = beam.uys()
        E0 = beam.Es()
        L = self.length_flattop
        gamma0s = energy2gamma(E0)
        n_particles = len(beam)

        # Uniform energy gain for all particles
        deltaEs = np.full(len(E0), self.nom_energy_gain_flattop)
        Ef = E0 + deltaEs
        gammaf = energy2gamma(Ef)
        dgamma_ds = (gammaf-gamma0s)/L
        
        for i in range(n_particles): #loop over all particles in the beam

            #Get individual values
            S0 = np.array([sx[i], sy[i], sz[i]]) #get the initial spin vector

            # Transverse motion evolution using Hill's equation
            x, ux, gamma = evolve_hills_equation_analytic(x0[i], ux0[i], L, gamma0s[i], dgamma_ds[i], k_p(self.plasma_density))

            y, uy, _ = evolve_hills_equation_analytic(y0[i], uy0[i], L, gamma0s[i], dgamma_ds[i], k_p(self.plasma_density))

           
            
            N = len(x)
            ss = np.linspace(0, L, N)  # Different positions along the particle's path through the stage
            gamma = gamma0s[i] + dgamma_ds[i] * ss / L

            #Transverse + longitudinal velocities
            ux = np.ravel(ux)
            uy = np.ravel(uy)
            gamma_vals = gamma
            beta_x = ux/SI.c
            beta_y = uy/SI.c
            total_beta2 = 1.0 - 1.0 / gamma_vals**2
            beta_z = np.sqrt(np.clip(total_beta2 - beta_x**2 - beta_y**2, 1e-12, None))

            #Time steps 
            ds = np.diff(ss)
            dt_steps = ds / (beta_z[:-1] * SI.c)
            dt_arr = np.concatenate(([dt_steps[0]], dt_steps))

            #Compute fields along trajectory
            Es = []
            Bs = []
            for j in range(N):
                r_vec = np.array([x[j], y[j], ss[j]])
                E = plasma_E_field(r_vec, j, k_p(self.plasma_density))
                B = plasma_B_field(r_vec, j, k_p(self.plasma_density))
                Es.append(E)
                Bs.append(B)
            Es, Bs = np.array(Es), np.array(Bs)
            
        #Track spin
        S_hist = np.zeros((N, 3))
        S_hist[0] = S0
        for j in range(1, N):
            S_prev = S_hist[j - 1]
            print(S_prev)
            beta_vec = np.array([beta_x[j - 1], beta_y[j - 1], beta_z[j - 1]])
            dt = dt_arr[j - 1]
            gamma_val = gamma_vals[j - 1]
            E = Es[j - 1]
            B = Bs[j - 1]
            S_hist[j] = tbmt_boris_spin_update(S_prev, E, B, beta_vec, gamma_val, dt)
            print(S_hist)
            S_hist[j] /= np.linalg.norm(S_hist[j])  # Normalize spin

        final_spins.append(S_hist[-1])
        all_spins.append(S_hist)
        
        plot_spin_tracking(all_spins, ss)

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
