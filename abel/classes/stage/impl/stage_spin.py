from abel.classes.stage.stage import Stage
from abel.classes.source.source import Source
from abel.physics_models.spin_tracking import *
from abel.utilities.plasma_physics import k_p
from abel.utilities.relativity import energy2proper_velocity, proper_velocity2energy, momentum2proper_velocity, proper_velocity2momentum, proper_velocity2gamma, energy2gamma, gamma2momentum
import numpy as np
import scipy.constants as SI
import copy
import warnings
from scipy.signal import hilbert

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

        # compute plasma quantities
        omega_p = np.sqrt(self.plasma_density * SI.e**2 / (SI.epsilon_0 * SI.m_e))
        kp = omega_p / SI.c
 
        # Initial particle properties
        sx = beam.spxs()
        sy = beam.spys()
        sz = beam.spzs()
        x0 = beam.xs()
        y0 = beam.ys()
        ux0 = beam.uxs()
        uy0 = beam.uys()
        E0 = np.array(beam.Es())
        gamma0s = energy2gamma(E0)
        
        # set initial radius and length using paper normalized units
        #r0_norm = 0.1                 # 0.1 * c/omega_p
        #L_norm  = 2.4e4               # 2.4e4 * c/omega_p
        
        #r0_m = r0_norm * SI.c / omega_p   # meters  (use this for initial x,y spread)
        #L_m  = L_norm  * SI.c / omega_p   # meters  (use this for stage length)
        
        L = self.length_flattop #0.1275
        n_particles = len(beam)


        final_spins = np.zeros((n_particles, 3), dtype=float)
        all_spins = []
        all_spin_norms = []
        all_ss = []

        # Uniform energy gain for all particles
        #deltaEs = np.full(len(E0), self.nom_energy_gain)
#np.full(len(E0), self.nom_energy_gain_flattop)
        #Ef = E0 + deltaEs
        deltaE_eV = np.full_like(E0, self.nom_energy_gain)
        Ef = E0 + deltaE_eV
        gammaf = energy2gamma(Ef)
        dgamma_ds = (gammaf - gamma0s) / L #(gammaf - gamma0s) / L

        last_ss = None
        last_gamma = None
   
        for i in range(n_particles): #loop over all particles in the beam

            # Initial spin vector
            S0 = np.array([sx[i], sy[i], sz[i]], dtype=float)
            S0 /= np.linalg.norm(S0) if np.linalg.norm(S0) > 0 else 1.0

            # Transverse motion evolution using Hill's equation
            x, ux, gamma_arr = evolve_hills_equation_analytic_evolution(x0[i], ux0[i], L, gamma0s[i], dgamma_ds[i], k_p(self.plasma_density))

            y, uy, _ = evolve_hills_equation_analytic_evolution(y0[i], uy0[i], L, gamma0s[i], dgamma_ds[i], k_p(self.plasma_density))


            N = len(x)
            ss = np.linspace(0, L, N) # Different positions along the particle's path through the stage
            ds = np.diff(ss)

            gamma = np.ravel(gamma_arr) if gamma_arr is not None else (gamma0s[i] + dgamma_ds[i] * ss)
            ux = np.ravel(ux)
            uy = np.ravel(uy)
            beta_x = ux / gamma
            beta_y = uy / gamma
            beta_z = np.sqrt(np.clip(1.0 - beta_x**2 - beta_y**2, 1e-16, 1.0))

            """
            ""
            #Time steps 
            ds = np.diff(ss)
            dt_steps = ds / (beta_z[:-1] * SI.c) # N-1 values? 
            dt_arr = np.concatenate(([dt_steps[0]], dt_steps)) #N-values? 

            #Compute fields along trajectory
            #Es = np.empty((N,3), dtype=float)
            #Bs = np.empty((N,3), dtype=float)
            #for j in range(N):
                #r_vec = np.array([x[j], y[j], ss[j]], dtype=float)
                #Es[j] = plasma_E_field(r_vec, j, k_p(self.plasma_density))
                #Bs[j] = plasma_B_field(r_vec, j, k_p(self.plasma_density))
            
            #Track spin along the trajectory
            S_hist = np.empty((N, 3), dtype=float)
            S_hist[0,:] = S0
            spin_norms = []

            
            for j in range(1, N):
                beta_vec = np.array([beta_x[j - 1], beta_y[j - 1], beta_z[j - 1]], dtype=float)
                dt = dt_arr[j] #time spent between step j-1 and j
                gamma_val = gamma[j - 1]
                #E = Es[j - 1]
                #B = Bs[j - 1]
                r_vec = np.array([x[j], y[j], ss[j]], dtype=float)
                E = plasma_E_field(r_vec, j, k_p(self.plasma_density), Ez0=Ez0)
                B = plasma_B_field(r_vec, j, k_p(self.plasma_density), beta_z=beta_z[j - 1])
                
                S_next = tbmt_boris_spin_update(S_hist[j -1], E, B, beta_vec, gamma_val, dt)
                
                if not np.all(np.isfinite(S_next)):
                    # revert to previous normalized spin
                    S_next = S_hist[j - 1].copy()
                nrm = np.linalg.norm(S_next)
                
                if nrm < 1e-12 or not np.isfinite(nrm):
                    S_next = S_hist[j -1].copy()
                else:
                    S_next /= nrm
               
                spin_norms.append(np.linalg.norm(S_next))

                S_hist[j, :] = S_next
            """
            
            
            #dt_steps = ds / (beta_z[:-1] * SI.c)
            #dt_steps[~np.isfinite(dt_steps)] = 1e-20
            
            # Ensure dt_steps has the correct length
            #if len(dt_steps) != N - 1:
                #dt_steps = np.full(
                    #N - 1,
                    #np.mean(dt_steps[np.isfinite(dt_steps)]) if np.any(np.isfinite(dt_steps)) else 1e-6
                #)
            
            S_hist = np.empty((N,3))
            S_hist[0, :] = S0
            a_e = 0.00115965218128
            tiny = 1e-20
            theta_max = 0.5

            for j in range(1, N):
                gamma_t = max(gamma[j-1], tiny)
                x_t = x[j-1]   # meters (ensure evolve_hills returns meters)
                y_t = y[j-1]
                
                alpha_E = (SI.m_e * omega_p**2) / (2.0 * -SI.e)
                #pref_omega_s = (e / m_e) * (a_e + 1.0 / gamma_t) * (alpha_E / c*gamma[j-1])
                #alpha_loc = (kp**2) / (2.0 * gamma_t) #0.5
                pref_omega_s = (SI.e / SI.m_e) * (a_e + 1.0 / gamma_t) * (alpha_E / SI.c)
                # pref dimensionless * 1/m^2 => 1/m^2; multiplied by x (m) -> 1/m
                #pref = alpha_loc * (a_e + 1.0 / gamma_t)
            
                Omega = np.array([-pref_omega_s * x_t, pref_omega_s * y_t, 0.0])
                Omega_mag = np.linalg.norm(Omega)
                
                ds_j = max(ds[j-1], 0.0)
                dt = ds_j / (beta_z[j-1] * SI.c)  
                
                if Omega_mag < 1e-18 or ds_j <= 0:
                    S_hist[j,:] = S_hist[j-1, :]
                    continue
                
                delta = Omega_mag * dt
                d = (Omega / Omega_mag) * np.tan(0.5 * delta)  # dimensionless
                s_prev = S_hist[j-1,:]
                s_prime = s_prev + np.cross(s_prev, d)
                s_next  = s_prev + 2.0*np.cross(s_prime, d)/(1.0 + np.dot(d,d))

                s_next /= np.linalg.norm(s_next)
                S_hist[j,:] = s_next / np.linalg.norm(s_next)
    
            # after loop
            # after finishing the j-loop for particle i
            spin_norms = np.linalg.norm(S_hist, axis=1)
            all_spin_norms.append(spin_norms)
            all_spins.append(S_hist.copy())
            S_last = S_hist[-1, :].copy()   # last spin for this particle
            final_spins[i, :] = S_last      # assign into row i
             
        last_ss = ss
        last_gamma = gamma



        print(f"Final spins stds: sx_std, sy_std, sz_std =", np.std(final_spins[:, 0]), np.std(final_spins[:, 1]), np.std(final_spins[:, 2]))
        # ensure shape and finiteness
        print("alpha_E [V/m]:", alpha_E)
        print("pref_omega_s [1/s/m]:", pref_omega_s)
        print(f"beta {beta_z}")
        assert final_spins.shape == (n_particles, 3)
        if not np.all(np.isfinite(final_spins)):
            warnings.warn("NaN/Inf found in final_spins", RuntimeWarning)

        # optional plotting (first particle only to inspect)
        if len(all_spins) > 0 and last_ss is not None and last_gamma is not None:
            try:
                plot_spin_tracking(all_spins, last_ss)
                plot_spin_tracking_gamma(all_spins, last_gamma)
            except Exception:
                warnings.warn("Spin plots could not be generated.", RuntimeWarning)
                
        plt.figure()
        for norms in all_spin_norms:
            plt.plot(ss/1e3, norms, alpha=0.3)   # one curve per particle
        plt.title("Spin vector norm along stage")
        plt.xlabel("s [m]")
        plt.ylabel("|S|")


        min_steps = min(sp.shape[0] for sp in all_spins)
        S_arr = np.array([sp[:min_steps] for sp in all_spins]).transpose(1, 0, 2)
        P_vec = np.mean(S_arr, axis=1)
        P_evol = np.linalg.norm(P_vec, axis=1) 
        Pz_evol = np.mean(S_arr[:, :, 2], axis=1)       
        P0_vec = np.mean(S_arr[0, :, :], axis=0)
        P0_mag = np.linalg.norm(P0_vec)
        P0z = P0_vec[2]
        Dz = ((P0z - Pz_evol) / max(abs(P0z), tiny)) * 100.0
        D  = ((P0_mag - P_evol) / max(P0_mag, tiny)) * 100.0
        final_D = (P0_mag - P_evol[-1]) / P0_mag * 100
        
        analytic_signal = hilbert(D) 
        envelope = np.abs(analytic_signal)
        analytic_signal_z = hilbert(Dz)
        envelope_z = np.abs(analytic_signal_z)
        ss_plot = last_ss[:min_steps]
        plt.figure()
        #plt.plot(ss_plot, Dz, label=r'$\Delta P_z/P_{z0}$')
        #plt.plot(ss_plot, D,  label=r'$\Delta P/P_0$')
        plt.plot(last_gamma/1e5, Dz, label=r'$\Delta P_z/P_{z0}$')
        plt.plot(last_gamma/1e5, envelope, label=r'Envelope $|\Delta P/P_0|$', linewidth=2)
        
        plt.plot(last_gamma/1e5, D,  label=r'$\Delta P/P_0$')
        plt.plot(last_gamma/1e5, envelope_z, label=r'Envelope $|\Delta P_z/P_{z0}|$', linewidth=2, linestyle='--')

        plt.xlabel("Stage length [m]")
        plt.xlabel("Gamma [10^5]")
        plt.ylabel("Depolarization [%]")
        #plt.ylim(min(D) + 0.25*min(D), 0)
        plt.legend()
        plt.grid(True)  


        data = D
        #np.savetxt("depolarization_vs_gradient.txt", data, header="plasma_density[m^-3] depolarization[%]", fmt="%.5e")
        #print("✅ Saved results to depolarization_vs_density.txt")
        # --- prepare arrays (safe for variable step counts) ---
        min_steps = min(sp.shape[0] for sp in all_spins)
        ss_cut = ss[:min_steps]   # positions
        S_arr = np.array([sp[:min_steps] for sp in all_spins])  # shape (n_particles, min_steps, 3)
        S_arr = S_arr.transpose(1,0,2)  # shape (min_steps, n_particles, 3)
        n_steps, n_part, _ = S_arr.shape
        
        # --- polarization vector evolution (ensemble mean) ---
        P_vec = np.mean(S_arr, axis=1)          # shape (n_steps, 3)
        P_mag = np.linalg.norm(P_vec, axis=1)   # ensemble polarization magnitude vs s
        
        # --- per-particle norms and deviations ---
        norms = np.linalg.norm(S_arr, axis=2)   # shape (n_steps, n_particles)
        # mean norm deviation from 1.0 (absolute), and std across ensemble
        mean_norm = np.mean(norms, axis=1)
        std_norm  = np.std(norms, axis=1)
        dev_from_one = mean_norm - 1.0
        
        # --- envelope of ensemble polarization decay (optional) ---
        analytic = hilbert(P_mag - P_mag.mean())   # remove mean for envelope
        envelope = np.abs(analytic)
        
        # --- Depolarization metrics ---
        P0_vec = np.mean(S_arr[0,:,:], axis=0)
        P0_mag = np.linalg.norm(P0_vec)
        final_P_vec = P_vec[-1,:]
        final_P_mag = np.linalg.norm(final_P_vec)
        final_depol_percent = (P0_mag - final_P_mag) / (P0_mag + 1e-30) * 100.0
        self.final_depolarization = final_D
        
        print("P0_mag:", P0_mag)
        print("final_P_mag:", final_P_mag)
        print("final depolarization [%]:", final_depol_percent)
        
        # --- Plot 1: mean norm deviation in ppm (clear visualization) ---
        plt.figure(figsize=(8,4))
        plt.plot(ss_cut, (dev_from_one)*1e6, label='mean Δ|S| (ppm)')
        plt.fill_between(ss_cut, (dev_from_one - std_norm)*1e6, (dev_from_one + std_norm)*1e6,
                         color='C0', alpha=0.2, label='±1σ')
        plt.xlabel("s [m]")
        plt.ylabel("Δ|S| [ppm]")
        plt.title("Spin norm deviation (mean ± σ) along stage")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # --- Plot 2: Ensemble polarization magnitude + envelope and percent depol ---
        plt.figure(figsize=(8,4))
        plt.plot(ss_cut, P_mag, label='|⟨S⟩|(s)')
        plt.plot(ss_cut, envelope + P_mag.mean(), '--', label='Envelope (abs)')
        plt.xlabel("s [m]")
        plt.ylabel("Polarization magnitude |⟨S⟩|")
        plt.title(f"Ensemble polarization (final depol = {final_depol_percent:.3e} %)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # --- Plot 3: fractional depolarization vs s in ppm ---
        frac_depol = (P0_mag - P_mag) / (P0_mag + 1e-30)  # fraction
        plt.figure(figsize=(8,4))
        plt.plot(ss_cut, frac_depol*1e6, label='ΔP/P0 [ppm]')
        plt.xlabel("s [m]")
        plt.ylabel("Depolarization [ppm]")
        plt.title("Fractional depolarization along stage")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        min_steps = min(sp.shape[0] for sp in all_spins)
        S_init = np.array([sp[0,:] for sp in all_spins])   # (n_part, 3)
        S_final = np.array([sp[min_steps-1,:] for sp in all_spins])
        
        P0_vec = np.mean(S_init, axis=0)
        Pf_vec = np.mean(S_final, axis=0)
        P0_mag = np.linalg.norm(P0_vec)
        Pf_mag = np.linalg.norm(Pf_vec)
        delta = Pf_mag - P0_mag
        delta_percent = (P0_mag - Pf_mag)/ (P0_mag + 1e-30) * 100.0
        
        print("P0_mag:", P0_mag)
        print("Pf_mag:", Pf_mag)
        print("Delta (Pf - P0) :", delta)
        print("Delta percent (signed):", delta_percent, "%")
        
        # Bootstrap uncertainty on delta_percent
        npart = S_init.shape[0]
        nboot = 2000
        rng = np.random.default_rng(12345)
        boot_vals = np.empty(nboot)
        idx = np.arange(npart)
        for k in range(nboot):
            sel = rng.choice(idx, size=npart, replace=True)
            P0_b = np.linalg.norm(np.mean(S_init[sel], axis=0))
            Pf_b = np.linalg.norm(np.mean(S_final[sel], axis=0))
            boot_vals[k] = (P0_b - Pf_b) / (P0_b + 1e-30) * 100.0
        
        mean_boot = np.mean(boot_vals)
        ci_low, ci_high = np.percentile(boot_vals, [2.5, 97.5])
        print(f"Bootstrap mean depol [%]: {mean_boot:.6e}, 95% CI = [{ci_low:.6e}, {ci_high:.6e}]")
        return final_D
                
        beam.plot_spins()
        beam.set_spxs(final_spins[:, 0])
        beam.set_spys(final_spins[:, 1])
        beam.set_spzs(final_spins[:, 2])
        beam.plot_spins()
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
