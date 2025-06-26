from abc import abstractmethod
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled
from types import SimpleNamespace
import numpy as np
import scipy.constants as SI

class Interstage(Trackable, CostModeled):
    
    @abstractmethod
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0, lens_radius=2e-3, charge_sign=-1,
                 cancel_chromaticity=True, cancel_sec_order_dispersion=False, use_apertures=True, enable_csr=True, enable_isr=True, enable_space_charge=False):
        
        super().__init__()

        # main parameters
        self.nom_energy = nom_energy
        self._R56 = R56
        self.lens_radius = lens_radius
        self.charge_sign = charge_sign

        # main parameters (functions of energy)
        self._beta0 = beta0
        self._length_dipole = length_dipole
        self._field_dipole = field_dipole

        # length ratios
        self.length_ratio_gap = 0.025
        self.length_ratio_plasma_lens = 0.05
        self.length_ratio_chicane_dipole = 0.85
        self.length_ratio_central_gap_or_sextupole = 0.25

        # derivable (but also settable) parameters
        self._field_ratio_chicane_dipole1 = None
        self._field_ratio_chicane_dipole2 = None
        self._strength_plasma_lens = None # [1/m]
        self._nonlinearity_plasma_lens = None # [1/m]
        self._strength_sextupole = None # [1/m^2]
        
        # feature toggles
        self.cancel_chromaticity = cancel_chromaticity
        self.cancel_sec_order_dispersion = cancel_sec_order_dispersion
        self.use_apertures = use_apertures

        # physics flags
        self.enable_csr = enable_csr
        self.enable_isr = enable_isr
        self.enable_space_charge = enable_space_charge
        
        # lens alignment and jitter
        self.lens_offset_x = 0
        self.lens_offset_y = 0
        self.jitter = SimpleNamespace()
        self.jitter.lens_offset_x = 0
        self.jitter.lens_offset_y = 0
        self.jitter.lens_angle_x = 0
        self.jitter.lens_angle_y = 0
        
        # evolution (saved when tracking)
        self.evolution = SimpleNamespace()

    
    ## TRACKING

    @abstractmethod
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)


    ## OVERALL LENGTH
    
    # lattice length
    def get_length(self):
        if self.length_dipole is not None:
            ls, *_ = self.matrix_lattice(k_lens=0, tau_lens=0, B_chic1=0, B_chic2=0, m_sext=0, half_lattice=False)
            return np.sum(ls)
        else:
            return None

    
    ## ENERGY-FUNCTION PARAMETERS
    
    @property
    def R56(self) -> float:
        if callable(self._R56):
            return self._R56(self.nom_energy)
        else:
            return self._R56
    @R56.setter
    def R56(self, val):
        self._R56 = val
        
    @property
    def beta0(self) -> float:
        if callable(self._beta0):
            return self._beta0(self.nom_energy)
        else:
            return self._beta0
    @beta0.setter
    def beta0(self, val):
        self._beta0 = val

    @property
    def field_dipole(self) -> float:
        if callable(self._field_dipole):
            return self._field_dipole(self.nom_energy)
        else:
            return self._field_dipole
    @field_dipole.setter
    def field_dipole(self, val):
        self._field_dipole = val

    @property
    def length_dipole(self) -> float:
        if callable(self._length_dipole):
            return self._length_dipole(self.nom_energy)
        else:
            return self._length_dipole
    @length_dipole.setter
    def length_dipole(self, val):
        self._length_dipole = val
    
    
    ## RATIO-DEFINED LENGTHS
    
    @property
    def length_gap(self) -> float:
        return self.length_dipole * self.length_ratio_gap
        
    @property
    def length_plasma_lens(self) -> float:
        return self.length_dipole * self.length_ratio_plasma_lens

    @property
    def length_chicane_dipole(self) -> float:
        return self.length_dipole * self.length_ratio_chicane_dipole

    @property
    def length_central_gap_or_sextupole(self) -> float:
        return self.length_dipole * self.length_ratio_central_gap_or_sextupole

    
    ## STRENGTH VALUES

    @property
    def strength_plasma_lens(self) -> float:
        if self._strength_plasma_lens is None:
            self.match_beta_function()
        return self._strength_plasma_lens

    @property
    def nonlinearity_plasma_lens(self) -> float:
        if self._nonlinearity_plasma_lens is None:
            self.match_chromatic_amplitude()
        return self._nonlinearity_plasma_lens
        
    @property
    def strength_sextupole(self) -> float:
        if self._strength_sextupole is None:
            self.match_second_order_dispersion()
        return self._strength_sextupole


    
    ## FIELD VALUES
    
    @property
    def field_gradient_plasma_lens(self) -> float:
        "Plasma-lens field gradient [T/m]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.charge_sign*self.strength_plasma_lens*p0/(SI.e*self.length_plasma_lens)

    @property
    def field_chicane_dipole1(self) -> float:
        if self._field_ratio_chicane_dipole1 is None:
            self.match_dispersion_and_R56()
        return self.field_dipole * self._field_ratio_chicane_dipole1

    @property
    def field_chicane_dipole2(self) -> float:
        if self._field_ratio_chicane_dipole2 is None:
            self.match_dispersion_and_R56()
        return self.field_dipole * self._field_ratio_chicane_dipole2

    @property
    def field_gradient_sextupole(self) -> float:
        "Sextupole field gradient [T/m^2]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.charge_sign*self.strength_sextupole*p0/(SI.e*self.length_central_gap_or_sextupole)

    
    
    ## MATRIX LATTICE

    # full lattice 
    def matrix_lattice(self, k_lens=None, tau_lens=None, B_chic1=None, B_chic2=None, m_sext=None, half_lattice=False):

        from abel.utilities.relativity import energy2momentum
        
        # element length array
        dL = self.length_gap
        ls = np.array([dL, self.length_dipole, dL, self.length_plasma_lens, dL, 
                       self.length_chicane_dipole, dL, self.length_chicane_dipole, dL, self.length_central_gap_or_sextupole/2])
        
        # bending strength array
        if B_chic1 is None:
            B_chic1 = self.field_chicane_dipole1
        if B_chic2 is None:
            B_chic2 = self.field_chicane_dipole2
        Bs = np.array([0, self.field_dipole, 0, 0, 0, B_chic1, 0, B_chic2, 0, 0])
        inv_rhos = -self.charge_sign * Bs * SI.e / energy2momentum(self.nom_energy)
        
        # focusing strength array
        if k_lens is None:
            k_lens = self.strength_plasma_lens/self.length_plasma_lens
        ks = np.array([0, 0, 0, k_lens, 0, 0, 0, 0, 0, 0])
        
        # sextupole strength array
        if m_sext is None:
            if self.cancel_sec_order_dispersion:
                m_sext = self.strength_sextupole/self.length_central_gap_or_sextupole
            else:
                m_sext = 0
        ms = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, m_sext])

        # plasma-lens transverse taper array
        if tau_lens is None:
            tau_lens = self.nonlinearity_plasma_lens
        taus = np.array([0, 0, 0, tau_lens, 0, 0, 0, 0, 0, 0])
        
        # mirror symmetrize the lattice
        if not half_lattice:
            ls = np.append(np.append(ls[:-1], 2*ls[-1]), np.flip(ls[:-1]))
            inv_rhos = np.append(np.append(inv_rhos[:-1], inv_rhos[-1]), np.flip(inv_rhos[:-1]))
            ks = np.append(np.append(ks[:-1], ks[-1]), np.flip(ks[:-1]))
            ms = np.append(np.append(ms[:-1], ms[-1]), np.flip(ms[:-1]))
            taus = np.append(np.append(taus[:-1], taus[-1]), np.flip(taus[:-1]))
        
        return ls, inv_rhos, ks, ms, taus
        

    
    ## MATCHING
    
    def match(self):
        "Combined matching the beta function, first- and second-order dispersion and the R56"
        self.match_beta_function()
        self.match_dispersion_and_R56(high_res=False)
        if self.cancel_chromaticity:
            self.match_chromatic_amplitude()
        if self.cancel_sec_order_dispersion:
            self.match_second_order_dispersion()

    
    def match_beta_function(self):
        "Matching the beta function by adjusting the plasma-lens strength."
        
        # minimizer function for beta matching (central alpha function is zero)
        from abel.utilities.beam_physics import evolve_beta_function
        def minfun_beta(params):
            ls, _, ks, _, _ = self.matrix_lattice(k_lens=params[0], tau_lens=0, B_chic1=0, B_chic2=0, m_sext=0, half_lattice=True)
            _, alpha, _ = evolve_beta_function(ls, ks, self.beta0, fast=True) 
            return alpha**2
    
        # initial guess for the lens strength
        f = 1/(1/(self.length_dipole + self.length_plasma_lens/2 + 2*self.length_gap) + 1/(2*self.length_chicane_dipole + self.length_central_gap_or_sextupole/2 + self.length_plasma_lens/2 + 3*self.length_gap))
        k_lens0 = 1/(f*self.length_plasma_lens)
        
        # match the beta function
        from scipy.optimize import minimize
        result_beta = minimize(minfun_beta, k_lens0, tol=1e-20, options={'maxiter': 200})
        self._strength_plasma_lens = result_beta.x[0]*self.length_plasma_lens

    
    def match_dispersion_and_R56(self, high_res=False):
        "Cancelling the dispersion and matchign the R56 by adjusting the chicane dipoles."
        
        # assume negative R56
        nom_R56 = -abs(self.R56)
        if self.R56 > 0:
            print('Positive R56 given, flipping sign to negative')
        
        # normalizing scale for the merit function
        Dpx_scale = self.length_dipole*self.field_dipole*SI.c/self.nom_energy
        R56_scale = self.length_dipole**3*self.field_dipole**2*SI.c**2/self.nom_energy**2
        
        # minimizer function for dispersion (central dispersion prime is zero)
        from abel.utilities.beam_physics import evolve_dispersion, evolve_R56
        def minfun_dispersion_R56(params):
            ls, inv_rhos, ks, _, _ = self.matrix_lattice(tau_lens=0, B_chic1=params[0], B_chic2=params[1], m_sext=0, half_lattice=True)
            _, Dpx_mid, _ = evolve_dispersion(ls, inv_rhos, ks, fast=True) 
            R56_mid, _ = evolve_R56(ls, inv_rhos, ks, high_res=high_res) 
            return (Dpx_mid/Dpx_scale)**2 + ((R56_mid - nom_R56/2)/R56_scale)**2

        # initial guess for the chicane dipole fields
        B_chic1_guess = self.field_dipole/2
        B_chic2_guess = -self.field_dipole/2
        
        # match the beta function
        from scipy.optimize import minimize
        result_dispersion_R56 = minimize(minfun_dispersion_R56, [B_chic1_guess, B_chic2_guess], tol=1e-16, options={'maxiter': 50})
        self._field_ratio_chicane_dipole1 = result_dispersion_R56.x[0]/self.field_dipole
        self._field_ratio_chicane_dipole2 = result_dispersion_R56.x[1]/self.field_dipole
    
    
    def match_chromatic_amplitude(self):
        "Matching the chroaticity of function by adjusting the plasma-lens nonlinearity."

        # stop if nonlinearity is turned off
        if not self.cancel_chromaticity:
            self._nonlinearity_plasma_lens = 0.0
            return
        
        # normalizing scale for the merit function
        W_scale = 2*self.length_dipole/self.beta0
        
        # minimizer function for beta matching (central alpha function is zero)
        from abel.utilities.beam_physics import evolve_chromatic_amplitude
        def minfun_W(params):
            ls, inv_rhos, ks, ms, taus = self.matrix_lattice(tau_lens=params[0], m_sext=0, half_lattice=True)
            W_mid, _ = evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, self.beta0, fast=True) 
            return (W_mid/W_scale)**2
        
        # calculate dispersion and dispersion prime in the lens
        from abel.utilities.relativity import energy2momentum
        Dpx_lens = -self.charge_sign*self.length_dipole*self.field_dipole*SI.e/energy2momentum(self.nom_energy)
        Dx_lens = Dpx_lens*(self.length_dipole/2 + self.length_gap + self.length_plasma_lens/2)
        tau_lens0 = -1/Dx_lens
            
        # match the beta function
        from scipy.optimize import minimize
        result_W = minimize(minfun_W, tau_lens0, tol=1e-16, options={'maxiter': 100})
        self._nonlinearity_plasma_lens = result_W.x[0]
    
        
    def match_second_order_dispersion(self):
        "Cancelling the second-order dispersion by adjusting the sextupole strength."

        # stop if nonlinearity is turned off
        if not self.cancel_sec_order_dispersion:
            self._strength_sextupole = 0.0
            return

        # normalizing scale for the merit function
        DDpx_scale = self.length_dipole*self.field_dipole*SI.c/self.nom_energy
        
        # minimizer function for second-order dispersion (central second-order dispersion prime is zero)
        from abel.utilities.beam_physics import evolve_second_order_dispersion
        def minfun_second_order_dispersion(params):
            ls, inv_rhos, ks, ms, taus = self.matrix_lattice(m_sext=params[0], half_lattice=True)
            _, DDpx, _ = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=True) 
            return (DDpx/DDpx_scale)**2

        # guesstimate the sextupole strength (starting point for optimization)
        Dx_scale = -self.length_dipole**2*self.field_dipole*SI.c/self.nom_energy/2
        m_sext0 = 1.11/(Dx_scale*self.length_central_gap_or_sextupole*self.length_dipole)
    
        # match the beta function
        from scipy.optimize import minimize
        result_dispersion = minimize(minfun_second_order_dispersion, m_sext0, method='Nelder-Mead', tol=1e-20, options={'maxiter': 50})
        self._strength_sextupole = result_dispersion.x[0]*self.length_central_gap_or_sextupole
        
    
    ## PLOTTING
    
    def plot_evolution(self):

        from matplotlib import pyplot as plt
        
        evol = self.evolution
        
        # prepare plot
        fig, axs = plt.subplots(3,3)
        fig.set_figwidth(20)
        fig.set_figheight(12)
        col0 = "tab:gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        long_label = 'Location [m]'
        long_limits = [min(evol.location), max(evol.location)]

        # plot energy
        axs[0,0].plot(evol.location, evol.energy / 1e9, color=col1)
        axs[0,0].set_ylabel('Energy [GeV]')
        axs[0,0].set_xlabel(long_label)
        axs[0,0].set_xlim(long_limits)
        
        # plot charge
        axs[0,1].plot(evol.location, abs(evol.charge[0]) * np.ones(evol.location.shape) * 1e9, ':', color=col0)
        axs[0,1].plot(evol.location, abs(evol.charge) * 1e9, color=col1)
        axs[0,1].set_ylabel('Charge [nC]')
        axs[0,1].set_xlim(long_limits)
        axs[0,1].set_ylim(0, abs(evol.charge[0]) * 1.3 * 1e9)
        
        # plot normalized emittance
        axs[0,2].plot(evol.location, evol.emit_ny*1e6, color=col2)
        axs[0,2].plot(evol.location, evol.emit_nx*1e6, color=col1)
        axs[0,2].set_ylabel('Emittance, rms [mm mrad]')
        axs[0,2].set_xlim(long_limits)
        axs[0,2].set_yscale('log')
        
        # plot energy spread
        axs[1,0].plot(evol.location, evol.rel_energy_spread*1e2, color=col1)
        axs[1,0].set_ylabel('Energy spread, rms [%]')
        axs[1,0].set_xlabel(long_label)
        axs[1,0].set_xlim(long_limits)
        axs[1,0].set_yscale('log')

        # plot bunch length
        axs[1,1].plot(evol.location, evol.bunch_length*1e6, color=col1)
        axs[1,1].set_ylabel(r'Bunch length, rms [$\mathrm{\mu}$m]')
        axs[1,1].set_xlabel(long_label)
        axs[1,1].set_xlim(long_limits)

        # plot beta function
        axs[1,2].plot(evol.location, evol.beta_y, color=col2)  
        axs[1,2].plot(evol.location, evol.beta_x, color=col1)
        axs[1,2].set_ylabel('Beta function [m]')
        axs[1,2].set_xlabel(long_label)
        axs[1,2].set_xlim(long_limits)
        axs[1,2].set_yscale('log')
        
        # plot transverse offset
        axs[2,0].plot(evol.location, evol.y*1e6, color=col2)  
        axs[2,0].plot(evol.location, evol.x*1e6, color=col1)
        axs[2,0].set_ylabel(r'Transverse offset [$\mathrm{\mu}$m]')
        axs[2,0].set_xlabel(long_label)
        axs[2,0].set_xlim(long_limits)
        
        # plot dispersion
        axs[2,1].plot(evol.location, evol.dispersion_y*1e3, color=col2)  
        axs[2,1].plot(evol.location, evol.dispersion_x*1e3, color=col1)
        #axs[2,1].plot(evol.location, evol.second_order_dispersion_x*1e3, ':', color=col1)
        axs[2,1].set_ylabel('First-order dispersion [mm]')
        axs[2,1].set_xlabel(long_label)
        axs[2,1].set_xlim(long_limits)

        # plot beam size
        axs[2,2].plot(evol.location, evol.beam_size_y*1e6, color=col2)  
        axs[2,2].plot(evol.location, evol.beam_size_x*1e6, color=col1)
        axs[2,2].set_ylabel(r'Beam size, rms [$\mathrm{\mu}$m]')
        axs[2,2].set_xlabel(long_label)
        axs[2,2].set_xlim(long_limits)
        
        
        plt.show()

    
    ## PLOTTING OPTICS

    def plot_optics(self, savefig=None):

        from matplotlib import pyplot as plt
        from matplotlib import patches
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_second_order_dispersion, evolve_R56

        # calculate evolution
        ls, inv_rhos, ks, ms, taus = self.matrix_lattice()
        _, _, evol_beta = evolve_beta_function(ls, ks, self.beta0, fast=False)
        _, _, evol_second_order_dispersion = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=False);
        _, evol_R56 = evolve_R56(ls, inv_rhos, ks);
        ssl = np.append([0.0], np.cumsum(ls))
        
        # extract into readable format
        ss_beta = evol_beta[0]
        betas = evol_beta[1]
        ss_disp2 = evol_second_order_dispersion[0]
        second_order_dispersion = evol_second_order_dispersion[2]
        ss_disp1 = ss_disp2
        dispersion = evol_second_order_dispersion[1]
        ss_R56 = evol_R56[0]
        R56 = evol_R56[1]
        
        # prepare plots
        fig, axs = plt.subplots(4,1, gridspec_kw={'height_ratios': [0.1, 1, 1, 1]})
        fig.set_figwidth(7)
        fig.set_figheight(11)
        col0 = "tab:gray"
        colx1 = "tab:blue"
        colx2 = "#d7e9f5" # lighter version of tab:blue
        colz = "tab:green"
        long_label = 'Location [m]'
        long_limits = [min(ss_beta), max(ss_beta)]

        # layout
        axs[0].plot(ss_beta, np.zeros_like(ss_beta), '-', linewidth=0.5, color='k')
        axs[0].axis('off')
        for i in range(len(ls)):
            if abs(inv_rhos[i]) > 0: # add dipoles
                axs[0].add_patch(patches.Rectangle((ssl[i],-0.75), ls[i], 1.5, fc='#d9d9d9'))
            if abs(ks[i]) > 0: # add plasma lenses
                axs[0].add_patch(patches.Rectangle((ssl[i],0), ls[i], 1, fc='#fcb577'))
            if abs(ms[i]) > 0: # add sextupole
                axs[0].add_patch(patches.Rectangle((ssl[i],-0.5), ls[i], 1, fc='#abd4ab'))
        axs[0].set_xlim(long_limits)
        axs[0].set_ylim([-1, 1])

        # shift the layout box down
        box = axs[0].get_position()
        vshift = 0.025
        box.y0 = box.y0 - vshift
        box.y1 = box.y1 - vshift
        axs[0].set_position(box)
        
        # plot beta function
        axs[1].plot(ss_beta, self.beta0*np.ones_like(ss_beta), ':', color=col0)
        axs[1].plot(ss_beta, np.sqrt(betas), color=colx1)
        axs[1].set_ylabel(r'$\sqrt{\mathrm{Beta\hspace{0.3}function}}$ (m$^{0.5}$)')
        axs[1].set_xlim(long_limits)
        
        # plot dispersion
        axs[2].plot(ss_disp1, np.zeros_like(ss_disp1), ':', color=col0)
        axs[2].plot(ss_disp2, second_order_dispersion / 1e-3, '-', color=colx2, label='2nd order')
        axs[2].plot(ss_disp1, dispersion / 1e-3, '-', color=colx1, label='1st order')
        axs[2].set_ylabel('Horizontal dispersion (mm)')
        axs[2].set_xlim(long_limits)
        axs[2].legend(loc='best', reverse=True, fontsize='small')
        
        # plot R56
        axs[3].plot(ss_R56, np.zeros_like(ss_R56), ':', color=col0)
        axs[3].plot(ss_R56, R56/1e-3, color=colz)
        axs[3].set_ylabel(r'Longitudinal dispersion, $R_{56}$ (mm)')
        axs[3].set_xlim(long_limits)
        axs[3].set_xlabel(long_label)

         # save figure to file
        if savefig is not None:
            fig.savefig(str(savefig), format="pdf", bbox_inches="tight")


    def plot_chromaticity(self, savefig=None):

        from matplotlib import pyplot as plt
        from matplotlib import patches
        from abel.utilities.beam_physics import evolve_chromatic_amplitude

        # calculate evolution
        ls, inv_rhos, ks, ms, taus = self.matrix_lattice()
        ssl = np.append([0.0], np.cumsum(ls))
        _, evol_W = evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, self.beta0, fast=False, plot=False);
        ss = evol_W[0]
        Ws = evol_W[1]
        
        # prepare plots
        fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [0.1, 1]})
        fig.set_figwidth(7)
        fig.set_figheight(5)
        col0 = "tab:gray"
        colx1 = "tab:blue"
        colx2 = "#d7e9f5" # lighter version of tab:blue
        colz = "tab:green"
        long_label = 'Location [m]'
        long_limits = [min(ss), max(ss)]

        # layout
        axs[0].plot(ss, np.zeros_like(ss), '-', linewidth=0.5, color='k')
        axs[0].axis('off')
        for i in range(len(ls)):
            if abs(inv_rhos[i]) > 0: # add dipoles
                axs[0].add_patch(patches.Rectangle((ssl[i],-0.75), ls[i], 1.5, fc='#d9d9d9'))
            if abs(ks[i]) > 0: # add plasma lenses
                axs[0].add_patch(patches.Rectangle((ssl[i],0), ls[i], 1, fc='#fcb577'))
            if abs(ms[i]) > 0: # add sextupole
                axs[0].add_patch(patches.Rectangle((ssl[i],-0.5), ls[i], 1, fc='#abd4ab'))
        axs[0].set_xlim(long_limits)
        axs[0].set_ylim([-1, 1])

        # shift the layout box down
        box = axs[0].get_position()
        vshift = 0.025
        box.y0 = box.y0 - vshift
        box.y1 = box.y1 - vshift
        axs[0].set_position(box)
        
        # plot beta function
        axs[1].plot(ss, Ws, color=colx1)
        axs[1].set_ylabel('W_x')
        axs[1].set_xlim(long_limits)
        
         # save figure to file
        if savefig is not None:
            fig.savefig(str(savefig), format="pdf", bbox_inches="tight")

    
    def plot_layout(self, delta=0.1, axes_equal=False, use_second_order_dispersion=True, savefig=None):
        "Plot the layout with beam orbit and dispersion."
        
        from matplotlib import pyplot as plt
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_second_order_dispersion, evolve_orbit
        from scipy.integrate import cumulative_trapezoid

        # get the lengths and strengths
        ls, inv_rhos, ks, ms, taus = self.matrix_lattice()
        
        # prepare plots
        fig, ax = plt.subplots(1)
        fig.set_figwidth(12)
        fig.set_figheight(2)

        # get the orbit
        theta, _ = evolve_orbit(ls, inv_rhos)
        _, evol_orbit = evolve_orbit(ls, inv_rhos, theta0=-theta/2)
        xs = evol_orbit[0,:]
        ys = evol_orbit[1,:]
        ss = evol_orbit[2,:]
        thetas = evol_orbit[3,:]
        
        # define the width of the dipole
        width_dipole = 0.1*max(ss)/6

        # calculate dispersions
        _, _, evol_disp2 = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=False)
        delta = abs(delta)
        Dx = np.interp(ss, evol_disp2[0], evol_disp2[1])
        DDx = np.interp(ss, evol_disp2[0], evol_disp2[2])
        if use_second_order_dispersion:
            offset_disp = Dx*delta - DDx*delta**2
        else:
            offset_disp = Dx*delta
        xs_low = xs - offset_disp*np.sin(thetas)
        ys_low = ys - offset_disp*np.cos(thetas)
        xs_high = xs + offset_disp*np.sin(thetas)
        ys_high = ys + offset_disp*np.cos(thetas)

        # calculate beam size
        _, _, evol_beta = evolve_beta_function(ls, ks, self.beta0, fast=False)
        betas = np.interp(ss, evol_beta[0], evol_beta[1])
        if abs(delta) > 0:
            emit_gx = offset_disp[np.argmax(betas)]**2/max(betas)
        else:
            emit_gx = width_dipole**2/max(betas)
        sigxs = np.sqrt(betas*emit_gx)/2
        phis = np.append([0], cumulative_trapezoid(ss, 1/betas))
        
        # plot the orbits
        alpha = 0.15
        lw_dotted = 0.8
        lw_solid = 1.2
        col_low = '#E42F2C'
        if abs(delta) > 0:
            col_mid = '#67b607'
        else:
            col_mid = '#0182EC'
        col_high = '#0182EC'

        if abs(delta) > 0:
            
            # add lower energy
            if not axes_equal:
                ax.fill(np.concatenate([xs_low - sigxs*np.sin(thetas), np.flip(xs_low+sigxs*np.sin(thetas))]),
                    np.concatenate([ys_low - sigxs*np.cos(thetas), np.flip(ys_low+sigxs*np.cos(thetas))]), col_low, alpha=alpha)
                ax.plot(xs_low - sigxs*np.sin(thetas)*np.sin(phis), ys_low - sigxs*np.cos(thetas)*np.sin(phis), ':', lw=lw_dotted, c=col_low)
            ax.plot(xs_low, ys_low, lw=lw_solid, c=col_low)

        # add high energy
        if not axes_equal or abs(delta)==0:
            ax.fill(np.concatenate([xs - sigxs*np.sin(thetas), np.flip(xs+sigxs*np.sin(thetas))]),
                np.concatenate([ys - sigxs*np.cos(thetas), np.flip(ys+sigxs*np.cos(thetas))]), col_mid, alpha=alpha)
        if not axes_equal:
            ax.plot(xs - sigxs*np.sin(thetas)*np.sin(phis), ys - sigxs*np.cos(thetas)*np.sin(phis), ':', lw=lw_dotted, c=col_mid)
        ax.plot(xs, ys, lw=lw_solid, c=col_mid)

        if abs(delta) > 0:
            # add average energy
            if not axes_equal:
                ax.fill(np.concatenate([xs_high - sigxs*np.sin(thetas), np.flip(xs_high+sigxs*np.sin(thetas))]),
                    np.concatenate([ys_high - sigxs*np.cos(thetas), np.flip(ys_high+sigxs*np.cos(thetas))]), col_high, alpha=alpha)
                ax.plot(xs_high - sigxs*np.sin(thetas)*np.sin(phis), ys_high - sigxs*np.cos(thetas)*np.sin(phis), ':', lw=lw_dotted, c=col_high)
            ax.plot(xs_high, ys_high, lw=lw_solid, c=col_high)
            
        
        # add elements
        lw_element=0.75
        if not axes_equal:
            width_dipole = max(abs(offset_disp))*2
        width_lens = width_dipole*0.8
        width_sextupole = width_dipole*0.6
        ssl = np.append([0.0], np.cumsum(ls))
        ssl = ssl[:-1]
        for i in range(len(ls)-1):
            inds = np.logical_and(ss <= ssl[i+1], ss >= ssl[i])
            # add dipole
            if abs(inv_rhos[i]) > 0:
                xs_left = xs[inds] - width_dipole*np.sin(thetas[inds])
                ys_left = ys[inds] - width_dipole*np.cos(thetas[inds])
                xs_right = xs[inds] + width_dipole*np.sin(thetas[inds])
                ys_right = ys[inds] + width_dipole*np.cos(thetas[inds])
                ax.fill(np.concatenate([xs_left, np.flip(xs_right)]), np.concatenate([ys_left, np.flip(ys_right)]), '#f2f2f2', edgecolor='#d9d9d9', lw=lw_element, zorder=0)
            # add plasma lens
            if abs(ks[i]) > 0:
                xs_left = xs[inds] - width_lens*np.sin(thetas[inds])
                ys_left = ys[inds] - width_lens*np.cos(thetas[inds])
                xs_right = xs[inds] + width_lens*np.sin(thetas[inds])
                ys_right = ys[inds] + width_lens*np.cos(thetas[inds])
                ax.fill(np.concatenate([xs_left, np.flip(xs_right)]), np.concatenate([ys_left, np.flip(ys_right)]), '#fad1ac', edgecolor='#fcb577', lw=lw_element, zorder=0)
            # add sextupole
            if abs(ms[i]) > 0:
                xs_left = xs[inds] - width_sextupole*np.sin(thetas[inds])
                ys_left = ys[inds] - width_sextupole*np.cos(thetas[inds])
                xs_right = xs[inds] + width_sextupole*np.sin(thetas[inds])
                ys_right = ys[inds] + width_sextupole*np.cos(thetas[inds])
                ax.fill(np.concatenate([xs_left, np.flip(xs_right)]), np.concatenate([ys_left, np.flip(ys_right)]), '#cfe3cf', edgecolor='#abd4ab', lw=lw_element, zorder=0)

        # add labels
        ax.set_xlabel('z (m)')
        ax.set_ylabel('x (m)')

        # set axis limits
        if axes_equal:
            ax.axis('equal')
        else:
            ax.set_ylim(min(ys)-width_dipole*2, max(ys)+width_dipole*2)
        ax.set_xlim(min(xs)-(max(xs)-min(xs))*0.02, max(xs)+(max(xs)-min(xs))*0.02)

        # invert axis to have positive values on the "right"
        ax.yaxis.set_inverted(True)

        # save figure to file
        if savefig is not None:
            fig.savefig(str(savefig), format="pdf", bbox_inches="tight")

    
    ## PRINT INFO

    def print_summary(self):
        print('------------------------------------------------')
        print(f'Main dipole (2x):          {self.length_dipole:.3f} m,  B = {self.field_dipole:.2f} T')
        print(f'Plasma lens (2x):          {self.length_plasma_lens:.3f} m,  g = {self.field_gradient_plasma_lens:.1f} T/m')
        print(f'Outer chicane dipole (2x): {self.length_chicane_dipole:.3f} m,  B = {self.field_chicane_dipole1:.3f} T')
        print(f'Inner chicane dipole (2x): {self.length_chicane_dipole:.3f} m,  B = {self.field_chicane_dipole2:.3f} T')
        if self.cancel_sec_order_dispersion:
            print(f'Sextupole:                 {self.length_central_gap_or_sextupole:.3f} m,  m = {self.field_gradient_sextupole:.1f} T/m^2')
            print(f'Gaps (10x):                {self.length_gap:.3f} m')
        else:
            print(f'Central gap:               {self.length_central_gap_or_sextupole + 2*self.length_gap:.3f} m')
            print(f'Other gaps (8x):           {self.length_gap:.3f} m')
        
        print('------------------------------------------------')
        print(f'             Total length: {self.get_length():.3f} m')
        print(f'         Total bend angle:           {np.rad2deg(self.total_bend_angle()):.2f} deg')
        print('------------------------------------------------')

    
    
    ## SURVEY PLOTS
    
    def total_bend_angle(self):
        from abel.utilities.relativity import energy2momentum
        BL = 2*(self.length_dipole*self.field_dipole + self.length_chicane_dipole*(self.field_chicane_dipole1 + self.field_chicane_dipole2))
        theta = self.charge_sign*BL*SI.e/energy2momentum(self.nom_energy)
        return theta

    def survey_object(self):
        npoints = 10
        x_points = np.linspace(0, self.get_length(), npoints)
        y_points = np.linspace(0, 0, npoints)
        final_angle = 0 
        label = 'Interstage'
        color = 'orange'
        return x_points, y_points, final_angle, label, color

        
    ## COST MODEL
    
    def get_cost_breakdown(self):
        return ('Interstage', self.get_length() * CostModeled.cost_per_length_interstage)

    