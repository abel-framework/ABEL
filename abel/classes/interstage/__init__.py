from abc import abstractmethod
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled
from types import SimpleNamespace
import numpy as np
import scipy.constants as SI

class Interstage(Trackable, CostModeled):
    
    @abstractmethod
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0, lens_radius=2e-3,
                 use_nonlinearity=True, use_chicane=True, use_sextupole=True, use_gaps=True, use_thick_lenses=True, use_apertures=True,
                 enable_csr=True, enable_isr=True, enable_space_charge=False):
        
        super().__init__()

        # main parameters
        self.nom_energy = nom_energy
        self._R56 = R56
        self.lens_radius = lens_radius

        # main parameters (functions of energy)
        self._beta0 = beta0
        self._length_dipole = length_dipole
        self._field_dipole = field_dipole

        # length ratios
        self.length_ratio_gap = 0.02
        self.length_ratio_plasma_lens = 0.05
        self.length_ratio_chicane_dipole = 0.85
        self.length_ratio_sextupole = 0.3

        # derivable (but also settable) parameters
        self._field_ratio_chicane_dipole1 = None
        self._field_ratio_chicane_dipole2 = None
        self._strength_plasma_lens = None # [1/m]
        self._nonlinearity_plasma_lens = None # [1/m]
        self._strength_sextupole = None # [1/m^2]
        
        # feature toggles
        self.use_nonlinearity = use_nonlinearity
        self.use_chicane = use_chicane
        self.use_sextupole = use_sextupole
        self.use_gaps = use_gaps
        self.use_thick_lenses = use_thick_lenses
        self.use_apertures = use_apertures

        # physics flags
        self.enable_csr = enable_csr
        self.enable_isr = enable_isr
        self.enable_space_charge = enable_space_charge
        
        # evolution (saved when tracking)
        self.evolution = SimpleNamespace()

    
    ## TRACKING
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)


    ## OVERALL LENGTH
    
    # lattice length
    def get_length(self):
        if self.length_dipole is not None:
            return 2*self.length_dipole + 10*self.length_gap + 2*self.length_plasma_lens + 4*self.length_chicane_dipole + self.length_sextupole
        else:
            return None

    
    ## ENERGY-FUNCTION PARAMETERS
    
    # evaluate R56 (if it is a function)
    @property
    def R56(self) -> float:
        if callable(self._R56):
            return self._R56(self.nom_energy)
        else:
            return self._R56
    @R56.setter
    def R56(self, val):
        self._R56 = val
        
    # evaluate initial beta function (if it is a function)
    @property
    def beta0(self) -> float:
        if callable(self._beta0):
            return self._beta0(self.nom_energy)
        else:
            return self._beta0
    @beta0.setter
    def beta0(self, val):
        self._beta0 = val

    # evaluate dipole field (if it is a function)
    @property
    def field_dipole(self) -> float:
        if callable(self._field_dipole):
            return self._field_dipole(self.nom_energy)
        else:
            return self._field_dipole
    @field_dipole.setter
    def field_dipole(self, val):
        self._field_dipole = val

    # evaluate dipole length (if it is a function)
    @property
    def length_dipole(self) -> float:
        if callable(self._length_dipole):
            return self._length_dipole(self.nom_energy)
        else:
            return self._length_dipole
    @length_dipole.setter
    def length_dipole(self, val):
        self._length_dipole = val
    
    
    ## RATIO-DEFINED PARAMETERS
    
    @property
    def length_gap(self):
        return float(self.use_gaps) * self.length_dipole * self.length_ratio_gap
        
    @property
    def length_plasma_lens(self):
        if self.use_thick_lenses:
            return self.length_dipole * self.length_ratio_plasma_lens
        else:
            return self.length_dipole * 1e-6

    @property
    def length_chicane_dipole(self) -> float:
        return self.length_dipole * self.length_ratio_chicane_dipole

    @property
    def length_sextupole(self):
        if self.use_thick_lenses:
            return float(self.use_sextupole) * self.length_dipole * self.length_ratio_sextupole
        else:
            return float(self.use_sextupole) * self.length_dipole * 1e-6

    
    ## DERIVED PARAMETERS

    @property
    def field_chicane_dipole1(self) -> float:
        if not self.use_chicane:
            return 0.0
        else:
            if self._field_ratio_chicane_dipole1 is None:
                self.match_dispersion_and_R56()
            return self.field_dipole * self._field_ratio_chicane_dipole1

    @property
    def field_chicane_dipole2(self) -> float:
        if not self.use_chicane:
            return 0.0
        else:
            if self._field_ratio_chicane_dipole2 is None:
                self.match_dispersion_and_R56()
            return self.field_dipole * self._field_ratio_chicane_dipole2

        
    @property
    def strength_plasma_lens(self) -> float:
        if self._strength_plasma_lens is None:
            self.match_beta_function()
        return self._strength_plasma_lens

    @property
    def field_gradient_plasma_lens(self) -> float:
        "Plasma-lens field gradient [T/m]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_plasma_lens*p0/(SI.e*self.length_plasma_lens)

    @property
    def nonlinearity_plasma_lens(self) -> float:
        if self._nonlinearity_plasma_lens is None:
            
            # calculate dispersion and dispersion prime in the lens
            p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
            Dpx_lens = self.length_dipole*self.field_dipole*SI.e/p0
            Dx_lens = Dpx_lens*(self.length_dipole/2 + self.length_gap + self.length_plasma_lens/2)

            # calculate exact nonlinearity (inverse of the dispersion)
            tau_lens = float(self.use_nonlinearity)/Dx_lens
            
            return tau_lens
        else:
            return self._nonlinearity_plasma_lens
   
    @property
    def strength_sextupole(self) -> float:
        if not self.use_sextupole:
            return 0.0
        else:
            if self._strength_sextupole is None:
                self.match_second_order_dispersion()
            return self._strength_sextupole

    @property
    def field_gradient_sextupole(self) -> float:
        "Sextupole field gradient [T/m^2]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_sextupole*p0/(SI.e*self.length_sextupole)

    
    ## MATCHING
    
    def match_all(self):
        "Combined matching the beta function, first- and second-order dispersion and the R56"
        self.match_beta_function()
        self.match_dispersion_and_R56()
        self.match_second_order_dispersion()
    
    def match_beta_function(self):
        "Matching the beta function by adjusting the plasma-lens strength."
        
        # approzimate the focal length
        L1eff = self.length_dipole + 2*self.length_gap + self.length_plasma_lens/2
        L2eff = 2*self.length_chicane_dipole + self.length_plasma_lens/2 + 3*self.length_gap + self.length_sextupole/2
        f0 = 1/(1/L1eff + 1/L2eff)
        kl_lens0 = 1/f0
        k_lens0 = kl_lens0/self.length_plasma_lens
        
        # minimizer function for beta matching (central alpha function is zero)
        from abel.utilities.beam_physics import evolve_beta_function
        def minfun_beta(params):
            ls, _, ks, _, _ = self.matrix_lattice(k_lens=params[0], tau_lens=0, B_chic1=0, B_chic2=0, m_sext=0, half_lattice=True)
            _, alpha, _ = evolve_beta_function(ls, ks, self.beta0, fast=True) 
            return alpha**2
    
        # match the beta function
        from scipy.optimize import minimize
        result_beta = minimize(minfun_beta, k_lens0, tol=1e-10, options={'maxiter': 50})
        self._strength_plasma_lens = result_beta.x[0]*self.length_plasma_lens
    
    def match_dispersion_and_R56(self):
        "Cancelling the dispersion and matchign the R56 by adjusting the chicane dipoles."
        
        # assume negative R56
        nom_R56 = -abs(self.R56)
        if self.R56 > 0:
            print('Positive R56 given, flipping sign to negative')
        
        # get plasma-lens strength
        k_lens = self.strength_plasma_lens/self.length_plasma_lens
        
        # minimizer function for dispersion (central dispersion prime is zero)
        from abel.utilities.beam_physics import evolve_dispersion, evolve_R56
        def minfun_dispersion_R56(params):
            ls, inv_rhos, ks, _, _ = self.matrix_lattice(k_lens=k_lens, tau_lens=0, B_chic1=params[0], B_chic2=params[1], m_sext=0, half_lattice=True)
            _, Dpx_mid, _ = evolve_dispersion(ls, inv_rhos, ks, fast=True) 
            R56_mid, _ = evolve_R56(ls, inv_rhos, ks) 
            return Dpx_mid**2 + (R56_mid - nom_R56/2)**2
    
        # match the beta function
        from scipy.optimize import minimize
        result_dispersion_R56 = minimize(minfun_dispersion_R56, [self.field_dipole/2, -self.field_dipole/2], tol=1e-20, options={'maxiter': 50})
        self._field_ratio_chicane_dipole1 = result_dispersion_R56.x[0]/self.field_dipole
        self._field_ratio_chicane_dipole2 = result_dispersion_R56.x[1]/self.field_dipole
    
    def match_second_order_dispersion(self):
        "Cancelling the second-order dispersion by adjusting the sextupole strength."
    
        # calculate dispersion and dispersion prime in the lens
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        Dpx_lens = self.length_dipole*self.field_dipole*SI.e/p0
        Dx_lens = Dpx_lens*(self.length_dipole/2 + self.length_gap + self.length_plasma_lens/2)

        # get lens parameters
        k_lens = self.strength_plasma_lens/self.length_plasma_lens
        tau_lens = float(self.use_nonlinearity)/Dx_lens
        
        # estimate the central dispersion and second order dispersion prime
        L1 = self.length_dipole
        L2 = 2*self.length_chicane_dipole
        B1 = self.field_dipole
        B2 = self.field_chicane_dipole1
        Dx_mid = Dx_lens*(1/4)*(1 + 3*L2/L1 + 2*(B2*L2**2)/(B1*L1**2))
        DDxp_mid = (2*Dpx_lens)*((3/4)*(L1/L2 + 1) - 1)*(2.0-int(self.use_nonlinearity))

        # guesstimate the sextupole strength (starting point for optimization)
        ml_sext0 = int(self.use_sextupole)*2*DDxp_mid/Dx_mid**2
        m_sext0 = ml_sext0/self.length_sextupole

        # get the second dipole
        B_chic2 = self.field_chicane_dipole2
        
        # minimizer function for second-order dispersion (central second-order dispersion prime is zero)
        from abel.utilities.beam_physics import evolve_second_order_dispersion
        def minfun_second_order_dispersion(params):
            ls, inv_rhos, ks, ms, taus = self.matrix_lattice(k_lens=k_lens, tau_lens=tau_lens, B_chic2=B_chic2, m_sext=params[0], half_lattice=True)
            _, DDpx, _ = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=True) 
            return DDpx**2
    
        # match the beta function
        from scipy.optimize import minimize
        result_dispersion = minimize(minfun_second_order_dispersion, m_sext0, method='Nelder-Mead', tol=1e-20, options={'maxiter': 50})
        self._strength_sextupole = result_dispersion.x[0]*self.length_sextupole
    
        
    
    ## MATRIX LATTICE

    # full lattice 
    def matrix_lattice(self, k_lens=None, tau_lens=None, B_chic1=None, B_chic2=None, m_sext=None, half_lattice=False):

        # nominal momentum
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c;
        
        # element length array
        dL = self.length_gap
        ls = np.array([dL, self.length_dipole, dL, self.length_plasma_lens, dL, 
                       self.length_chicane_dipole, dL, self.length_chicane_dipole, dL, self.length_sextupole/2])
        
        # bending strength array
        if B_chic1 is None:
            B_chic1 = self.field_chicane_dipole1
        if B_chic2 is None:
            B_chic2 = self.field_chicane_dipole2
        Bs = np.array([0, self.field_dipole, 0, 0, 0, B_chic1, 0, B_chic2, 0, 0])
        inv_rhos = Bs * SI.e / p0
        
        # focusing strength array
        if k_lens is None:
            k_lens = self.strength_plasma_lens/self.length_plasma_lens
        ks = np.array([0, 0, 0, k_lens, 0, 0, 0, 0, 0, 0])
        
        # sextupole strength array
        if m_sext is None:
            m_sext = self.strength_sextupole/self.length_sextupole
        ms = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, m_sext])

        # plasma-lens transverse taper array
        if tau_lens is None:
            tau_lens = self.nonlinearity_plasma_lens
        taus = np.array([0, 0, 0, tau_lens, 0, 0, 0, 0, 0, 0])
        
        # mirror symmetrize the lattice
        if not half_lattice:
            ls = np.append(ls, np.flip(ls))
            inv_rhos = np.append(inv_rhos, np.flip(inv_rhos))
            ks = np.append(ks, np.flip(ks))
            ms = np.append(ms, np.flip(ms))
            taus = np.append(taus, np.flip(taus))
        
        return ls, inv_rhos, ks, ms, taus
        
    
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
        axs[2,0].plot(evol.location, evol.y*1e6, color=col1)
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

    def plot_optics(self):

        from matplotlib import pyplot as plt
        from matplotlib import patches
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_second_order_dispersion, evolve_R56

        # calculate evolution
        ls, inv_rhos, ks, ms, taus = self.matrix_lattice()
        _, _, evol_beta = evolve_beta_function(ls, ks, self.beta0, fast=False)
        _, _, evol_dispersion = evolve_dispersion(ls, inv_rhos, ks, fast=False);
        _, _, evol_second_order_dispersion = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=False);
        _, evol_R56 = evolve_R56(ls, inv_rhos, ks);
        ssl = np.append([0.0], np.cumsum(ls))
        
        # extract into readable format
        ss_beta = evol_beta[0]
        betas = evol_beta[1]
        ss_disp1 = evol_dispersion[0]
        dispersion = evol_dispersion[1]
        ss_disp2 = evol_second_order_dispersion[0]
        second_order_dispersion = evol_second_order_dispersion[2]
        ss_R56 = evol_R56[0]
        R56 = evol_R56[1]
        
        # prepare plots
        fig, axs = plt.subplots(4,1, gridspec_kw={'height_ratios': [0.1, 1, 1, 1]})
        fig.set_figwidth(7)
        fig.set_figheight(11)
        col0 = "tab:gray"
        colx1 = "tab:blue"
        colx2 = "#c2e6ff" # lighter version of tab:blue
        colz = "tab:green"
        long_label = 'Location [m]'
        long_limits = [min(ss_beta), max(ss_beta)]

        # layout
        axs[0].plot(ss_beta, self.beta0*np.ones_like(ss_beta), '-', linewidth=0.5, color='k')
        axs[0].axis('off')
        for i in range(len(ls)):
            if abs(inv_rhos[i]) > 0: # add dipoles
                axs[0].add_patch(patches.Rectangle((ssl[i],-0.75), ls[i], 1.5, fc='lightgray'))
            if abs(ks[i]) > 0: # add plasma lenses
                axs[0].add_patch(patches.Rectangle((ssl[i],0), ls[i], 1, fc='black'))
            if abs(ms[i]) > 0: # add sextupole
                axs[0].add_patch(patches.Rectangle((ssl[i],-0.5), ls[i], 1, fc='#7dc451'))
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
        axs[1].plot(ss_beta, betas, color=colx1)
        axs[1].set_ylabel('Beta function [m]')
        axs[1].set_xlim(long_limits)
        axs[1].set_yscale('log')
        
        # plot dispersion
        axs[2].plot(ss_disp1, np.zeros_like(ss_disp1), ':', color=col0)
        axs[2].plot(ss_disp2, second_order_dispersion / 1e-3, color=colx2, label='2nd order')
        axs[2].plot(ss_disp1, dispersion / 1e-3, '-', color=colx1, label='1st order')
        axs[2].set_ylabel('Dispersion')
        axs[2].set_xlim(long_limits)
        axs[2].legend(loc='upper right', reverse=True, fontsize='small')
        
        # plot R56
        axs[3].plot(ss_R56, np.zeros_like(ss_R56), ':', color=col0)
        axs[3].plot(ss_R56, R56/1e-3, color=colz)
        axs[3].set_ylabel('Longitudinal dispersion, R56 (mm)')
        axs[3].set_xlim(long_limits)
        axs[3].set_xlabel(long_label)
        
        
    
    ## SURVEY PLOTS
    
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

    