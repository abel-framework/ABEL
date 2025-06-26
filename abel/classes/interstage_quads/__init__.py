from abc import abstractmethod
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled
from types import SimpleNamespace
import numpy as np
import scipy.constants as SI

class InterstageQuads(Trackable, CostModeled):
    
    @abstractmethod
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0, polarity_quads=1, beta_ratio_central = 2, 
                 use_apertures=True, use_chromaticity_correction=True, use_central_sextupole=True,
                 enable_csr=True, enable_isr=True, enable_space_charge=False, charge_sign=-1):
        
        super().__init__()

        # main parameters
        self.nom_energy = nom_energy
        self._R56 = R56
        self.polarity_quads = polarity_quads
        self.beta_ratio_central = beta_ratio_central
        self.charge_sign = charge_sign
        
        # main parameters (functions of energy)
        self._beta0 = beta0
        self._length_dipole = length_dipole
        self._field_dipole = field_dipole

        # length ratios
        self.length_ratio_gap = 0.025
        self.length_ratio_quad_gap_or_sextupole = 0.35
        self.length_ratio_central_gap_or_sextupole = 0.35
        self.length_ratio_quadrupole = 0.50
        self.length_ratio_chicane_dipole = 1.2

        # quad alignment and jitter
        self.jitter = SimpleNamespace()
        self.jitter.quad_offset_x = 0
        self.jitter.quad_offset_y = 0
        
        # derivable (but also settable) parameters
        self._field_ratio_chicane_dipole1 = None
        self._field_ratio_chicane_dipole2 = None
        self._strength_quadrupole1 = None # [1/m]
        self._strength_quadrupole2 = None # [1/m]
        self._strength_quadrupole3 = None # [1/m]
    
        self._strength_sextupole1 = None # [1/m]
        self._strength_sextupole2 = None # [1/m]
        self._strength_sextupole3 = None # [1/m]
        
        # feature toggles
        self.use_chromaticity_correction = use_chromaticity_correction
        self.use_central_sextupole = use_central_sextupole
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
            ls, *_ = self.matrix_lattice(k1=0, k2=0, k3=0, B_chic1=0, B_chic2=0, m1=0, m2=0, m3=0, half_lattice=False)
            return np.sum(ls)
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
        return self.length_dipole * self.length_ratio_gap

    @property
    def length_quad_gap_or_sextupole(self):
        return self.length_dipole * self.length_ratio_quad_gap_or_sextupole

    @property
    def length_central_gap_or_sextupole(self):
        return self.length_dipole * self.length_ratio_central_gap_or_sextupole
        
    @property
    def length_quadrupole(self):
        return self.length_dipole * self.length_ratio_quadrupole

    @property
    def length_chicane_dipole(self) -> float:
        return self.length_dipole * self.length_ratio_chicane_dipole

    ## DERIVED PARAMETERS

    @property
    def field_chicane_dipole1(self) -> float:
        if self._field_ratio_chicane_dipole1 is None:
            self.match_lattice()
        return self.field_dipole * self._field_ratio_chicane_dipole1

    @property
    def field_chicane_dipole2(self) -> float:
        if self._field_ratio_chicane_dipole2 is None:
            self.match_lattice()
        return self.field_dipole * self._field_ratio_chicane_dipole2

        
    @property
    def strength_quadrupole1(self) -> float:
        if self._strength_quadrupole1 is None:
            self.match_beta_function()
        return self._strength_quadrupole1

    @property
    def field_gradient_quadrupole1(self) -> float:
        "First quadrupole field gradient [T/m]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_quadrupole1*p0/(SI.e*self.length_quadrupole)
   
    @property
    def strength_quadrupole2(self) -> float:
        if self._strength_quadrupole2 is None:
            self.match_beta_function()
        return self._strength_quadrupole2

    @property
    def field_gradient_quadrupole2(self) -> float:
        "Second quadrupole field gradient [T/m]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_quadrupole2*p0/(SI.e*self.length_quadrupole)

    @property
    def strength_quadrupole3(self) -> float:
        if self._strength_quadrupole3 is None:
            self.match_beta_function()
        return self._strength_quadrupole3

    @property
    def field_gradient_quadrupole3(self) -> float:
        "Third quadrupole field gradient [T/m]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_quadrupole3*p0/(SI.e*self.length_quadrupole)


    @property
    def strength_sextupole1(self) -> float:
        if self._strength_sextupole1 is None:
            self.cancel_chromaticity()
        return self._strength_sextupole1

    @property
    def strength_sextupole2(self) -> float:
        if self._strength_sextupole2 is None:
            self.cancel_chromaticity()
        return self._strength_sextupole2
        
    @property
    def strength_sextupole3(self) -> float:
        if self._strength_sextupole3 is None:
            self.cancel_second_order_dispersion()
        return self._strength_sextupole3
    
    @property
    def field_gradient_sextupole1(self) -> float:
        "Sextupole 1 field gradient [T/m^2]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_sextupole1*p0/(SI.e*self.length_quad_gap_or_sextupole)

    @property
    def field_gradient_sextupole2(self) -> float:
        "Sextupole 2 field gradient [T/m^2]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_sextupole2*p0/(SI.e*self.length_quad_gap_or_sextupole)

    @property
    def field_gradient_sextupole3(self) -> float:
        "Central sextupole field gradient [T/m^2]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_sextupole3*p0/(SI.e*self.length_central_gap_or_sextupole)

    
    ## MATRIX LATTICE

    # full lattice 
    def matrix_lattice(self, k1=None, k2=None, k3=None, B_chic1=None, B_chic2=None, m1=None, m2=None, m3=None, half_lattice=False):

        # nominal momentum
        from abel.utilities.relativity import energy2momentum
        
        # element length array
        dL = self.length_gap
        ls = np.array([dL, self.length_dipole, dL, self.length_quadrupole, dL, self.length_quadrupole, dL, self.length_quad_gap_or_sextupole, dL, self.length_quadrupole, dL, self.length_quad_gap_or_sextupole, dL, self.length_chicane_dipole, dL, self.length_chicane_dipole, dL, self.length_central_gap_or_sextupole/2])
        
        # bending strength array
        if B_chic1 is None:
            B_chic1 = self.field_chicane_dipole1
        if B_chic2 is None:
            B_chic2 = self.field_chicane_dipole2
        Bs = np.array([0, self.field_dipole, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, B_chic1, 0, B_chic2, 0, 0])
        inv_rhos = -self.charge_sign*Bs * SI.e / energy2momentum(self.nom_energy)
        
        # focusing strength array
        if k1 is None:
            k1 = self.strength_quadrupole1/self.length_quadrupole
        if k2 is None:
            k2 = self.strength_quadrupole2/self.length_quadrupole
        if k3 is None:
            k3 = self.strength_quadrupole3/self.length_quadrupole
        ks = np.array([0, 0, 0, k1, 0, k2, 0, 0, 0, k3, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # sextupole strength array
        if m1 is None:
            m1 = self.strength_sextupole1/self.length_quad_gap_or_sextupole
        if m2 is None:
            m2 = self.strength_sextupole2/self.length_quad_gap_or_sextupole
        if m3 is None:
            m3 = self.strength_sextupole3/self.length_central_gap_or_sextupole
        ms = np.array([0, 0, 0, 0, 0, 0, 0, m1, 0, 0, 0, m2, 0, 0, 0, 0, 0, m3])

        # plasma-lens transverse taper array
        taus = np.zeros_like(ls)
        
        # mirror symmetrize the lattice
        if not half_lattice:
            ls = np.append(np.append(ls[:-1], 2*ls[-1]), np.flip(ls[:-1]))
            inv_rhos = np.append(np.append(inv_rhos[:-1], inv_rhos[-1]), np.flip(inv_rhos[:-1]))
            ks = np.append(np.append(ks[:-1], ks[-1]), np.flip(ks[:-1]))
            ms = np.append(np.append(ms[:-1], ms[-1]), np.flip(ms[:-1]))
            taus = np.append(np.append(taus[:-1], taus[-1]), np.flip(taus[:-1]))
        
        return ls, inv_rhos, ks, ms, taus

        
    ## MATCHING
    
    def match_lattice(self):
        "Combined matching the beta function, first-order dispersion and the R56"
        self.match_beta_function()
        self.cancel_dispersion_and_match_R56(high_res=False)
        if self.use_chromaticity_correction:
            self.cancel_chromaticity()
        if self.use_central_sextupole:
            self.cancel_second_order_dispersion()

    
    def match_beta_function(self):
        "Matching the beta function by adjusting the quadrupole strengths."
        
        # minimizer function for beta matching (central alpha function is zero)
        from abel.utilities.beam_physics import evolve_beta_function
        def minfun_beta(params):
            ls, _, ks, _, _ = self.matrix_lattice(k1=params[0], k2=params[1], k3=params[2], B_chic1=0, B_chic2=0, m1=0, m2=0, m3=0, half_lattice=True)
            beta_x, alpha_x, _ = evolve_beta_function(ls, ks, self.beta0, fast=True) 
            beta_y, alpha_y, _ = evolve_beta_function(ls, -ks, self.beta0, fast=True)
            return alpha_x**2 + alpha_y**2 + (beta_x/beta_y-float(self.beta_ratio_central)**np.sign(self.polarity_quads))**2+ (max(beta_x/10, self.beta0)/self.beta0-1)**2 + (max(beta_y/10, self.beta0)/self.beta0-1)**2

        # initial guess for the quad strength
        k1_guess = self.polarity_quads*2/(self.length_dipole*self.length_quadrupole)
        k2_guess = -self.polarity_quads*2/(self.length_dipole*self.length_quadrupole)
        k3_guess = k1_guess
        
        # match the beta function
        from scipy.optimize import minimize
        result_beta = minimize(minfun_beta, [k1_guess, k2_guess, k3_guess], tol=1e-20, options={'maxiter': 200})
        self._strength_quadrupole1 = result_beta.x[0]*self.length_quadrupole
        self._strength_quadrupole2 = result_beta.x[1]*self.length_quadrupole
        self._strength_quadrupole3 = result_beta.x[2]*self.length_quadrupole

    
    def cancel_dispersion_and_match_R56(self, high_res=False):
        "Cancelling the dispersion and matching the R56 by adjusting the chicane dipoles."
        
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
            ls, inv_rhos, ks, _, _ = self.matrix_lattice(B_chic1=params[0], B_chic2=params[1], m1=0, m2=0, m3=0, half_lattice=True)
            _, Dpx_mid, _ = evolve_dispersion(ls, inv_rhos, ks, fast=True)
            R56_mid, _ = evolve_R56(ls, inv_rhos, ks, high_res=high_res)
            return (Dpx_mid/Dpx_scale)**2 + ((R56_mid - nom_R56/2)/R56_scale)**2

        # initial guess for the chicane dipole fields
        if self._field_ratio_chicane_dipole1 is not None:
            B_chic1_guess = self.field_chicane_dipole1
        else:
            B_chic1_guess = self.field_dipole/2
        if self._field_ratio_chicane_dipole2 is not None:
            B_chic2_guess = self.field_chicane_dipole2
        else:
            B_chic2_guess = -self.field_dipole/2
        
        # match the beta function
        from scipy.optimize import minimize
        result_dispersion_R56 = minimize(minfun_dispersion_R56, [B_chic1_guess, B_chic2_guess], tol=1e-16, options={'maxiter': 50})
        self._field_ratio_chicane_dipole1 = result_dispersion_R56.x[0]/self.field_dipole
        self._field_ratio_chicane_dipole2 = result_dispersion_R56.x[1]/self.field_dipole

    
    def cancel_chromaticity(self):
        
        # stop if nonlinearity is turned off
        if not self.use_chromaticity_correction:
            self._strength_sextupole1 = 0.0
            self._strength_sextupole2 = 0.0
            return
        
        # normalizing scale for the merit function
        W_scale = 2*self.length_dipole/self.beta0
        
        # minimizer function for beta matching (central alpha function is zero)
        from abel.utilities.beam_physics import evolve_chromatic_amplitude
        def minfun_Wxy(params):
            ls, inv_rhos, ks, ms, taus = self.matrix_lattice(m1=params[0], m2=params[1], m3=0, half_lattice=True)
            Wx_mid, _ = evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, self.beta0, fast=True) 
            Wy_mid, _ = evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, self.beta0, fast=True, bending_plane=False) 
            return (Wx_mid/W_scale)**2 + (Wy_mid/W_scale)**2
        
        # make estimate
        m1_guess = 90/self.length_quad_gap_or_sextupole
        m2_guess = -90/self.length_quad_gap_or_sextupole
        
        # match the beta function
        from scipy.optimize import minimize
        result_Wxy = minimize(minfun_Wxy, [m1_guess, m2_guess], tol=1e-16, options={'maxiter': 100})
        self._strength_sextupole1 = result_Wxy.x[0]*self.length_quad_gap_or_sextupole
        self._strength_sextupole2 = result_Wxy.x[1]*self.length_quad_gap_or_sextupole
            
    
    def cancel_second_order_dispersion(self):
        
        # stop if nonlinearity is turned off
        if not self.use_central_sextupole:
            self._strength_sextupole3 = 0.0
            return
        
        # guesstimate the sextupole strength (starting point for optimization)
        ml_sext0 = 100
        m_sext0 = ml_sext0/self.length_central_gap_or_sextupole

        # minimizer function for second-order dispersion (central second-order dispersion prime is zero)
        from abel.utilities.beam_physics import evolve_second_order_dispersion
        def minfun_second_order_dispersion(params):
            ls, inv_rhos, ks, ms, taus = self.matrix_lattice(m3=params[0], half_lattice=True)
            _, DDpx, _ = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=True) 
            return DDpx**2
    
        # match the beta function
        from scipy.optimize import minimize
        result_dispersion = minimize(minfun_second_order_dispersion, m_sext0, method='Nelder-Mead', tol=1e-20, options={'maxiter': 50})
        self._strength_sextupole3 = result_dispersion.x[0]*self.length_central_gap_or_sextupole
            
        
    
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
        _, _, evol_beta_x = evolve_beta_function(ls, ks, self.beta0, fast=False)
        _, _, evol_beta_y = evolve_beta_function(ls, -ks, self.beta0, fast=False)
        #_, _, evol_dispersion = evolve_dispersion(ls, inv_rhos, ks, fast=False);
        _, _, evol_second_order_dispersion = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=False);
        _, evol_R56 = evolve_R56(ls, inv_rhos, ks);
        ssl = np.append([0.0], np.cumsum(ls))
        
        # extract into readable format
        ss_beta_x = evol_beta_x[0]
        beta_xs = evol_beta_x[1]
        ss_beta_y = evol_beta_y[0]
        beta_ys = evol_beta_y[1]
        #ss_disp1 = evol_dispersion[0]
        #dispersion = evol_dispersion[1]
        ss_disp2 = evol_second_order_dispersion[0]
        ss_disp1 = ss_disp2
        dispersion = evol_second_order_dispersion[1]
        second_order_dispersion = evol_second_order_dispersion[2]
        ss_R56 = evol_R56[0]
        R56 = evol_R56[1]
        
        # prepare plots
        fig, axs = plt.subplots(4,1, gridspec_kw={'height_ratios': [0.1, 1, 1, 1]})
        fig.set_figwidth(7)
        fig.set_figheight(11)
        col0 = "tab:gray"
        colx1 = "tab:blue"
        coly = "tab:orange"
        colx2 = "#d7e9f5" # lighter version of tab:blue
        colz = "tab:green"
        long_label = 'Location [m]'
        long_limits = [min(ss_beta_x), max(ss_beta_x)]

        # layout
        axs[0].plot(ss_beta_x, np.zeros_like(ss_beta_x), '-', linewidth=0.5, color='k')
        axs[0].axis('off')
        for i in range(len(ls)):
            if abs(inv_rhos[i]) > 0: # add dipoles
                axs[0].add_patch(patches.Rectangle((ssl[i],-0.75), ls[i], 1.5, fc='#d9d9d9'))
            if abs(ks[i]) > 0: # add quads
                axs[0].add_patch(patches.Rectangle((ssl[i],0), ls[i], np.sign(ks[i]), fc='#fcb577'))
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
        axs[1].plot(ss_beta_x, self.beta0*np.ones_like(ss_beta_x), ':', color=col0)
        axs[1].plot(ss_beta_y, np.sqrt(beta_ys), color=coly)
        axs[1].plot(ss_beta_x, np.sqrt(beta_xs), color=colx1)
        axs[1].set_ylabel(r'$\sqrt{\mathrm{Beta\hspace{0.3}function}}$ (m$^{0.5}$)')
        axs[1].set_xlim(long_limits)
        #axs[1].set_yscale('log')
        
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
        _, evol_Wx = evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, self.beta0, fast=False, plot=False);
        _, evol_Wy = evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, self.beta0, fast=False, plot=False, bending_plane=False);
        ssx = evol_Wx[0]
        Wxs = evol_Wx[1]
        ssy = evol_Wy[0]
        Wys = evol_Wy[1]
        
        # prepare plots
        fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [0.1, 1]})
        fig.set_figwidth(7)
        fig.set_figheight(4)
        col0 = "tab:gray"
        colx = "tab:blue"
        coly = "tab:orange"
        long_label = 'Location [m]'
        long_limits = [min(ssx), max(ssx)]

        # layout
        axs[0].plot(ssx, np.zeros_like(ssx), '-', linewidth=0.5, color='k')
        axs[0].axis('off')
        for i in range(len(ls)):
            if abs(inv_rhos[i]) > 0: # add dipoles
                axs[0].add_patch(patches.Rectangle((ssl[i],-0.75), ls[i], 1.5, fc='#d9d9d9'))
            if abs(ks[i]) > 0: # add quads
                axs[0].add_patch(patches.Rectangle((ssl[i],0), ls[i], np.sign(ks[i]), fc='#fcb577'))
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
        axs[1].plot(ssy, Wys, color=coly)
        axs[1].plot(ssx, Wxs, color=colx)
        axs[1].set_ylabel(r'Chromatic amplitude, $W$')
        axs[1].set_xlim(long_limits)
        axs[1].set_yscale('log')
        axs[1].set_ylim([0.1, 2*max(max(Wys), max(Wxs))])
        
         # save figure to file
        if savefig is not None:
            fig.savefig(str(savefig), format="pdf", bbox_inches="tight")

    
    def plot_layout(self, delta=0.1, axes_equal=False, use_second_order_dispersion=True, savefig=None):

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
        print(f'Main dipoles (2x):          {self.length_dipole:.3f} m,  B = {self.field_dipole:.2f} T')
        print(f'Outer quadrupoles (2x):          {self.length_quadrupole:.3f} m,  g = {self.field_gradient_quadrupole1:.1f} T/m')
        print(f'Middle quadrupoles (2x):          {self.length_quadrupole:.3f} m,  g = {self.field_gradient_quadrupole2:.1f} T/m')
        print(f'Inner quadrupoles (2x):          {self.length_quadrupole:.3f} m,  g = {self.field_gradient_quadrupole3:.1f} T/m')
        print(f'Outer chicane dipoles (2x): {self.length_chicane_dipole:.3f} m,  B = {self.field_chicane_dipole1:.3f} T')
        print(f'Inner chicane dipoles (2x): {self.length_chicane_dipole:.3f} m,  B = {self.field_chicane_dipole2:.3f} T')
        if abs(self.field_gradient_sextupole1)>0 or abs(self.field_gradient_sextupole2)>0:
            print(f'Outer sextupoles (2x):          {self.length_quad_gap_or_sextupole:.3f} m,  m = {self.field_gradient_sextupole1:.1f} T/m^2')
            print(f'Inner sextupoles (2x):          {self.length_quad_gap_or_sextupole:.3f} m,  m = {self.field_gradient_sextupole2:.1f} T/m^2')
        if abs(self.field_gradient_sextupole3)>0:
            print(f'Central sextupole (2x):          {self.length_central_gap_or_sextupole:.3f} m,  m = {self.field_gradient_sextupole3:.1f} T/m^2')
        #print(f'Other gaps (12x):           {self.length_gap:.3f} m')
        
        print('------------------------------------------------')
        print(f'             Total length: {self.get_length():.3f} m')
        print(f'         Total bend angle:           {np.rad2deg(self.total_bend_angle()):.2f} deg')
        print('------------------------------------------------')

    def total_bend_angle(self):
        from abel.utilities.relativity import energy2momentum
        p0 = energy2momentum(self.nom_energy)
        BL = 2*(self.length_dipole*self.field_dipole + self.length_chicane_dipole*(self.field_chicane_dipole1 + self.field_chicane_dipole2))
        return BL*SI.e/p0
    
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

    