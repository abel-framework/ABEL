# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.bds.bds import BeamDeliverySystem
from abc import abstractmethod
from types import SimpleNamespace
import numpy as np
import scipy.constants as SI

class BeamDeliverySystemPlasmaLens(BeamDeliverySystem):
    
    @abstractmethod
    def __init__(self, nom_energy=None, beta0=None, alpha0=0, beta_star=None, L_star=2.0, length_scale=2.0, Dpx_star=0.04, charge_sign=-1, cancel_chromaticity=True, enable_csr=True, enable_isr=True, enable_space_charge=False):
        
        super().__init__()

        # main input/output parameters
        self.nom_energy = nom_energy
        self.beta0 = beta0
        self.alpha0 = alpha0
        self.beta_star = beta_star
        self.L_star = L_star
        
        self.Dpx_star = Dpx_star

        self.enable_csr = enable_csr
        self.enable_isr = enable_isr
        self.enable_space_charge = enable_space_charge
        self.cancel_chromaticity = cancel_chromaticity

        self.length_scale = length_scale
        self.length_ratio_gap = 0.15
        self.length_ratio_plasma_lens = 0.15
        self.length_ratio_sextupole = 0.5
        self.length_ratio_dipole1 = 15
        self.length_ratio_dipole2 = 13
        self.length_ratio_dipole3 = 4.5
        self.length_ratio_dipole4 = 2

        self.lens_radius = 1e-3

        # derivable (but also settable) parameters
        self._field_dipole1 = None
        self._field_ratio_dipole2 = None
        self._field_ratio_dipole3 = None
        self._field_ratio_dipole4 = None
        self._strength_plasma_lens1 = None # [1/m]
        self._strength_plasma_lens2 = None # [1/m]
        self._strength_plasma_lens3 = None # [1/m]
        self._nonlinearity_plasma_lens1 = None # [1/m]
        self._nonlinearity_plasma_lens2 = None # [1/m]
        self._strength_sextupole = 0 # [1/m^2]

        self.charge_sign = charge_sign
        
        # evolution (saved when tracking)
        self.evolution = SimpleNamespace()

    
    ## TRACKING

    @abstractmethod
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        "Track the input beam through the BDS lattice. Abstract method."
        return super().track(beam, savedepth, runnable, verbose)

    ## OVERALL LENGTH
    
    # lattice length
    def get_length(self):
        if self.length_dipole1 is not None:
            ls, *_ = self.matrix_lattice(B1=0, B2=0, B3=0, B4=0, k1=0, k2=0, k3=0, m_sext=0, tau_lens1=0, tau_lens2=0)
            return np.sum(ls)
        else:
            return None

    def get_nom_energy(self):
        return self.nom_energy

    @property
    def length_gap(self) -> float:
        "The length of a draft gap [m]."
        return self.length_scale * self.length_ratio_gap
        
    @property
    def length_dipole1(self) -> float:
        "The length of dipole 1 [m]."
        return self.length_scale * self.length_ratio_dipole1
        
    @property
    def length_dipole2(self) -> float:
        "The length of dipole 2 [m]."
        return self.length_scale * self.length_ratio_dipole2
        
    @property
    def length_dipole3(self) -> float:
        "The length of dipole 3 [m]."
        return self.length_scale * self.length_ratio_dipole3

    @property
    def length_dipole4(self) -> float:
        "The length of dipole 4 [m]."
        return self.length_scale * self.length_ratio_dipole4

    @property
    def length_plasma_lens(self) -> float:
        "The length of a plasma lens [m]."
        return self.length_scale * self.length_ratio_plasma_lens

    @property
    def length_sextupole(self) -> float:
        "The length of the sextupole [m]."
        return self.length_scale * self.length_ratio_sextupole

    
    @property
    def field_dipole1(self) -> float:
        if self._field_dipole1 is None:
            self.match()
        return self._field_dipole1
        
    @property
    def field_dipole2(self) -> float:
        if self._field_ratio_dipole2 is None:
            self.match()
        return self._field_dipole1 * self._field_ratio_dipole2

    @property
    def field_dipole3(self) -> float:
        if self._field_ratio_dipole3 is None:
            self.match()
        return self._field_dipole1 * self._field_ratio_dipole3
    
    @property
    def field_dipole4(self) -> float:
        if self._field_ratio_dipole4 is None:
            self.match()
        return self._field_dipole1 * self._field_ratio_dipole4

    
    @property
    def strength_plasma_lens1(self) -> float:
        if self._strength_plasma_lens1 is None:
            self.match_beta_function()
        return self._strength_plasma_lens1
    
    @property
    def strength_plasma_lens2(self) -> float:
        if self._strength_plasma_lens2 is None:
            self.match_beta_function()
        return self._strength_plasma_lens2

    @property
    def strength_plasma_lens3(self) -> float:
        if self._strength_plasma_lens3 is None:
            self.match_beta_function()
        return self._strength_plasma_lens3

    
    @property
    def nonlinearity_plasma_lens1(self) -> float:
        if self._nonlinearity_plasma_lens1 is None:
            Dx_lens = self.Dpx_star*(self.L_star + self.length_plasma_lens/4)
            self._nonlinearity_plasma_lens1 = 1/Dx_lens
        return self._nonlinearity_plasma_lens1
    
    @property
    def nonlinearity_plasma_lens2(self) -> float:
        if self._nonlinearity_plasma_lens2 is None:
            self.match()
        return self._nonlinearity_plasma_lens2

    
    @property
    def strength_sextupole(self) -> float:
        if self._strength_sextupole is None:
            self.match()
        return self._strength_sextupole  
        
        
    ## MATRIX LATTICE

    # full lattice 
    def matrix_lattice(self, B1=None, B2=None, B3=None, B4=None, k1=None, k2=None, k3=None, m_sext=None, tau_lens1=None, tau_lens2=None, invert=False):
            
        # element length array
        dL = self.length_gap
        ls = np.array([dL, self.length_dipole4, dL, self.length_plasma_lens, dL, self.length_dipole3, dL, self.length_plasma_lens, dL, 
                       self.length_dipole2, dL, self.length_sextupole, dL, self.length_dipole1, dL, self.length_plasma_lens, self.L_star])
        
        # bending strength array
        if B1 is None:
            B1 = self.field_dipole1
        if B2 is None:
            B2 = self.field_dipole2
        if B3 is None:
            B3 = self.field_dipole3
        if B4 is None:
            B4 = self.field_dipole3
        Bs = np.array([0, B4, 0, 0, 0, B3, 0, 0, 0, B2, 0, 0, 0, B1, 0, 0, 0])
        
        from abel.utilities.relativity import energy2momentum
        inv_rhos = -self.charge_sign * Bs * SI.e / energy2momentum(self.nom_energy)
        
        # focusing strength array
        if k1 is None:
            k1 = self.strength_plasma_lens1/self.length_plasma_lens
        if k2 is None:
            k2 = self.strength_plasma_lens2/self.length_plasma_lens
        if k3 is None:
            k3 = self.strength_plasma_lens3/self.length_plasma_lens
        ks = np.array([0, 0, 0, k3, 0, 0, 0, k2, 0, 0, 0, 0, 0, 0, 0, k1, 0])
        
        # sextupole strength array
        if m_sext is None:
            m_sext = self.strength_sextupole/self.length_sextupole
        ms = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m_sext, 0, 0, 0, 0, 0])

        # plasma-lens transverse taper array
        if tau_lens1 is None:
            tau_lens1 = self.nonlinearity_plasma_lens1
        if tau_lens2 is None:
            tau_lens2 = self.nonlinearity_plasma_lens2
        taus = np.array([0, 0, 0, 0, 0, 0, 0, tau_lens2, 0, 0, 0, 0, 0, 0, 0, tau_lens1, 0])

        # invert
        if invert:
            ls = np.flip(ls)
            inv_rhos = np.flip(inv_rhos)
            ks = np.flip(ks)
            ms = np.flip(ms)
            taus = np.flip(taus)
        
        return ls, inv_rhos, ks, ms, taus


    def match_beta_function(self):

        ## FIRST MATCH TO MIDPOINT/SEXTUPOLE (BACKWARDS)
        
        # minimizer function for beta matching (central alpha function is zero)
        from abel.utilities.beam_physics import evolve_beta_function
        def minfun_beta_mid(p):
            ls, _, ks, _, _ = self.matrix_lattice(B1=0, B2=0, B3=0, B4=0, k1=p[0], k2=0, k3=0, m_sext=0, tau_lens1=0, tau_lens2=0, invert=True)
            ls = ls[:6]
            ls[-1] = ls[-1]/2
            ks = ks[:6]
            beta, alpha, _ = evolve_beta_function(ls, ks, self.beta_star, fast=True) 
            return alpha**2 + (max(beta/100, self.beta_star)/self.beta_star-1)**2
    
        # initial guess for the lens strength
        f1 = 1/(1/(self.L_star + self.length_plasma_lens/2) + 1/(self.length_dipole1 + 2*self.length_gap+self.length_plasma_lens/2+self.length_sextupole/2))
        k1_guess = 1/(f1*self.length_plasma_lens)
        
        # match the beta function
        from scipy.optimize import minimize
        result_beta1 = minimize(minfun_beta_mid, k1_guess, tol=1e-16, options={'maxiter': 1000})
        k1 = result_beta1.x[0]
        self._strength_plasma_lens1 = k1*self.length_plasma_lens

        ## THEN MATCH TO END (BACKWARDS)

        # minimizer function for beta matching (final alpha function is zero and beta is matched)
        def minfun_beta_end(p):
            ls, _, ks, _, _ = self.matrix_lattice(B1=0, B2=0, B3=0, B4=0, k1=k1, k2=p[0], k3=p[1], m_sext=0, tau_lens1=0, tau_lens2=0, invert=True)
            beta, alpha, _ = evolve_beta_function(ls, ks, self.beta_star, fast=True) 
            return (beta/self.beta0-1)**2 + (alpha-self.alpha0)**2

        # initial guess for the lens strength2
        f2 = 1.1/(1/(self.length_dipole2 + 2*self.length_gap + self.length_plasma_lens/2 + self.length_sextupole/2) + 1/(self.length_dipole3 + 2*self.length_gap + self.length_plasma_lens))
        k2_guess = 1/(f2*self.length_plasma_lens)
        k3_guess = -2/(self.length_plasma_lens)

        # match the beta function
        result_beta2 = minimize(minfun_beta_end, [k2_guess, k2_guess], tol=1e-16, options={'maxiter': 1000})
        self._strength_plasma_lens2 = result_beta2.x[0]*self.length_plasma_lens
        self._strength_plasma_lens3 = result_beta2.x[1]*self.length_plasma_lens


    def match_dispersion(self):

        ## FIRST MATCH DISPERSION TO SECOND LENS (BACKWARDS)
        
        # minimizer function for beta matching (central alpha function is zero)
        from abel.utilities.beam_physics import evolve_chromatic_amplitude
        def minfun_chrom1(p):
            ls, inv_rhos, ks, ms, taus = self.matrix_lattice(B1=0, B2=0, B3=0, B4=0, m_sext=0, tau_lens1=p[0], tau_lens2=0, invert=True)
            ind = 2
            Wx, _ = evolve_chromatic_amplitude(ls[:ind], inv_rhos[:ind], ks[:ind], ms[:ind], taus[:ind], self.beta_star, Dpx0=self.Dpx_star, fast=True) 
            return Wx**2

        tau1_guess = 1/(self.Dpx_star*self.L_star)
        
        # cancel the chromaticity
        if self.cancel_chromaticity:
            from scipy.optimize import minimize
            result_chrom1 = minimize(minfun_chrom1, tau1_guess, tol=1e-16, options={'maxiter': 1000})
            self._nonlinearity_plasma_lens1 = result_chrom1.x[0]
        else:
            self._nonlinearity_plasma_lens1 = 0
        
        
        ## FIRST MATCH DISPERSION TO SECOND LENS (BACKWARDS)

        # find betas in the lenses (to compare nonlinear kicks)
        from abel.utilities.beam_physics import evolve_beta_function
        ls, _, ks, _, _ = self.matrix_lattice(B1=0, B2=0, B3=0, B4=0, m_sext=0, tau_lens1=0, tau_lens2=0, invert=True)
        ind1 = 2
        beta1, _, _ = evolve_beta_function(ls[:ind1], ks[:ind1], self.beta_star, fast=True) 
        ind2 = 9
        beta2, _, _ = evolve_beta_function(ls[:ind2], ks[:ind2], self.beta_star, fast=True) 

        Dx_scale = self.Dpx_star*self.L_star
        Dpx_scale = self.Dpx_star
        Wx_scale = self.L_star/self.beta_star
        
        # minimizer function for beta matching (central alpha function is zero)
        from abel.utilities.beam_physics import evolve_dispersion, evolve_chromatic_amplitude
        def minfun_chrom_disp(p):
            ls, inv_rhos, ks, ms, taus = self.matrix_lattice(B1=p[0], B2=p[0], B3=p[1], B4=p[2], m_sext=0, tau_lens2=p[3], invert=True)
            
            # dispersion at the end
            Dx, Dpx, _ = evolve_dispersion(ls, inv_rhos, ks, Dpx0=self.Dpx_star, fast=True) 
            
            # dispersion at the second lens
            ind = 9
            Dx2, _, _ = evolve_dispersion(ls[:ind], inv_rhos[:ind], ks[:ind], Dpx0=self.Dpx_star, fast=True) 

            kl1 = self._strength_plasma_lens1
            kl2 = self._strength_plasma_lens2
            tau1 = self._nonlinearity_plasma_lens1
            tau2 = p[3]
            
            # chromaticity at the end
            Wx, _ = evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, self.beta_star, Dpx0=self.Dpx_star, fast=True) 
            
            return (Dx/Dx_scale)**2 + (Dpx/Dpx_scale)**2 + (Wx/Wx_scale)**2 + ((kl2*tau2*beta2)/(kl1*tau1*beta1)-1)**2
                
        # cancel the chromaticity
        from scipy.optimize import minimize
        B0 = Dx_scale*self.nom_energy/(SI.c*self.length_dipole1**2)
        guess = [B0, B0, B0, tau1_guess*5]
        result_chrom_disp = minimize(minfun_chrom_disp, guess, tol=1e-16, options={'maxiter': 1000})
        self._field_dipole1 = result_chrom_disp.x[0]
        self._field_ratio_dipole2 = 1
        self._field_ratio_dipole3 = result_chrom_disp.x[1]/result_chrom_disp.x[0]
        self._field_ratio_dipole4 = result_chrom_disp.x[2]/result_chrom_disp.x[0]
        if self.cancel_chromaticity:
            self._nonlinearity_plasma_lens2 = result_chrom_disp.x[3]
        else:
            self._nonlinearity_plasma_lens2 = 0
        
    def match(self):
        self.match_beta_function()
        self.match_dispersion()
        

    ## PLOTTING OPTICS

    def plot_optics(self, show_beta_function=True, show_dispersion=True, show_second_order_dispersion=False, show_chromaticity=True, invert=False, save_fig=None):
        """
        Plot the beta function, dispersion and chromaticity along the BDS.
        """
        
        from matplotlib import pyplot as plt
        from matplotlib import patches
        from copy import deepcopy
        import string
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_chromatic_amplitude, evolve_second_order_dispersion

        
        # calculate evolution
        ls, inv_rhos, ks, ms, taus = self.matrix_lattice(invert=invert)
        ssl = np.append([0.0], np.cumsum(ls))

        if invert:
            Dpx0 = self.Dpx_star
            beta0 = self.beta_star
        else:
            Dpx0 = 0
            beta0 = self.beta0
            
        if show_beta_function:
            _, _, evol_beta_x = evolve_beta_function(ls, ks, beta0, inv_rhos=inv_rhos, fast=False)
            ss_beta = evol_beta_x[0]
            beta_xs = evol_beta_x[1]

        if show_dispersion:
            _, _, evol_dispersion = evolve_dispersion(ls, inv_rhos, ks, Dx0=0, Dpx0=Dpx0);
            ss_disp = evol_dispersion[0]
            dispersion = evol_dispersion[1]
            if show_second_order_dispersion:
                _, _, evol_second_order_dispersion = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=False);
                ss_disp2 = evol_second_order_dispersion[0]
                second_order_dispersion = evol_second_order_dispersion[2]
        
        if show_chromaticity:
            _, evol_Wx = evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, beta0, Dx0=0, Dpx0=Dpx0);
            ss_W = evol_Wx[0]
            Wxs = evol_Wx[1]
            
        # prepare plots
        num_plots = 1 + int(show_beta_function) + int(show_dispersion) + int(show_chromaticity)
        height_ratios = np.ones((num_plots,1))
        height_ratios[0] = 0.1
        fig, axs = plt.subplots(num_plots,1, gridspec_kw={'height_ratios': height_ratios})
        fig.set_figwidth(7)
        fig.set_figheight(11/3.1*np.sum(height_ratios))
        col0 = "tab:gray"
        colx1 = "tab:blue"
        coly = "tab:orange"
        colx2 = "#d7e9f5" # lighter version of tab:blue
        colz = "tab:green"
        coloff = "#e69596" # lighter version of tab:red
        long_label = 'Location (m)'
        long_limits = [min(ssl)-1, max(ssl)+1]

        # layout
        n = 0
        axs[n].plot(ssl, np.zeros_like(ssl), '-', linewidth=0.5, color='k')
        axs[n].axis('off')
        for i in range(len(ls)):
            if abs(inv_rhos[i]) > 0: # add dipoles
                axs[n].add_patch(patches.Rectangle((ssl[i],-0.75), ls[i], 1.5, fc='#d9d9d9'))
            if abs(ks[i]) > 0: # add quad or plasma lenses
                axs[n].add_patch(patches.Rectangle((ssl[i],0), ls[i], np.sign(ks[i]), fc='#fcb577'))
            if abs(ms[i]) > 0: # add sextupole
                axs[n].add_patch(patches.Rectangle((ssl[i],-0.5), ls[i], 1, fc='#abd4ab'))
        axs[n].set_xlim(long_limits)
        axs[n].set_ylim([-1, 1])

        # shift the layout box down
        box = axs[0].get_position()
        vshift = 0.025
        box.y0 = box.y0 - vshift
        box.y1 = box.y1 - vshift
        axs[0].set_position(box)
        
        # plot beta function
        if show_beta_function:
            n += 1
            axs[n].plot(ss_beta, self.beta0*np.ones_like(ss_beta), ':', color=col0, label=r'Initial beta function, $\beta_0$')
            axs[n].plot(ss_beta, self.beta_star*np.ones_like(ss_beta), ':', color=colz, label=r'Final beta function, $\beta^*$')
            axs[n].plot(ss_beta, beta_xs, color=colx1)
            axs[n].legend(loc='lower left', reverse=True, fontsize='small')
            axs[n].set_ylabel('Beta function (m)')
            axs[n].set_xlim(long_limits)
            axs[n].set_yscale('log')
            axs[n].text(0.01, 0.90, f'({string.ascii_lowercase[n-1]})', transform=axs[n].transAxes, size=13)
        
        # plot dispersion
        if show_dispersion:
            n += 1
            axs[n].plot(ss_disp, np.zeros_like(ss_disp), ':', color=col0)
            axs[n].plot(ss_disp, dispersion / 1e-3, '-', color=colx1)
            if show_second_order_dispersion:
                axs[n].plot(ss_disp2, second_order_dispersion / 1e-3, '-', color=colx2)
            axs[n].set_ylabel('Horizontal dispersion (mm)')
            axs[n].set_xlim(long_limits)
            #axs[n].legend(loc='lower left', reverse=True, fontsize='small')
            axs[n].text(0.01, 0.90, f'({string.ascii_lowercase[n-1]})', transform=axs[n].transAxes, size=13)
        
        # plot chromaticity
        if show_chromaticity:
            n += 1
            axs[n].plot(ss_W, Wxs, color=colx1)
            axs[n].set_ylabel(r'Chromatic amplitude, $W$')
            axs[n].set_xlim(long_limits)
            axs[n].text(0.01, 0.90, f'({string.ascii_lowercase[n-1]})', transform=axs[n].transAxes, size=13)

        axs[n].set_xlabel(r'Longitudinal position, $s$ (m)')
        
        # save figure to file
        if save_fig is not None:
            fig.savefig(str(save_fig), format="pdf", bbox_inches="tight")
        

    def plot_evolution(self, save_fig=None):
        """
        Plot the evolution of various beam parameters inside the BDS.
        """

        from matplotlib import pyplot as plt
        
        evol = self.evolution

        # stop if no evolution calculated
        if not hasattr(evol, 'location'):
            print('No evolution calculated.')
            return
        
        # prepare plot
        fig, axs = plt.subplots(3,3)
        fig.set_figwidth(20)
        fig.set_figheight(12)
        col0 = "tab:gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        long_label = 'Location [m]'
        long_limits = [min(evol.location)-1, max(evol.location)+1]

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
        axs[0,2].plot(evol.location, evol.emit_ny*1e6, color=col2, label='y')
        axs[0,2].plot(evol.location, evol.emit_nx*1e6, color=col1, label='x')
        axs[0,2].set_ylabel('Emittance, rms [mm mrad]')
        axs[0,2].set_xlim(long_limits)
        axs[0,2].set_yscale('log')
        axs[0,2].legend(loc='upper left', reverse=True, fontsize='small')
        
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
        axs[2,2].plot(evol.location, evol.beam_size_y*1e6, color=col2, label='y')  
        axs[2,2].plot(evol.location, evol.beam_size_x*1e6, color=col1, label='x')
        axs[2,2].set_ylabel(r'Beam size, rms [$\mathrm{\mu}$m]')
        axs[2,2].set_xlabel(long_label)
        axs[2,2].set_xlim(long_limits)
        axs[2,2].set_yscale('log')
        axs[2,2].legend(loc='upper left', reverse=True, fontsize='small')
        
        plt.show()
        
        # save figure to file
        if save_fig is not None:
            fig.savefig(str(save_fig), format="pdf", bbox_inches="tight")
            
    
    ## COST MODEL
    
    def get_cost_breakdown(self):
        return ('BDS', self.get_length() * CostModeled.cost_per_length_bds)
