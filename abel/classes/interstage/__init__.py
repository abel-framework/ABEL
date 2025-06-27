from abc import abstractmethod
from abel.classes.trackable import Trackable
from abel.classes.cost_modeled import CostModeled
from types import SimpleNamespace
import numpy as np
import scipy.constants as SI

class Interstage(Trackable, CostModeled):
    
    @abstractmethod
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0, charge_sign=-1,
                 cancel_chromaticity=True, cancel_sec_order_dispersion=False, use_apertures=True, enable_csr=True, enable_isr=True, enable_space_charge=False, uses_plasma_lenses=None):
        
        super().__init__()

        # main parameters
        self.nom_energy = nom_energy
        self._R56 = R56
        self.charge_sign = charge_sign
        self.uses_plasma_lenses = uses_plasma_lenses
        
        # main parameters (functions of energy)
        self._beta0 = beta0
        self._length_dipole = length_dipole
        self._field_dipole = field_dipole
        
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
    
    
    ## MATRIX LATTICE

    @abstractmethod
    def matrix_lattice(self, orbit_only=False):
        pass
        

    
    ## MATCHING
    
    def match(self):
        "Combined matching the beta function, first- and second-order dispersion and the R56"
        self.match_beta_function()
        self.match_dispersion_and_R56()
        if self.cancel_chromaticity:
            self.match_chromatic_amplitude()
        if self.cancel_sec_order_dispersion:
            self.match_second_order_dispersion()

    @abstractmethod
    def match_beta_function(self):
        pass

    @abstractmethod
    def match_dispersion_and_R56(self):
        pass
    
    @abstractmethod
    def match_chromatic_amplitude(self):
        pass

    @abstractmethod
    def match_second_order_dispersion(self):
        pass
        
    
    ## PLOTTING
    
    def plot_evolution(self):

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
        if self.uses_plasma_lenses:
            ks_y = ks
        else:
            ks_y = -ks
        _, _, evol_beta_x = evolve_beta_function(ls, ks, self.beta0, inv_rhos=inv_rhos, fast=False)
        _, _, evol_beta_y = evolve_beta_function(ls, ks_y, self.beta0, fast=False)
        _, _, evol_second_order_dispersion = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=False);
        _, evol_R56 = evolve_R56(ls, inv_rhos, ks);
        ssl = np.append([0.0], np.cumsum(ls))
        
        # extract into readable format
        ss_beta = evol_beta_x[0]
        beta_xs = evol_beta_x[1]
        ss_beta_y = evol_beta_y[0]
        beta_ys = evol_beta_y[1]
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
        long_limits = [min(ss_beta), max(ss_beta)]

        # layout
        axs[0].plot(ss_beta, np.zeros_like(ss_beta), '-', linewidth=0.5, color='k')
        axs[0].axis('off')
        for i in range(len(ls)):
            if abs(inv_rhos[i]) > 0: # add dipoles
                axs[0].add_patch(patches.Rectangle((ssl[i],-0.75), ls[i], 1.5, fc='#d9d9d9'))
            if abs(ks[i]) > 0: # add quad or plasma lenses
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
        axs[1].plot(ss_beta, self.beta0*np.ones_like(ss_beta), ':', color=col0)
        axs[1].plot(ss_beta_y, np.sqrt(beta_ys), color=coly)
        axs[1].plot(ss_beta, np.sqrt(beta_xs), color=colx1)
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
        _, evol_Wx = evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, self.beta0, fast=False, plot=False);
        if self.uses_plasma_lenses:
            evol_Wy = evol_Wx
        else:
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
        axs[1].set_ylim([0.3, 1.3*max(max(Wys), max(Wxs))])
        
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

    @abstractmethod
    def print_summary(self):
        pass

    
    ## SURVEY PLOTS
    
    def total_bend_angle(self):
        ls, inv_rhos, ks, ms, taus = self.matrix_lattice(orbit_only=True)
        from abel.utilities.beam_physics import evolve_orbit
        final_angle, _ = evolve_orbit(ls, inv_rhos, theta0=0, fast=True)
        return final_angle

    def survey_object(self):
        
        ls, inv_rhos, ks, ms, taus = self.matrix_lattice(orbit_only=True)
        from abel.utilities.beam_physics import evolve_orbit
        final_angle, evol_orbit = evolve_orbit(ls, inv_rhos, theta0=0)
        x_points = evol_orbit[0,:]
        y_points = evol_orbit[1,:]
        ss = evol_orbit[2,:]
        
        label = 'Interstage'
        color = 'orange'
        return x_points, y_points, final_angle, label, color

        
    ## COST MODEL
    
    def get_cost_breakdown(self):
        return ('Interstage', self.get_length() * CostModeled.cost_per_length_interstage)

    