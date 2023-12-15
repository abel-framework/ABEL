from abc import abstractmethod
from matplotlib import patches
from abel import Trackable, CONFIG
import numpy as np
import scipy.constants as SI
from matplotlib import pyplot as plt
from types import SimpleNamespace
from matplotlib.colors import LogNorm
from abel.utilities.plasma_physics import wave_breaking_field, blowout_radius

class Stage(Trackable):
    
    @abstractmethod
    def __init__(self, length, nom_energy_gain, plasma_density):

        self.length = length
        self.nom_energy_gain = nom_energy_gain
        self.plasma_density = plasma_density
        
        self.evolution = SimpleNamespace()
        
        self.efficiency = SimpleNamespace()
        
        self.initial = SimpleNamespace()
        self.initial.beam = SimpleNamespace()
        self.initial.beam.current = SimpleNamespace()
        self.initial.beam.density = SimpleNamespace()
        self.initial.plasma = SimpleNamespace()
        self.initial.plasma.density = SimpleNamespace()
        self.initial.plasma.wakefield = SimpleNamespace()
        self.initial.plasma.wakefield.onaxis = SimpleNamespace()

        self.final = SimpleNamespace()
        self.final.beam = SimpleNamespace()
        self.final.beam.current = SimpleNamespace()
        self.final.beam.density = SimpleNamespace()
        self.final.plasma = SimpleNamespace()
        self.final.plasma.density = SimpleNamespace()
        self.final.plasma.wakefield = SimpleNamespace()
        self.final.plasma.wakefield.onaxis = SimpleNamespace()
        
        
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        beam.stage_number += 1
        return super().track(beam, savedepth, runnable, verbose)
    
    def get_length(self):
        return self.length
    
    def get_nom_energy_gain(self):
        return self.nom_energy_gain
    
    @abstractmethod
    def matched_beta_function(self, energy):
        pass
    
    def energy_efficiency(self):
        return self.efficiency

    def calculate_efficiency(self, beam0, driver0, beam, driver):
        Etot0_beam = beam0.total_energy()
        Etot_beam = beam.total_energy()
        Etot0_driver = driver0.total_energy()
        Etot_driver = driver.total_energy()
        self.efficiency.driver_to_wake = (Etot0_driver-Etot_driver)/Etot0_driver
        self.efficiency.wake_to_beam = (Etot_beam-Etot0_beam)/(Etot0_driver-Etot_driver)
        self.efficiency.driver_to_beam = self.efficiency.driver_to_wake*self.efficiency.wake_to_beam

    
    def calculate_beam_current(self, beam0, driver0, beam=None, driver=None):
        
        dz = 40*np.mean([driver0.bunch_length(clean=True)/np.sqrt(len(driver0)), beam0.bunch_length(clean=True)/np.sqrt(len(beam0))])
        num_sigmas = 6
        z_min = beam0.z_offset() - num_sigmas * beam0.bunch_length()
        z_max = driver0.z_offset() + num_sigmas * driver0.bunch_length()
        tbins = np.arange(z_min, z_max, dz)/SI.c
        
        Is0, ts0 = (driver0 + beam0).current_profile(bins=tbins)
        self.initial.beam.current.zs = ts0*SI.c
        self.initial.beam.current.Is = Is0

        if beam is not None and driver is not None:
            Is, ts = (driver + beam).current_profile(bins=tbins)
            self.final.beam.current.zs = ts*SI.c
            self.final.beam.current.Is = Is

    
    @abstractmethod
    def energy_usage(self):
        pass
    
    def plot_wakefield(self):
        
        # extract wakefield if not already existing
        assert hasattr(self.initial.plasma.wakefield.onaxis, 'Ezs'), 'No wakefield'
        assert hasattr(self.initial.beam.current, 'Is'), 'No beam current'

        # preprate plot
        fig, axs = plt.subplots(2, 1)
        fig.set_figwidth(CONFIG.plot_width_default*0.7)
        fig.set_figheight(CONFIG.plot_width_default*1)
        col0 = "xkcd:light gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        af = 0.1

        # extract wakefields
        zs0 = self.initial.plasma.wakefield.onaxis.zs
        Ezs0 = self.initial.plasma.wakefield.onaxis.Ezs
        has_final = hasattr(self.final.beam.current, 'Ezs')
        if has_final:
            zs = self.final.plasma.wakefield.onaxis.zs
            Ezs = self.final.plasma.wakefield.onaxis.Ezs
        
        # get wakefield
        axs[0].plot(zs0*1e6, np.zeros(zs0.shape), '-', color=col0)
        if self.nom_energy_gain is not None:
            axs[0].plot(zs0*1e6, -self.nom_energy_gain/self.get_length()*np.ones(zs0.shape)/1e9, ':', color=col2)
        if self.driver_source.energy is not None:
            Ez_driver_max = self.driver_source.energy/self.get_length()
            axs[0].plot(zs0*1e6, Ez_driver_max*np.ones(zs0.shape)/1e9, ':', color=col0)
        if has_final:
            axs[0].plot(zs*1e6, Ezs/1e9, '-', color=col1, alpha=0.2)
        axs[0].plot(zs0*1e6, Ezs0/1e9, '-', color=col1)
        axs[0].set_xlabel('z (um)')
        axs[0].set_ylabel('Longitudinal electric field (GV/m)')
        zlims = [min(zs0)*1e6, max(zs0)*1e6]
        axs[0].set_xlim(zlims)
        axs[0].set_ylim(bottom=-wave_breaking_field(self.plasma_density)/1e9, top=1.3*max(Ezs0)/1e9)
        
        # plot beam current
        zs_I = self.initial.beam.current.zs
        Is = self.initial.beam.current.Is
        axs[1].fill(np.concatenate((zs_I, np.flip(zs_I)))*1e6, np.concatenate((-Is, np.zeros(Is.shape)))/1e3, color=col1, alpha=af)
        axs[1].plot(zs_I*1e6, -Is/1e3, '-', color=col1)
        axs[1].set_xlabel('z (um)')
        axs[1].set_ylabel('Beam current (kA)')
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=1.2*min(-Is)/1e3, top=1.2*max(-Is)/1e3)

        
    # plot wake
    def plot_wake(self, savefig=None):
        
        # extract density if not already existing
        assert hasattr(self.initial.plasma.density, 'rho'), 'No wake'
        assert hasattr(self.initial.plasma.wakefield.onaxis, 'Ezs'), 'No wakefield'
        
        # calculate densities and extents
        Ezmax = 0.8*wave_breaking_field(self.plasma_density)
        
        # make figures
        has_final_step = hasattr(self.final.plasma.density, 'rho')
        num_plots = 1 + int(has_final_step)
        fig, ax = plt.subplots(num_plots,1)
        fig.set_figwidth(CONFIG.plot_width_default*0.7)
        fig.set_figheight(CONFIG.plot_width_default*0.5*num_plots)

        # cycle through initial and final step
        for i in range(num_plots):
            if not has_final_step:
                ax1 = ax
            else:
                ax1 = ax[i]

            # extract initial or final
            if i==0:
                data_struct = self.initial
                title = 'Initial step'
            elif i==1:
                data_struct = self.final
                title = 'Final step'

            # get data
            extent = data_struct.plasma.density.extent
            zs0 = data_struct.plasma.wakefield.onaxis.zs
            Ezs0 = data_struct.plasma.wakefield.onaxis.Ezs
            rho0_plasma = data_struct.plasma.density.rho
            rho0_beam = data_struct.beam.density.rho

            # plot on-axis wakefield and axes
            zlims = [min(zs0)*1e6, max(zs0)*1e6]
            ax2 = ax1.twinx()
            ax2.plot(zs0*1e6, Ezs0/1e9, color = 'black')
            ax2.set_ylabel(r'$E_{z}$' ' (GV/m)')
            ax2.set_xlim(zlims)
            ax2.set_ylim(bottom=-Ezmax/1e9, top=Ezmax/1e9)
            axpos = ax1.get_position()
            pad_fraction = 0.15  # Fraction of the figure width to use as padding between the ax and colorbar
            cbar_width_fraction = 0.03  # Fraction of the figure width for the colorbar width
    
            # create colorbar axes based on the relative position and size
            cax1 = fig.add_axes([axpos.x1 + pad_fraction, axpos.y0, cbar_width_fraction, axpos.height])
            cax2 = fig.add_axes([axpos.x1 + pad_fraction + cbar_width_fraction, axpos.y0, cbar_width_fraction, axpos.height])
            clims = np.array([1e-2, 1e3])*self.plasma_density
            
            # plot plasma electrons
            initial = ax1.imshow(rho0_plasma/1e6, extent=extent*1e6, norm=LogNorm(), origin='lower', cmap='Blues', alpha=np.array(rho0_plasma>clims.min()*2, dtype=float))
            cb = plt.colorbar(initial, cax=cax1)
            initial.set_clim(clims/1e6)
            cb.ax.tick_params(axis='y',which='both', direction='in')
            cb.set_ticklabels([])
            
            # plot beam electrons
            charge_density_plot0 = ax1.imshow(rho0_beam/1e6, extent=data_struct.beam.density.extent*1e6, norm=LogNorm(), origin='lower', cmap=CONFIG.default_cmap, alpha=np.array(rho0_beam>clims.min()*2, dtype=float))
            cb2 = plt.colorbar(charge_density_plot0, cax = cax2)
            cb2.set_label(label=r'Electron density ' + r'$\mathrm{cm^{-3}}$',size=10)
            cb2.ax.tick_params(axis='y',which='both', direction='in')
            charge_density_plot0.set_clim(clims/1e6)
    
            # set labels
            if i==(num_plots-1):
                ax1.set_xlabel('z (um)')
            ax1.set_ylabel('x (um)')
            ax1.set_title(title)
            ax1.grid(False)
            ax2.grid(False)
            
        # save the figure
        if savefig is not None:
            fig.savefig(str(savefig), bbox_inches='tight', dpi=1000)
        
        return 
    
    def survey_object(self):
        return patches.Rectangle((0, -1), self.get_length(), 2)