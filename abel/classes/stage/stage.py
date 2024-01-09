from abc import abstractmethod
from matplotlib import patches
from abel import Trackable, CONFIG
from abel.utilities.plasma_physics import beta_matched
import numpy as np
import scipy.constants as SI
from matplotlib import pyplot as plt
from types import SimpleNamespace
from matplotlib.colors import LogNorm
from abel.utilities.plasma_physics import wave_breaking_field, blowout_radius

class Stage(Trackable):
    
    @abstractmethod
    def __init__(self, length, nom_energy_gain, plasma_density, driver_source=None, ramp_beta_mag=1):

        # common variables
        self.length = length
        self.nom_energy_gain = nom_energy_gain
        self.plasma_density = plasma_density
        self.driver_source = driver_source
        self.ramp_beta_mag = ramp_beta_mag
        
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

    
    @abstractmethod   
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        beam.stage_number += 1
        return super().track(beam, savedepth, runnable, verbose)
    
    def get_length(self):
        return self.length
    
    def get_nom_energy_gain(self):
        return self.nom_energy_gain

    def matched_beta_function(self, energy):
        return beta_matched(self.plasma_density, energy)*self.ramp_beta_mag
    
    def energy_usage(self):
        return self.driver_source.energy_usage()
    
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

    
    def plot_evolution(self):

        # extract wakefield if not already existing
        if not hasattr(self.evolution, 'location'):
            print('No evolution calculated')
            return
            
        # preprate plot
        fig, axs = plt.subplots(2,3)
        fig.set_figwidth(CONFIG.plot_fullwidth_default)
        fig.set_figheight(CONFIG.plot_width_default*0.8)
        col0 = "tab:gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        long_label = 'Location (m)'
        long_limits = [min(self.evolution.location), max(self.evolution.location)]

        # plot energy
        axs[0,0].plot(self.evolution.location, self.evolution.energy / 1e9, color=col1)
        axs[0,0].set_ylabel('Energy (GeV)')
        axs[0,0].set_xlim(long_limits)

        # plot charge
        axs[0,1].plot(self.evolution.location, -self.evolution.charge[0] * np.ones(self.evolution.location.shape) * 1e9, ':', color=col0)
        axs[0,1].plot(self.evolution.location, -self.evolution.charge * 1e9, color=col1)
        axs[0,1].set_ylabel('Charge (nC)')
        axs[0,1].set_xlim(long_limits)
        
        # plot normalized emittance
        axs[0,2].plot(self.evolution.location, self.evolution.emit_ny*1e6, color=col2)
        axs[0,2].plot(self.evolution.location, self.evolution.emit_nx*1e6, color=col1)
        axs[0,2].set_ylabel('Emittance, rms (mm mrad)')
        axs[0,2].set_xlim(long_limits)
        
        # plot energy spread
        axs[1,0].plot(self.evolution.location, self.evolution.rel_energy_spread*1e2, color=col1)
        axs[1,0].set_ylabel('Energy spread, rms (%)')
        axs[1,0].set_xlabel(long_label)
        axs[1,0].set_xlim(long_limits)
        
        # plot beam size
        axs[1,2].plot(self.evolution.location, self.evolution.beam_size_y*1e6, color=col2)
        axs[1,2].plot(self.evolution.location, self.evolution.beam_size_x*1e6, color=col1)
        axs[1,2].set_ylabel('Beam size, rms (um)')
        axs[1,2].set_xlabel(long_label)
        axs[1,2].set_xlim(long_limits)

        # plot transverse offset
        axs[1,1].plot(self.evolution.location, np.zeros(self.evolution.location.shape), ':', color=col0)
        axs[1,1].plot(self.evolution.location, self.evolution.y*1e6, color=col2)  
        axs[1,1].plot(self.evolution.location, self.evolution.x*1e6, color=col1)
        axs[1,1].set_ylabel('Transverse offset (um)')
        axs[1,1].set_xlabel(long_label)
        axs[1,1].set_xlim(long_limits)
        
        plt.show()
        return fig

        
    def plot_wakefield(self):
        
        # extract wakefield if not already existing
        if not hasattr(self.initial.plasma.wakefield.onaxis, 'Ezs'):
            print('No wakefield calculated')
            return
        if not hasattr(self.initial.beam.current, 'Is'):
            print('No beam current calculated')
            return

        # preprate plot
        fig, axs = plt.subplots(2, 1)
        fig.set_figwidth(CONFIG.plot_width_default*0.7)
        fig.set_figheight(CONFIG.plot_width_default*1)
        col0 = "xkcd:light gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        af = 0.1
        
        # extract wakefields and beam currents
        zs0 = self.initial.plasma.wakefield.onaxis.zs
        Ezs0 = self.initial.plasma.wakefield.onaxis.Ezs
        has_final = hasattr(self.final.plasma.wakefield.onaxis, 'Ezs')
        if has_final:
            zs = self.final.plasma.wakefield.onaxis.zs
            Ezs = self.final.plasma.wakefield.onaxis.Ezs
        zs_I = self.initial.beam.current.zs
        Is = self.initial.beam.current.Is

        # find field at the driver and beam
        z_mid = zs_I.min() + (zs_I.max()-zs_I.min())*0.4
        mask = zs_I < z_mid
        zs_masked = zs_I[mask]
        z_beam = zs_masked[np.abs(Is[mask]).argmax()]
        Ez_driver = Ezs0[zs0 > z_mid].max()
        Ez_beam = np.interp(z_beam, zs0, Ezs0)
        
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
        axs[0].set_ylim(bottom=-1.7*np.max([np.abs(Ez_beam), Ez_driver])/1e9, top=1.3*Ez_driver/1e9)
        
        # plot beam current
        axs[1].fill(np.concatenate((zs_I, np.flip(zs_I)))*1e6, np.concatenate((-Is, np.zeros(Is.shape)))/1e3, color=col1, alpha=af)
        axs[1].plot(zs_I*1e6, -Is/1e3, '-', color=col1)
        axs[1].set_xlabel('z (um)')
        axs[1].set_ylabel('Beam current (kA)')
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=1.2*min(-Is)/1e3, top=1.2*max(-Is)/1e3)

        
    # plot wake
    def plot_wake(self, savefig=None):
        
        # extract density if not already existing
        if not hasattr(self.initial.plasma.density, 'rho'):
            print('No wake calculated')
            return
        if not hasattr(self.initial.plasma.wakefield.onaxis, 'Ezs'):
            print('No wakefield calculated')
            return
        
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

            # find field at the driver and beam
            if i==0:
                zs_I = self.initial.beam.current.zs
                Is = self.initial.beam.current.Is
                z_mid = zs_I.max()-(zs_I.max()-zs_I.min())/2
                z_beam = zs_I[np.abs(Is[zs_I < z_mid]).argmax()]
                Ez_driver = Ezs0[zs0 > z_mid].max()
                Ez_beam = np.interp(z_beam, zs0, Ezs0)
                Ezmax = 1.7*np.max([np.abs(Ez_driver), np.abs(Ez_beam)])
            
            # plot on-axis wakefield and axes
            ax2 = ax1.twinx()
            ax2.plot(zs0*1e6, Ezs0/1e9, color = 'black')
            ax2.set_ylabel(r'$E_{z}$' ' (GV/m)')
            ax2.set_ylim(bottom=-Ezmax/1e9, top=Ezmax/1e9)
            axpos = ax1.get_position()
            pad_fraction = 0.13  # Fraction of the figure width to use as padding between the ax and colorbar
            cbar_width_fraction = 0.015  # Fraction of the figure width for the colorbar width
    
            # create colorbar axes based on the relative position and size
            cax1 = fig.add_axes([axpos.x1 + pad_fraction, axpos.y0, cbar_width_fraction, axpos.height])
            cax2 = fig.add_axes([axpos.x1 + pad_fraction + cbar_width_fraction, axpos.y0, cbar_width_fraction, axpos.height])
            cax3 = fig.add_axes([axpos.x1 + pad_fraction + 2*cbar_width_fraction, axpos.y0, cbar_width_fraction, axpos.height])
            clims = np.array([1e-2, 1e3])*self.plasma_density
            
            # plot plasma ions
            p_ions = ax1.imshow(-rho0_plasma/1e6, extent=extent*1e6, norm=LogNorm(), origin='lower', cmap='Greens', alpha=np.array(-rho0_plasma>clims.min(), dtype=float))
            p_ions.set_clim(clims/1e6)
            cb_ions = plt.colorbar(p_ions, cax=cax3)
            cb_ions.set_label(label=r'Beam/plasma-electron/ion density $\mathrm{cm^{-3}}$', size=10)
            cb_ions.ax.tick_params(axis='y',which='both', direction='in')
            
            # plot plasma electrons
            p_electrons = ax1.imshow(rho0_plasma/1e6, extent=extent*1e6, norm=LogNorm(), origin='lower', cmap='Blues', alpha=np.array(rho0_plasma>clims.min()*2, dtype=float))
            p_electrons.set_clim(clims/1e6)
            cb_electrons = plt.colorbar(p_electrons, cax=cax2)
            cb_electrons.ax.tick_params(axis='y',which='both', direction='in')
            cb_electrons.set_ticklabels([])
            
            # plot beam electrons
            p_beam = ax1.imshow(rho0_beam/1e6, extent=extent*1e6,  norm=LogNorm(), origin='lower', cmap='Oranges', alpha=np.array(rho0_beam>clims.min()*2, dtype=float))
            p_beam.set_clim(clims/1e6)
            cb_beam = plt.colorbar(p_beam, cax=cax1)
            cb_beam.set_ticklabels([])
            cb_beam.ax.tick_params(axis='y', which='both', direction='in')
            
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
    