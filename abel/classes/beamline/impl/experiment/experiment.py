# Copyright 2022-, The ABEL Authors
# Authors: C.A. Lindstrøm, B. Chen, K. Sjobak, E. Adli
# License: GPL-3.0-or-later

from abc import abstractmethod
from abel.CONFIG import CONFIG
from abel.classes.beamline.beamline import Beamline
from abel.classes.beamline.impl.linac.linac import Linac
from abel.classes.source.source import Source
from abel.classes.spectrometer.spectrometer import Spectrometer
import numpy as np

class Experiment(Beamline):
    
    def __init__(self, linac=None, component=None, spectrometer=None, num_bunches_in_train=1, bunch_separation=0, rep_rate_trains=10):
        super().__init__(num_bunches_in_train, bunch_separation, rep_rate_trains)
        
        self.linac = linac
        self.component = component
        self.spectrometer = spectrometer
        
    
    
    # assemble the trackables
    def assemble_trackables(self):

        self.trackables = []
        
        # add the linac/source
        assert(isinstance(self.linac, Linac) or isinstance(self.linac, Source))
        self.trackables.append(self.linac)

        # add the experimental component
        self.trackables.append(self.component)

        # add the spectrometer (if it exists
        if self.spectrometer is not None:
            assert(isinstance(self.spectrometer, Spectrometer))
            self.trackables.append(self.spectrometer)
        
        # set the bunch train pattern etc.
        super().assemble_trackables()


    def energy_usage(self):
        return 0

    def get_nom_beam_power(self):
        return 0

    def get_cost_breakdown():
        return ('Experiment', 0)
    
    # density plots
    def plot_spectrometer_screen(self, xlims=None, ylims=None, E_calib = False, diverg = None, plot_m12 = False, savefig = None):
        
        from matplotlib import pyplot as plt
        import warnings

        # load phase space
        beam = self.final_beam

        # make screen projection
        xbins, ybins = None, None
        num_bins = round(np.sqrt(len(beam))/2)
        if xlims is not None:
            xbins = np.linspace(min(xlims), max(xlims), num_bins)
        if ylims is not None:
            ybins = np.linspace(min(ylims), max(ylims), num_bins)
        dQdxdy, xedges, yedges = beam.density_transverse(hbins=xbins, vbins=ybins)
        
        # prepare figure
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
        fig.set_figwidth(8)
        fig.set_figheight(8)
        
        # current profile
        c0 = ax[0].pcolor(xedges*1e3, yedges * 1e3, abs(dQdxdy) * 1e3, cmap=CONFIG.default_cmap, shading='auto')
        
        # make plot
        ax[0].set_xlabel('x (mm)')
        ax[0].set_ylabel('y (mm)')
        ax[0].set_title('Spectrometer screen (shot ' + str(self.shot+1) + ')')
        ax[0].grid(False, which='major')
        cbar0 = fig.colorbar(c0, ax=ax[0], pad = 0.015*fig.get_figwidth())
        cbar0.ax.set_ylabel('Charge density (nC ' r'$\mathrm{mm^{-2})}$')
        ax[0].set_ylim(min(yedges * 1e3), max(yedges * 1e3))
        
        # calculate energy axis (E = E0*Dy/y)
        Dy_img = self.spectrometer.get_dispersion(energy=self.spectrometer.img_energy)
        E_times_Dy = self.spectrometer.img_energy*Dy_img
        
        # add imaging energies
        y_img = -E_times_Dy/self.spectrometer.img_energy
        y_img_y = -E_times_Dy/self.spectrometer.img_energy_y
        ax[0].axhline(y_img*1e3, color = 'black', linestyle = '--', linewidth = 1, label = 'Imaging energy x (GeV)')
        ax[0].axhline(y_img_y*1e3, color = 'black', linestyle = ':', linewidth = 1, label = 'Imaging energy y (GeV)')
        
        # add energy axis
        warnings.simplefilter('ignore', category=RuntimeWarning)
        ax2 = ax[0].secondary_yaxis('right', functions=(lambda y: -E_times_Dy/(y/1e3)/1e9, lambda y: -E_times_Dy/(y/1e3)/1e9))
        ax2.set_ylabel('Energy (GeV)')
        
        if diverg is not None:
            energies = -E_times_Dy/yedges
            m12s = self.spectrometer.get_m12(energies)        
            x_pos = m12s*diverg
            x_neg = -m12s*diverg
            ax[0].plot(x_pos*1e3, yedges*1e3, color = 'orange', label = str(diverg*1e3) + ' mrad butterfly', alpha = 0.7)
            ax[0].plot(x_neg*1e3, yedges*1e3, color = 'orange', alpha = 0.7)
            ax[1].plot(energies/1e9, m12s)
            ax[1].set_ylabel(r'$\mathrm{m_{12}}$')
            ax[1].set_xlabel('E (GeV)')
            ax[1].grid(False, which='major')
            
        ax[0].set_xlim(min(xedges*1e3), max(xedges*1e3))
        ax[0].legend(loc = 'center right', fontsize = 6)
        
        if plot_m12 == False:
            fig.delaxes(ax[1])
        if savefig is not None:
            plt.savefig(str(savefig), dpi = 700)
    