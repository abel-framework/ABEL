from abc import abstractmethod
from abel.CONFIG import CONFIG
from abel.classes.beamline.beamline import Beamline
from abel.classes.beamline.impl.linac.linac import Linac
from abel.classes.source.source import Source
from abel.classes.spectrometer import Spectrometer
import numpy as np

class Experiment(Beamline):
    
    def __init__(self, linac=None, test_device=None, spectrometer=None, num_bunches_in_train=1, bunch_separation=0, rep_rate_trains=10):
        super().__init__(num_bunches_in_train, bunch_separation, rep_rate_trains)
        
        self.linac = linac
        self.test_device = test_device
        self.spectrometer = spectrometer
        
    
    
    # assemble the trackables
    def assemble_trackables(self):

        self.trackables = []
        
        # add the linac/source
        assert(isinstance(self.linac, Linac) or isinstance(self.linac, Source))
        self.trackables.append(self.linac)

        # add the experimental test device
        self.trackables.append(self.test_device)

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

    
    # plot the spectrometer screen
    def plot_spectrometer_screen(self, xlims=None, ylims=None):
        
        from matplotlib import pyplot as plt
        
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
        fig, ax = plt.subplots(1, 1)
        fig.set_figwidth(6)
        fig.set_figheight(9)
        
        # plot spectrometer image
        c0 = ax.pcolor(xedges*1e3, yedges * 1e3, abs(dQdxdy) * 1e3, cmap=CONFIG.default_cmap, shading='auto')
        
        # make plot
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title('Spectrometer screen (shot ' + str(self.shot+1) + ')')
        ax.grid(False, which='major')
        cbar0 = fig.colorbar(c0, ax=ax, pad = 0.015*fig.get_figwidth())
        cbar0.ax.set_ylabel('Charge density (nC ' r'$\mathrm{mm^{-2})}$')
        ax.set_ylim(min(yedges * 1e3), max(yedges * 1e3))
    