from abel import CONFIG, Experiment, Source, BeamDeliverySystem, Stage, Spectrometer
from matplotlib import pyplot as plt
import numpy as np
import warnings
from abel.utilities.plasma_physics import wave_breaking_field, k_p, beta_matched
import scipy.constants as SI
from scipy.stats import linregress


class ExperimentPWFA(Experiment):
    
    def __init__(self, linac=None, stage=None, spectrometer=None):
        self.stage = stage
        
        super().__init__(linac=linac, component=stage, spectrometer=spectrometer)
            
    
    # assemble the trackables
    def assemble_trackables(self):
        
        # check element classes, then assemble
        if self.component is None:
            self.component = self.stage
        assert(isinstance(self.component, Stage))
        
        # run beamline constructor
        super().assemble_trackables()
        
    # Lebedev eta_t
    def eta_t(self):
        transverse_normalised = self.get_initial_normalised_deflecting()
        focusing_normalised = self.stage.get_normalised_initial_focusing()
        eta_t = transverse_normalised/focusing_normalised
        analytical_eta = transverse_normalised/self.stage.analytical_focusing()
        
        return abs(eta_t), abs(analytical_eta)
    
    # initial normalised deflecting force
    def get_initial_normalised_deflecting(self):
        transverse_normalised = self.stage.get_transverse_sliced()/self.get_beam(1).x_offset()
        return transverse_normalised
        
    # betatron phase advance
    def betatron_phase_advance(self):
        phase_advance = np.sqrt(2)*(np.sqrt(self.get_beam(-2).gamma()) - np.sqrt(self.get_beam(-3).gamma())) * wave_breaking_field(self.stage.plasma_density)/(abs(self.get_rms_accel_initial()))
        return phase_advance
    
    
    def rms_amplitude_flat(self, etabetatron_adv):
        rms_amplitude_A0 = np.exp((etabetatron_adv)**2 /(60 + 2.2*((etabetatron_adv)**1.57)))
        return rms_amplitude_A0
    
    # plot instability evolution
    def plot_instability(self):
        # Frst compute initial and final point on plot from ABEL
        # Get betatron advance
        phase_advance = self.betatron_phase_advance()
        
        amplitude_initial = np.sqrt(np.mean(self.get_beam(-3).gammas()*(self.get_beam(-3).xs()**2/beta_matched(self.stage.plasma_density, self.get_beam(-3).Es()) + beta_matched(self.stage.plasma_density, self.get_beam(-3).Es()) * self.get_beam(-3).xps()**2)))

        amplitude_final = np.sqrt(np.mean(self.get_beam(-2).gammas()*(self.get_beam(-2).xs()**2/beta_matched(self.stage.plasma_density, self.get_beam(-2).Es()) + beta_matched(self.stage.plasma_density, self.get_beam(-2).Es()) * self.get_beam(-2).xps()**2)))

        eta_t, analytical_eta = self.eta_t()
        advances, amplitudes = self.stage.get_amplitudes()
        advances = advances/self.get_rms_accel_initial()
        amplitudes_LEBEDEV = self.rms_amplitude_flat(eta_t*advances)
        ABEL_x = np.array((0, phase_advance*eta_t))
        ABEL_amp = np.array((1, amplitude_final/amplitude_initial))
        amplitudes = amplitudes/amplitudes[0]
        
        # Plot expected figure from HiPACE++ and predicted from LEBEDEV
        plt.plot(advances*eta_t, amplitudes, label = 'HiPACE++', color = 'green')
        plt.plot(advances*eta_t, amplitudes_LEBEDEV, label = 'Lebedev', color = 'grey')
        plt.scatter(ABEL_x, ABEL_amp, label = 'ABEL endpoints', color = 'r')
        y_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        plt.yscale('log')
        plt.yticks(y_ticks, [str(val) for val in y_ticks])
        plt.xlabel(r'$\mu \eta_{t}$')
        plt.ylabel(r'$A/A_{0}$')
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.legend()

        
    def get_rms_accel_initial(self):
        field = (self.get_beam(2).energy() - self.get_beam(1).energy())/(self.stage.length)
        return field
        
    def spectrometer_energy_cal(self, xlims=None, ylims=None, plot_figure = False):
        shots = range(self.num_shots)
        y_mean = np.zeros(len(shots))
        Bs = np.zeros(len(shots))
        Es = np.zeros(len(shots))
        for id, shot in enumerate(shots):
            beam = self[shot].final_beam
            xbins, ybins = None, None
            num_bins = round(np.sqrt(len(beam))/2)
            if xlims is not None:
                xbins = np.linspace(min(xlims), max(xlims), num_bins)
            if ylims is not None:
                ybins = np.linspace(min(ylims), max(ylims), num_bins)
            dQdxdy, xedges, yedges = beam.density_transverse(hbins=xbins, vbins=ybins)
            dQdxdy = abs(dQdxdy)
            proj_y = np.sum(dQdxdy, axis = 1)*1e3
            y_mean[id] = yedges[np.argmax(proj_y)]
            Bs[id] = self[shot].spectrometer.dipole_field
            Es[id] = beam.energy()
        slope, intercept, r_value, p_value, std_err = linregress(Bs/Es, y_mean)
        if plot_figure == True:
            plt.plot(Bs/Es*1e9, (slope*Bs/Es + intercept)*1e3)
            plt.ylabel('y (mm)')
            plt.xlabel(r'B/E'' (T/GeV)')
            plt.scatter(Bs/Es*1e9, y_mean*1e3)
        
        return slope, intercept