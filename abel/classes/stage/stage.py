from abc import abstractmethod
from matplotlib import patches
from abel import Trackable, CONFIG
from abel.classes.cost_modeled import CostModeled
from abel.classes.source.impl.source_capsule import SourceCapsule
from abel.utilities.plasma_physics import beta_matched
import numpy as np
import copy
import scipy.constants as SI
from matplotlib import pyplot as plt
from types import SimpleNamespace
from matplotlib.colors import LogNorm
from abel.utilities.plasma_physics import wave_breaking_field, blowout_radius, beta_matched

class Stage(Trackable, CostModeled):
    
    @abstractmethod
    def __init__(self, nom_accel_gradient, nom_energy_gain, plasma_density, driver_source=None, ramp_beta_mag=None):

        super().__init__()
        
        # common variables
        self.nom_accel_gradient = nom_accel_gradient
        self.nom_energy_gain = nom_energy_gain
        self.plasma_density = plasma_density
        self.driver_source = driver_source
        self.ramp_beta_mag = ramp_beta_mag
        
        self.stage_number = None
        
        # nominal initial energy
        self.nom_energy = None 

        self.upramp = None
        self.downramp = None
        self._return_tracked_driver = False
        
        self.evolution = SimpleNamespace()
        self.evolution.beam = SimpleNamespace()
        self.evolution.beam.slices = SimpleNamespace()
        self.evolution.driver = SimpleNamespace()
        self.evolution.driver.slices = SimpleNamespace()
        
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

        self.name = 'Plasma stage'

    
    @abstractmethod   
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        beam.stage_number += 1
        return super().track(beam, savedepth, runnable, verbose)

    # upramp to be tracked before the main tracking
    def track_upramp(self, beam0, driver0=None):
        if self.upramp is not None and isinstance(self.upramp, Stage):

            # set driver
            self.upramp.driver_source = SourceCapsule(beam=driver0)

            # determine density if not already set
            if self.upramp.plasma_density is None:
                self.upramp.plasma_density = self.plasma_density/self.ramp_beta_mag

            # determine length if not already set
            if self.upramp.length is None:
            
                # set ramp length (uniform step ramp)
                if self.nom_energy is None:
                    self.nom_energy = beam0.energy()
                
                self.upramp.length = beta_matched(self.plasma_density, self.nom_energy)*np.pi/(2*np.sqrt(1/self.ramp_beta_mag))
            
            # perform tracking
            self.upramp._return_tracked_driver = True
            beam, driver = self.upramp.track(beam0)
            beam.stage_number -= 1
            driver.stage_number -= 1
            
        else:
            beam = beam0
            driver = driver0
            
        return beam, driver

    # downramp to be tracked after the main tracking
    def track_downramp(self, beam0, driver0):
        if self.downramp is not None and isinstance(self.downramp, Stage):

            # set driver
            self.downramp.driver_source = SourceCapsule(beam=driver0)
            
            # determine density if not already set
            if self.downramp.plasma_density is None:
                
                # set ramp density
                self.downramp.plasma_density = self.plasma_density/self.ramp_beta_mag
            
            # determine length if not already set
            if self.downramp.length is None:
                
                # set ramp length (uniform step ramp)
                if self.nom_energy is None:
                    self.nom_energy = beam0.energy()
                
                self.downramp.length = beta_matched(self.plasma_density, self.nom_energy+self.nom_energy_gain)*np.pi/(2*np.sqrt(1/self.ramp_beta_mag))
            
            # perform tracking
            self.downramp._return_tracked_driver = True
            beam, driver = self.downramp.track(beam0)
            beam.stage_number -= 1
            driver.stage_number -= 1
            
        else:
            beam = beam0
            driver = driver0
            
        return beam, driver

    #length/length_flattop, nom_accel_gradient, and nom_energy_gain are interdependent
    # Allow explicitly setting 2 out of 3, and overwriting
    # Setting variable to None means it is not set.
    def _lengthGradientEnergy_rdy(self):
        hasData = 0
        if self._length_rdy():
            hasData += 1
        if self._nom_energy_gain    is not None:
            hasData += 1
        if self._nom_accel_gradient is not None:
            hasData += 1

        if hasData == 2:
            return True
        if hasData > 2:
            raise VariablesOverspecifiedError("Internal error, length/gradient/energy overspecified")
        return False

    #Define all length/gradient/energy gain as undefined to begin with
    _length             = None
    _nom_accel_gradient = None
    _nom_energy_gain    = None

    #Can we get the length of the total thing?
    def _length_rdy(self) -> bool:
        if self._length is not None or self._length_flattop is not None:
            return True
        return False
    #Negative length_flattop?
    def _length_sanitycheck(self):
        if self._length_rdy() and self.length_flattop < 0.0:
            print(f"WARNING: The current total length and ramp length settings implicitly makes length_flattop = {self.length_flattop} < 0")

    #Similar, between length/length_flattop
    _length_flattop             = None
    #_nom_accel_gradient_flattop = None
    #_nom_energy_gain_flattop    = None

    @property
    def length_flattop(self) -> float:
        if self._length_flattop is not None:
            return self._length_flattop
        if self._length_rdy():
            L = self.length
            if self.length_upramp is not None:
                L -= self.length_upramp
            if self.length_downramp is not None:
                L -= self.length_downramp
            return L
        if self._lengthGradientEnergy_rdy():
            return self.nom_energy_gain/self.nom_accel_gradient
        return None
    @length_flattop.setter
    def length_flattop(self, length_flattop : float):
        if self._length_flattop is not None:
            self._length_flattop = length_flattop
            return
        if self._length_rdy():
            raise VariablesOverspecifiedError("Have already set length, cannot also set length_flattop")
        if self._lengthGradientEnergy_rdy():
            raise VariablesOverspecifiedError("Have already set gradient/energy, cannot also set length_flattop/length")
        self._length_flattop = length_flattop

    @property
    def length_upramp(self) -> float:
        if self.upramp is not None:
            return self.upramp.length
        else:
            return None
    @length_upramp.setter
    def length_upramp(self, length_upramp : float):
        if self.upramp is None:
            raise("No upramp to set length of")
        self.upramp.length = length_upramp
        self._length_sanitycheck()

    @property
    def length_downramp(self) -> float:
        if self.downramp is not None:
            return self.downramp.length
        else:
            return None
    @length_downramp.setter
    def length_downramp(self, length_downramp : float):
        if self.downramp is None:
            raise StageError("No downramp to set length of")
        self.downramp.length = length_downramp
        self._length_sanitycheck()

    @property
    def length(self) -> float:
        if self._length is not None:
            return self._length
        if self._length_rdy():
            L = self.length_flattop
            if self.length_upramp is not None:
                L += self.length_upramp
            if self.length_downramp is not None:
                L += self.length_downramp
            return L
        if self._lengthGradientEnergy_rdy():
            return self.nom_energy/self.nom_accel_gradient
        return None
    @length.setter
    def length(self, length : float):
        if self._length is not None:
            self._length = length
            self._length_sanitycheck()
            return
        if self._length_rdy():
            raise VariablesOverspecifiedError("Have already set length_flattop, cannot also set length")
        if self._lengthGradientEnergy_rdy():
            raise VariablesOverspecifiedError("Have already set gradient/energy, cannot also set length_flattop/length")
        self._length = length
        self._length_sanitycheck()
    def get_length(self) -> float:
        return self.length

    @property
    def nom_energy_gain(self) -> float:
        if self._nom_energy_gain is not None:
            return self._nom_energy_gain
        if self._lengthGradientEnergy_rdy():
            return self.nom_accel_gradient*self.length_flattop
        return None
    @nom_energy_gain.setter
    def nom_energy_gain(self, nom_energy_gain : float):
        if self._nom_energy_gain is not None:
            self._nom_energy_gain = nom_energy_gain
            return
        if self._lengthGradientEnergy_rdy():
            raise VariablesOverspecifiedError("Have already set length/gradient, cannot also set energy")
        self._nom_energy_gain = nom_energy_gain
    def get_nom_energy_gain(self):
        return self.nom_energy_gain

    @property
    def nom_accel_gradient(self) -> float:
        if self._nom_accel_gradient is not None:
            return self._nom_accel_gradient
        if self._lengthGradientEnergy_rdy():
            return self.nom_energy_gain/self.length_flattop
        return None
    @nom_accel_gradient.setter
    def nom_accel_gradient(self, nom_accel_gradient : float):
        if self._nom_accel_gradient is not None:
            self._nom_accel_gradient = nom_accel_gradient
            return
        if self._lengthGradientEnergy_rdy():
            raise VariablesOverspecifiedError("Have already set length/energy, cannot also set gradient.")
        self._nom_accel_gradient = nom_accel_gradient
    def get_nom_accel_gradient(self):
        return self.nom_accel_gradient



    def get_cost_breakdown(self):
        breakdown = []
        breakdown.append(('Plasma cell', self.get_length() * CostModeled.cost_per_length_plasma_stage))
        breakdown.append(('Driver dump', CostModeled.cost_per_driver_dump))
        return (self.name, breakdown)

    def matched_beta_function(self, energy_incoming):
        if self.ramp_beta_mag is not None:
            return beta_matched(self.plasma_density, energy_incoming)*self.ramp_beta_mag
        else:
            return beta_matched(self.plasma_density, energy_incoming)
            
    def matched_beta_function_flattop(self, energy):
        return beta_matched(self.plasma_density, energy)
    
    def energy_usage(self):
        return self.driver_source.energy_usage()
    
    def energy_efficiency(self):
        return self.efficiency

    #@abstractmethod   # TODO: calculate the dumped power and use it for the dump cost model.
    def dumped_power(self):
        return None

    
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

    def save_driver_to_file(self, driver, runnable):
        driver.save(runnable, beam_name='driver_stage' + str(driver.stage_number+1))

    
    def save_evolution_to_file(self, bunch='beam'):
    
        # select bunch
        if bunch == 'beam':
            evol = self.evolution.beam
        elif bunch == 'driver':
            evol = self.evolution.driver

        # arrange numbers into a matrix
        matrix = np.empty((len(evol.location),14))
        matrix[:,0] = evol.location
        matrix[:,1] = evol.charge
        matrix[:,2] = evol.energy
        matrix[:,3] = evol.x
        matrix[:,4] = evol.y
        matrix[:,5] = evol.rel_energy_spread
        matrix[:,6] = evol.rel_energy_spread_fwhm
        matrix[:,7] = evol.beam_size_x
        matrix[:,8] = evol.beam_size_y
        matrix[:,9] = evol.emit_nx
        matrix[:,10] = evol.emit_ny
        matrix[:,11] = evol.beta_x
        matrix[:,12] = evol.beta_y
        matrix[:,13] = evol.peak_spectral_density

        # save to CSV file
        filename = bunch + '_evolution.csv'
        np.savetxt(filename, matrix, delimiter=',')
        
    
    def plot_driver_evolution(self):
        self.plot_evolution(bunch='driver')
    
    def plot_evolution(self, bunch='beam'):
        
        # select bunch
        if bunch == 'beam':
            evol = copy.deepcopy(self.evolution.beam)
        elif bunch == 'driver':
            evol = copy.deepcopy(self.evolution.driver)
            
        # extract wakefield if not already existing
        if not hasattr(evol, 'location'):
            print('No evolution calculated')
            return

        # add upramp evolution
        if self.upramp is not None and hasattr(self.upramp.evolution.beam, 'location'):
            if bunch == 'beam':
                upramp_evol = self.upramp.evolution.beam
            elif bunch == 'driver':
                upramp_evol = self.upramp.evolution.driver
            evol.location = np.append(upramp_evol.location, evol.location-np.min(evol.location)+np.max(upramp_evol.location))
            evol.energy = np.append(upramp_evol.energy, evol.energy)
            evol.charge = np.append(upramp_evol.charge, evol.charge)
            evol.emit_nx = np.append(upramp_evol.emit_nx, evol.emit_nx)
            evol.emit_ny = np.append(upramp_evol.emit_ny, evol.emit_ny)
            evol.rel_energy_spread = np.append(upramp_evol.rel_energy_spread, evol.rel_energy_spread)
            evol.beam_size_x = np.append(upramp_evol.beam_size_x, evol.beam_size_x)
            evol.beam_size_y = np.append(upramp_evol.beam_size_y, evol.beam_size_y)
            evol.bunch_length = np.append(upramp_evol.bunch_length, evol.bunch_length)
            evol.x = np.append(upramp_evol.x, evol.x)
            evol.y = np.append(upramp_evol.y, evol.y)
            evol.z = np.append(upramp_evol.z, evol.z)
            evol.beta_x = np.append(upramp_evol.beta_x, evol.beta_x)
            evol.beta_y = np.append(upramp_evol.beta_y, evol.beta_y)
            evol.plasma_density = np.append(upramp_evol.plasma_density, evol.plasma_density)

        # add downramp evolution
        if self.downramp is not None and hasattr(self.downramp.evolution.beam, 'location'):
            if bunch == 'beam':
                downramp_evol = self.downramp.evolution.beam
            elif bunch == 'driver':
                downramp_evol = self.downramp.evolution.driver
            evol.location = np.append(evol.location, downramp_evol.location-np.min(downramp_evol.location)+np.max(evol.location))
            evol.energy = np.append(evol.energy, downramp_evol.energy)
            evol.charge = np.append(evol.charge, downramp_evol.charge)
            evol.emit_ny = np.append(evol.emit_ny, downramp_evol.emit_ny)
            evol.emit_nx = np.append(evol.emit_nx, downramp_evol.emit_nx)
            evol.rel_energy_spread = np.append(evol.rel_energy_spread, downramp_evol.rel_energy_spread)
            evol.beam_size_x = np.append(evol.beam_size_x, downramp_evol.beam_size_x)
            evol.beam_size_y = np.append(evol.beam_size_y, downramp_evol.beam_size_y)
            evol.bunch_length = np.append(evol.bunch_length, downramp_evol.bunch_length)
            evol.x = np.append(evol.x, downramp_evol.x)
            evol.y = np.append(evol.y, downramp_evol.y)
            evol.z = np.append(evol.z, downramp_evol.z)
            evol.beta_x = np.append(evol.beta_x, downramp_evol.beta_x)
            evol.beta_y = np.append(evol.beta_y, downramp_evol.beta_y)
            evol.plasma_density = np.append(evol.plasma_density, downramp_evol.plasma_density)
        
        # preprate plot
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
        axs[1,2].plot(evol.location, evol.beta_y*1e3, color=col2)  
        axs[1,2].plot(evol.location, evol.beta_x*1e3, color=col1)
        axs[1,2].set_ylabel('Beta function [mm]')
        axs[1,2].set_xlabel(long_label)
        axs[1,2].set_xlim(long_limits)
        axs[1,2].set_yscale('log')
        
        # plot longitudinal offset
        axs[2,0].plot(evol.location, evol.plasma_density / 1e6, color=col1)
        axs[2,0].set_ylabel(r'Plasma density [$\mathrm{cm}^{-3}$]')
        axs[2,0].set_xlabel(long_label)
        axs[2,0].set_xlim(long_limits)
        axs[2,0].set_yscale('log')
        
        # plot longitudinal offset
        axs[2,1].plot(evol.location, evol.z * 1e6, color=col1)
        axs[2,1].set_ylabel(r'Longitudinal offset [$\mathrm{\mu}$m]')
        axs[2,1].set_xlabel(long_label)
        axs[2,1].set_xlim(long_limits)
        
        # plot transverse offset
        axs[2,2].plot(evol.location, np.zeros(evol.location.shape), ':', color=col0)
        axs[2,2].plot(evol.location, evol.y*1e6, color=col2)  
        axs[2,2].plot(evol.location, evol.x*1e6, color=col1)
        axs[2,2].set_ylabel(r'Transverse offset [$\mathrm{\mu}$m]')
        axs[2,2].set_xlabel(long_label)
        axs[2,2].set_xlim(long_limits)
        
        plt.show()

        
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
        z_mid = zs_I.min() + (zs_I.max()-zs_I.min())*0.3
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
        axs[0].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
        axs[0].set_ylabel('Longitudinal electric field [GV/m]')
        zlims = [min(zs0)*1e6, max(zs0)*1e6]
        axs[0].set_xlim(zlims)
        axs[0].set_ylim(bottom=-1.7*np.max([np.abs(Ez_beam), Ez_driver])/1e9, top=1.3*Ez_driver/1e9)
        
        # plot beam current
        axs[1].fill(np.concatenate((zs_I, np.flip(zs_I)))*1e6, np.concatenate((-Is, np.zeros(Is.shape)))/1e3, color=col1, alpha=af)
        axs[1].plot(zs_I*1e6, -Is/1e3, '-', color=col1)
        axs[1].set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
        axs[1].set_ylabel('Beam current [kA]')
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
                z_mid = zs_I.max()-(zs_I.max()-zs_I.min())*0.3
                z_beam = zs_I[np.abs(Is[zs_I < z_mid]).argmax()]
                Ez_driver = Ezs0[zs0 > z_mid].max()
                Ez_beam = np.interp(z_beam, zs0, Ezs0)
                Ezmax = 1.7*np.max([np.abs(Ez_driver), np.abs(Ez_beam)])
            
            # plot on-axis wakefield and axes
            ax2 = ax1.twinx()
            ax2.plot(zs0*1e6, Ezs0/1e9, color = 'black')
            ax2.set_ylabel(r'$E_{z}$' ' [GV/m]')
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
            cb_ions.set_label(label=r'Beam/plasma-electron/ion density [$\mathrm{cm^{-3}}$]', size=10)
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
                ax1.set_xlabel(r'$z$ [$\mathrm{\mu}$m]')
            ax1.set_ylabel(r'$x$ [$\mathrm{\mu}$m]')
            ax1.set_title(title)
            ax1.grid(False)
            ax2.grid(False)
            
        # save the figure
        if savefig is not None:
            fig.savefig(str(savefig), bbox_inches='tight', dpi=1000)
        
        return 

    
    def survey_object(self):
        npoints = 10
        x_points = np.linspace(0, self.get_length(), npoints)
        y_points = np.linspace(0, 0, npoints)
        final_angle = 0 
        label = 'Plasma stage'
        color = 'red'
        return x_points, y_points, final_angle, label, color
        
class VariablesOverspecifiedError(Exception):
    "Exception class to throw when trying to set too many overlapping variables"
    pass
class StageError(Exception):
    "Exception class for Stege to throw in other cases"

