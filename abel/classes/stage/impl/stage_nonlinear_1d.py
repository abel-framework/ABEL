from abel import Stage, CONFIG
from matplotlib import pyplot as plt
import numpy as np
import scipy.constants as SI
import warnings, copy
from types import SimpleNamespace
from abel.utilities.plasma_physics import *
from abel.physics_models.plasma_wake_1d import wakefield_1d

class StageNonlinear1d(Stage):
    
    def __init__(self, length=None, nom_energy_gain=None, plasma_density=None, driver_source=None, ramp_beta_mag=1, enable_betatron=True, add_driver_to_beam=False):
        
        super().__init__(length, nom_energy_gain, plasma_density, driver_source, ramp_beta_mag)
        
        self.enable_betatron = enable_betatron
        self.add_driver_to_beam = add_driver_to_beam
        
        self.driver_to_wake_efficiency = None
        self.wake_to_beam_efficiency = None
        self.driver_to_beam_efficiency = None
        
        self.reljitter = SimpleNamespace()
        self.reljitter.plasma_density = 0
        
        # internally sampled values (given some jitter)
        self.__n = None
        self.driver_initial = None
        
    
    def __get_initial_driver(self, resample=False):
        if resample or self.driver_initial is None:
            self.driver_initial = self.driver_source.track()
        return self.driver_initial
    
    def __get_plasma_density(self, resample=False):
        if resample or self.__n is None:
            self.__n = self.plasma_density * np.random.normal(loc = 1, scale = self.reljitter.plasma_density)
        return self.__n
    
    
    # track the particles through
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # get driver
        driver0 = self.__get_initial_driver(resample=True)
        
        # initial beam energy and charge
        Etot0_beam = beam.total_energy()
        
        # sample the density (with jitter)
        plasma_density = self.__get_plasma_density(resample=True)
        
        # apply plasma-density down ramp (demagnify beta function)
        if self.ramp_beta_mag is not None:
            beam.magnify_beta_function(1/self.ramp_beta_mag)
            driver0.magnify_beta_function(1/self.ramp_beta_mag)
        
        # calculate wakefield function
        EzFcn, rFcn = self.__wakefieldFcn(beam, driver=driver0, density=plasma_density)

        # remove particles beyond the wake radius
        del beam[beam.rs() > rFcn(beam.zs())]
        
        # calculate energy change (zero for particles outside the wake)
        deltaEs = np.sign(beam.qs()) * EzFcn(beam.zs()) * self.length
        
        # find driver offset (to shift the beam relative) and apply betatron motion
        x0_driver = np.random.normal(scale=self.driver_source.jitter.x)
        y0_driver = np.random.normal(scale=self.driver_source.jitter.y)
        beam.apply_betatron_motion(self.length, plasma_density, deltaEs, x0_driver=x0_driver, y0_driver=y0_driver)
        
        # add energy gain
        beam.accelerate(deltaEs)
        
        # remove particles with nans
        beam.remove_nans()
        
        # simulate the driver
        driver = copy.deepcopy(driver0)
        deltaEs_driver = np.where(driver.rs() > rFcn(driver.zs()), 0, np.sign(driver.qs()) * EzFcn(driver.zs()) * self.length)
        depleted_frac = np.sum(driver.Es() + deltaEs_driver < 0)/len(driver)
        if depleted_frac > 0:
            print(f"WARNING: {depleted_frac*100:.1f}% of driver particles were energy depleted.")
        driver.apply_betatron_damping(deltaEs_driver)
        driver.flip_transverse_phase_spaces()
        driver.accelerate(deltaEs_driver)
        driver.remove_nans()
        
        # calculate efficiency
        Etot_beam = beam.total_energy()
        Etot0_driver = driver0.total_energy()
        Etot_driver = driver.total_energy()
        self.driver_to_wake_efficiency = (Etot0_driver-Etot_driver)/Etot0_driver
        self.wake_to_beam_efficiency = (Etot_beam-Etot0_beam)/(Etot0_driver-Etot_driver)
        self.driver_to_beam_efficiency = self.driver_to_wake_efficiency*self.wake_to_beam_efficiency
        
        # apply plasma-density up ramp (magnify beta function)
        if self.ramp_beta_mag is not None:
            beam.magnify_beta_function(self.ramp_beta_mag)
            driver.magnify_beta_function(self.ramp_beta_mag)
           
        # add the driver to the beam (if desired)
        if self.add_driver_to_beam:
            beam += driver
         
        return super().track(beam, savedepth, runnable, verbose)
    
    
    # wakefield (Lu equation)
    def __wakefield(self, beam=None, driver=None, density=None):
        
        # get density
        if density is None:
            density = self.__get_plasma_density()
            
        # get driver
        if driver is None:
            driver = self.__get_initial_driver()
        
        # try several times in case of solver issues (new driver every time)
        Ntries = 5
        for n in range(Ntries):
            #try:
            Ezs, zs, rs = wakefield_1d(density, driver, beam)
            #    break
            #except:
            #    driver = self.__get_initial_driver(resample=True)
            #    print(f">> Recalculating wakefield with new driver, problem with ODE solver (attempt #{n+1})")
          
        return Ezs, zs, rs
    
    
    # wakefield function (Lu equation)
    def __wakefieldFcn(self, beam=None, driver=None, density=None):
        Ezs, zs, rbs = self.__wakefield(beam, driver, density)
        nanmask = ~np.isnan(zs * rbs * Ezs)
        EzFcn = lambda z: np.interp(z, zs[nanmask], Ezs[nanmask], right=0, left=np.nan)
        rFcn = lambda z: np.interp(z, zs[nanmask], rbs[nanmask], left=0)
        return EzFcn, rFcn
    
    
    def plot_wakefield(self, beam=None, save_to_file=None, include_wake_radius=True):
        
        # get wakefield
        Ezs, zs, rs = self.__wakefield(beam)
        
        # get current profile
        driver = copy.deepcopy(self.__get_initial_driver())
        driver += beam
        Is, ts = driver.current_profile(bins=np.linspace(min(zs/SI.c), max(zs/SI.c), int(np.sqrt(len(driver))/2)))
        zs0 = ts*SI.c
        
        # plot it
        if include_wake_radius:
            fig, axs = plt.subplots(1, 3)
            fig.set_figwidth(CONFIG.plot_fullwidth_default)
            fig.set_figheight(4)
        else:
            fig, axs = plt.subplots(2,1)
            fig.set_figwidth(CONFIG.plot_fullwidth_default/3)
            fig.set_figheight(8)
        col0 = "xkcd:light gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        af = 0.1
        zlims = [min(zs)*1e6, max(zs)*1e6]
        
        axs[0].plot(zs*1e6, np.zeros(zs.shape), '-', color=col0)
        if self.nom_energy_gain is not None:
            axs[0].plot(zs*1e6, -self.nom_energy_gain/self.get_length()*np.ones(zs.shape)/1e9, ':', color=col2)
        axs[0].plot(zs*1e6, Ezs/1e9, '-', color=col1)
        axs[0].set_xlabel('z (um)')
        axs[0].set_ylabel('Longitudinal electric field (GV/m)')
        axs[0].set_xlim(zlims)
        axs[0].set_ylim(bottom=-max(wave_breaking_field(self.plasma_density), 1.5*max(Ezs))/1e9, top=1.3*max(Ezs)/1e9)
        
        axs[1].fill(np.concatenate((zs0, np.flip(zs0)))*1e6, np.concatenate((-Is, np.zeros(Is.shape)))/1e3, color=col1, alpha=af)
        axs[1].plot(zs0*1e6, -Is/1e3, '-', color=col1)
        axs[1].set_xlabel('z (um)')
        axs[1].set_ylabel('Beam current (kA)')
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=0, top=1.2*max(-Is)/1e3)
        
        if include_wake_radius:
            axs[2].fill(np.concatenate((zs, np.flip(zs)))*1e6, np.concatenate((rs, np.ones(zs.shape)))*1e6, color=col2, alpha=af)
            axs[2].plot(zs*1e6, rs*1e6, '-', color=col2)
            axs[2].set_xlabel('z (um)')
            axs[2].set_ylabel('Plasma-wake radius (um)')
            axs[2].set_xlim(zlims)
            axs[2].set_ylim(bottom=0, top=max(rs*1.2)*1e6)
        
        # save to file
        if save_to_file is not None:
            plt.savefig(save_to_file, format="pdf", bbox_inches="tight")
        
        