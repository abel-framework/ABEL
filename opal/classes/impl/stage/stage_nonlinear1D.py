from opal.physicsmodels.plasmawake1D import wakefield1D
from opal import Stage, CONFIG
from matplotlib import pyplot as plt
import numpy as np
from opal.utilities.plasmaphysics import *
from opal.utilities import SI
from copy import deepcopy
import warnings
from types import SimpleNamespace

class StageNonlinear1D(Stage):
    
    def __init__(self, deltaE=None, L=None, n0=None, driverSource=None, enableBetatron=True, addDriverToBeam=False):
        self.deltaE = deltaE
        self.L = L
        self.n0 = n0
        self.enableBetatron = enableBetatron
        self.addDriverToBeam = addDriverToBeam
        self.driverSource = driverSource
        
        self.driverToWakeEfficiency = None
        self.wakeToBeamEfficiency = None
        self.driverToBeamEfficiency = None
        
        self.reljitter = SimpleNamespace()
        self.reljitter.n0 = 0
        
        self.rampBetaMagnification = 1
        
        # implement radiation reaction in the betatron motion
        self.radiationReaction = False
        
        # internally sampled values (given some jitter)
        self.__n = None
        self.driverInitial = None
        
    
    def __getInitialDriver(self, resample=False):
        if resample or self.driverInitial is None:
            self.driverInitial = self.driverSource.track()
        return self.driverInitial
    
    def __getDensity(self, resample=False):
        if resample or self.__n is None:
            self.__n = self.n0 * np.random.normal(loc = 1, scale = self.reljitter.n0)
        return self.__n
    
    
    # matched beta function of the stage (for a given energy)
    def matchedBetaFunction(self, E):
        return beta_matched(self.n0, E) * self.rampBetaMagnification
    
    # track the particles through
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # get driver
        driver0 = self.__getInitialDriver(resample=True)
        
        # initial beam energy and charge
        Etot0_beam = beam.totalEnergy()
        
        # sample the density (with jitter)
        n0 = self.__getDensity(resample=True)
        
        # apply plasma-density down ramp (demagnify beta function)
        if self.rampBetaMagnification is not None:
            beam.magnifyBetaFunction(1/self.rampBetaMagnification)
            driver0.magnifyBetaFunction(1/self.rampBetaMagnification)
        
        # calculate wakefield function
        EzFcn, rFcn = self.__wakefieldFcn(beam, driver=driver0, density=n0)

        # remove particles beyond the wake radius
        beam.filterPhaseSpace(beam.rs() > rFcn(beam.zs()))
        
        # calculate energy change (zero for particles outside the wake)
        deltaEs = np.sign(beam.qs()) * EzFcn(beam.zs()) * self.L
        
        # perform betatron motion
        if self.enableBetatron:
            
            # find driver offset (to shift the beam relative)
            x0_driver = np.random.normal(scale=self.driverSource.jitter.x0)
            y0_driver = np.random.normal(scale=self.driverSource.jitter.y0)
            
            # calculate betatron motion (with or without radiation reaction)
            if self.radiationReaction:
                beam.betatronMotionRadiative(self.L, n0, deltaEs, x0_driver=x0_driver, y0_driver=y0_driver) # TODO: (Daniel)
            else:
                beam.betatronMotion(self.L, n0, deltaEs, x0_driver=x0_driver, y0_driver=y0_driver)
                
        else:
            beam.betatronDamping(deltaEs) # TODO: seems to not work at damping beta function
            beam.flipTransversePhaseSpaces()
        
        # add energy gain
        beam.accelerate(deltaEs)
        
        # remove particles with nan energies
        beam.filterPhaseSpace(np.isnan(beam.Es()))
        
        # simulate the driver
        driver = deepcopy(driver0)
        deltaEs_driver = np.where(driver.rs() > rFcn(driver.zs()), 0, np.sign(driver.qs()) * EzFcn(driver.zs()) * self.L)
        depleted_frac = np.sum(driver.Es() + deltaEs_driver < 0)/driver.Npart()
        if depleted_frac > 0:
            print(f"WARNING: {depleted_frac*100:.1f}% of driver particles were energy depleted.")
        driver.betatronDamping(deltaEs_driver)
        driver.flipTransversePhaseSpaces()
        driver.accelerate(deltaEs_driver)
        driver.filterPhaseSpace(np.isnan(beam.Es()))
        
        # calculate efficiency
        Etot_beam = beam.totalEnergy()
        Etot0_driver = driver0.totalEnergy()
        Etot_driver = driver.totalEnergy()
        self.driverToWakeEfficiency = (Etot0_driver-Etot_driver)/Etot0_driver
        self.wakeToBeamEfficiency = (Etot_beam-Etot0_beam)/(Etot0_driver-Etot_driver)
        self.driverToBeamEfficiency = self.driverToWakeEfficiency*self.wakeToBeamEfficiency
        
        # apply plasma-density up ramp (magnify beta function)
        if self.rampBetaMagnification is not None:
            beam.magnifyBetaFunction(self.rampBetaMagnification)
            driver.magnifyBetaFunction(self.rampBetaMagnification)
           
        # add the driver to the beam (if desired)
        if self.addDriverToBeam:
            beam.addBeam(driver)
         
        return super().track(beam, savedepth, runnable, verbose)
    
    
    # wakefield (Lu equation)
    def __wakefield(self, beam=None, driver=None, density=None):
        
        # get density
        if density is None:
            density = self.__getDensity()
            
        # get driver
        if driver is None:
            driver = self.__getInitialDriver()
        
        # try several times in case of solver issues (new driver every time)
        Ntries = 5
        for n in range(Ntries):
            try:
                Ezs, zs, rs = wakefield1D(density, driver, beam)
                break
            except:
                driver = self.__getInitialDriver(resample=True)
                print(f">> Recalculating wakefield with new driver, problem with ODE solver (attempt #{n+1})")
          
        return Ezs, zs, rs
    
    
    # wakefield function (Lu equation)
    def __wakefieldFcn(self, beam=None, driver=None, density=None):
        Ezs, zs, rbs = self.__wakefield(beam, driver, density)
        nanmask = ~np.isnan(zs * rbs * Ezs)
        EzFcn = lambda z: np.interp(z, zs[nanmask], Ezs[nanmask], right=0, left=np.nan)
        rFcn = lambda z: np.interp(z, zs[nanmask], rbs[nanmask], left=0)
        return EzFcn, rFcn
    
    
    def plotWakefield(self, beam=None, saveToFile=None, includeWakeRadius=True):
        
        # get wakefield
        Ezs, zs, rs = self.__wakefield(beam)
        
        # get current profile
        driver = deepcopy(self.__getInitialDriver())
        if beam is not None:
            driver.addBeam(beam)
        Is, ts = driver.currentProfile(bins=np.linspace(min(zs/SI.c), max(zs/SI.c), int(np.sqrt(driver.Npart())/2)))
        zs0 = ts*SI.c
        
        # plot it
        fig, axs = plt.subplots(1, 2+int(includeWakeRadius))
        fig.set_figwidth(CONFIG.plot_fullwidth_default*(2+int(includeWakeRadius))/3)
        fig.set_figheight(4)
        col0 = "xkcd:light gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        af = 0.1
        zlims = [min(zs)*1e6, max(zs)*1e6]
        
        axs[0].plot(zs*1e6, np.zeros(zs.shape), '-', color=col0)
        if self.deltaE is not None:
            axs[0].plot(zs*1e6, -self.deltaE/self.L*np.ones(zs.shape)/1e9, ':', color=col0)
        axs[0].plot(zs*1e6, Ezs/1e9, '-', color=col1)
        axs[0].set_xlabel('z (um)')
        axs[0].set_ylabel('Longitudinal electric field (GV/m)')
        axs[0].set_xlim(zlims)
        axs[0].set_ylim(bottom=-Ez_wavebreaking(self.n0)/1e9, top=1.3*max(Ezs)/1e9)
        
        axs[1].fill(np.concatenate((zs0, np.flip(zs0)))*1e6, np.concatenate((-Is, np.zeros(Is.shape)))/1e3, color=col1, alpha=af)
        axs[1].plot(zs0*1e6, -Is/1e3, '-', color=col1)
        axs[1].set_xlabel('z (um)')
        axs[1].set_ylabel('Beam current (kA)')
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=0, top=1.2*max(-Is)/1e3)
        
        if includeWakeRadius:
            axs[2].fill(np.concatenate((zs, np.flip(zs)))*1e6, np.concatenate((rs, np.ones(zs.shape)))*1e6, color=col2, alpha=af)
            axs[2].plot(zs*1e6, rs*1e6, '-', color=col2)
            axs[2].set_xlabel('z (um)')
            axs[2].set_ylabel('Plasma-wake radius (um)')
            axs[2].set_xlim(zlims)
            axs[2].set_ylim(bottom=0, top=max(rs*1.2)*1e6)

        # save to file
        if saveToFile is not None:
            plt.savefig(saveToFile, format="pdf", bbox_inches="tight")
        
        
        
    def length(self):
        return self.L
    
    def energyGain(self):
        return self.deltaE
    
    def energyEfficiency(self):
        return self.driverToBeamEfficiency, self.driverToWakeEfficiency, self.wakeToBeamEfficiency
    
    def wallplugEfficiency(self):
        return self.driverToBeamEfficiency*self.driverSource.energyEfficiency()
    
    def energyUsage(self):
        return self.driverSource.energyUsage()
    