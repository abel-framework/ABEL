from opal.physicsmodels.plasmawake1D import wakefield1D
from opal import Stage
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import root, root_scalar
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
            
        self.driverInitial = None
        self.driverFinal = None
        
        self.driverToWakeEfficiency = None
        self.wakeToBeamEfficiency = None
        self.driverToBeamEfficiency = None
        
        self.reljitter = SimpleNamespace()
        self.reljitter.n0 = 0
        
        self.__n = None
        
    
    def __getInitialDriver(self, resample=False):
        if resample or self.driverInitial is None:
            self.driverInitial = self.driverSource.track()
        return self.driverInitial
    
    def __getDensity(self, resample=False):
        if resample or self.__n is None:
            self.__n = self.n0 * np.random.normal(loc = 1, scale = self.reljitter.n0)
        return self.__n
        
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # get driver
        driver0 = self.__getInitialDriver(resample=True)
        
        # initial beam energy and charge
        Etot0_beam = beam.totalEnergy()
        
        # sample the density (with jitter)
        n0 = self.__getDensity(resample=True)
        
        # calculate wakefield function
        EzFcn, rFcn = self.__wakefieldFcn(beam, driver=driver0, density=n0)

        # remove particles beyond the wake radius
        beam.filterPhaseSpace(beam.rs() > rFcn(beam.zs()))
        
        # calculate energy change (zero for particles outside the wake)
        deltaEs = np.sign(beam.qs()) * EzFcn(beam.zs()) * self.L
        
        # perform betatron motion
        if self.enableBetatron:
            x0_driver = np.random.normal(scale=self.driverSource.jitter.x0)
            y0_driver = np.random.normal(scale=self.driverSource.jitter.y0)
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
    
    
    def plotWakefield(self, beam=None):
        
        # get wakefield
        Ezs, zs, rs = self.__wakefield(beam)
        
        # get current profile
        driver = deepcopy(self.__getInitialDriver())
        if beam is not None:
            driver.addBeam(beam)
        Is, ts = driver.currentProfile(bins=np.linspace(min(zs/SI.c), max(zs/SI.c), int(np.sqrt(driver.Npart()))))
        zs0 = ts*SI.c
        
        # plot it
        fig, axs = plt.subplots(1,3)
        fig.set_figwidth(20)
        fig.set_figheight(4)
        zlims = [min(zs)*1e6, max(zs)*1e6]
        
        axs[0].plot(zs0*1e6, -Is/1e3, '-')
        axs[0].set_xlabel('z (um)')
        axs[0].set_ylabel('Beam current (kA)')
        axs[0].set_xlim(zlims)
        
        axs[1].plot(zs*1e6, rs*1e6, '-')
        axs[1].set_xlabel('z (um)')
        axs[1].set_ylabel('Plasma-wake radius (um)')
        axs[1].set_xlim(zlims)
        axs[1].set_ylim(bottom=0)
        
        axs[2].plot(zs*1e6, Ezs/1e9, '-')
        axs[2].set_xlabel('z (um)')
        axs[2].set_ylabel('Electric field (GV/m)')
        axs[2].set_xlim(zlims)
        axs[2].set_ylim(bottom=-Ez_wavebreaking(self.n0)/1e9, top=Ez_wavebreaking(self.n0)/1e9)
        
        
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
    