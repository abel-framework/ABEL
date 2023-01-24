from opal.physicsmodels.plasmawake1D import wakefield1D
from opal import Stage
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import root, root_scalar
from opal.utilities.plasmaphysics import *
from opal.utilities import SI

class StageNonlinear1D(Stage):
    
    def __init__(self, deltaE=None, L=None, n0=None, kRb=None, driverSource=None, enableBetatron=True, addDriverToBeam=False):
        self.deltaE = deltaE
        self.L = L
        self.n0 = n0
        self.enableBetatron = enableBetatron
        self.addDriverToBeam = addDriverToBeam
        self.driverSource = driverSource
        self.driverInitial = None
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # calculate wakefield function
        EzFcn, rFcn = self.wakefieldFcn(beam)

        # remove particles beyond the wake radius
        beam.filterPhaseSpace(beam.rs() > rFcn(beam.zs()))
        
        # calculate energy change (zero for particles outside the wake)
        deltaEs = np.sign(beam.qs()) * EzFcn(beam.zs()) * self.L
        
        # perform betatron motion
        if self.enableBetatron:
            beam.betatronMotion(self.L, self.n0, deltaEs)
        else:
            beam.betatronDamping(deltaEs) # TODO: seems to not work at damping beta function
            beam.flipTransversePhaseSpaces()
        
        # add energy gain
        beam.accelerate(deltaEs)
        
        # remove particles with nan energies
        beam.filterPhaseSpace(np.isnan(beam.Es()))
        
        # simulate and add driver to the beam (if desired)
        if self.addDriverToBeam and self.driverSource is not None:
            driver = self.driverSource.track()
            deltaEs_driver = np.where(driver.rs() > rFcn(driver.zs()), 0, np.sign(driver.qs()) * EzFcn(driver.zs()) * self.L)
            driver.betatronDamping(deltaEs_driver)
            driver.flipTransversePhaseSpaces()
            driver.accelerate(deltaEs_driver)
            driver.filterPhaseSpace(np.isnan(beam.Es()))
            beam.addBeam(driver)
            
        return super().track(beam, savedepth, runnable, verbose)
    
    
    # wakefield (Lu equation)
    def wakefield(self, beam=None):
        
        # if driver has not been generated, do so
        if self.driverInitial is None:
            self.driverInitial = self.driverSource.track()
        
        # try several times in case of solver issues (new driver every time)
        Ntries = 5
        for n in range(Ntries):
            try:
                Ezs, zs, rs = wakefield1D(self.n0, self.driverInitial, beam)
                break
            except:
                self.driverInitial = self.driverSource.track()
                print(f">> Recalculating wakefield with new driver, problem with ODE solver (attempt #{n+1})")
          
        return Ezs, zs, rs
    
    
    # wakefield function (Lu equation)
    def wakefieldFcn(self, beam=None):
        Ezs, zs, rbs = self.wakefield(beam)
        nanmask = ~np.isnan(zs * rbs * Ezs)
        EzFcn = lambda z: np.interp(z, zs[nanmask], Ezs[nanmask], right=0, left=np.nan)
        rFcn = lambda z: np.interp(z, zs[nanmask], rbs[nanmask], left=0)
        return EzFcn, rFcn
    
    
    def plotWakefield(self, beam=None):
        
        # get wakefield
        Ezs, zs, rs = self.wakefield(beam)
        
        # get current profile
        driver = self.driverSource.track()
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