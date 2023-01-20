from opal import Stage
from opal.physicsmodels.wakeGolovanov import wakefield_Golovanov
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import root, root_scalar
from opal.utilities.plasmaphysics import *
from opal.utilities import SI

class StageDrivenNonlinear1D(Stage):
    
    def __init__(self, deltaE=None, L=None, n0=None, kRb=None, sigt_jitter=0, enableBetatron=False):
        self.deltaE = deltaE
        self.L = L
        self.n0 = n0
        self.enableBetatron = enableBetatron
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # calculate energy gain for each particle
        EzFcn, rFcn = self.wakefieldFcn(beam)

        # calculate energy change (zero for particles outside the wake)
        deltaEs = np.where(beam.rs() > rFcn(beam.zs()), 0, np.sign(beam.qs()) * EzFcn(beam.zs()) * self.L)
        
        # perform betatron motion
        if self.enableBetatron:
            beam.betatronMotion(self.L, self.n0, deltaEs)
        else:
            beam.betatronDamping(deltaEs)
            beam.flipTransversePhaseSpaces()
        
        # add energy gain
        beam.accelerate(deltaEs)
        
        return super().track(beam, savedepth, runnable, verbose)
    
    
    # wakefield (Lu equation)
    def wakefield(self, beam=None):
        Ezs, zs, rs = wakefield_Golovanov(self.n0, beam)
        return Ezs, zs, rs
    
    
    # wakefield function (Lu equation)
    def wakefieldFcn(self, beam=None):
        Ezs, zs, rbs = self.wakefield(beam)
        nanmask = ~np.isnan(zs * rbs * Ezs)
        EzFcn = lambda z: np.interp(z, zs[nanmask], Ezs[nanmask], right=0, left=np.nan)
        rFcn = lambda z: np.interp(z, zs[nanmask], rbs[nanmask], left=0)
        return EzFcn, rFcn
    
    
    def plotWakefield(self, beam):
        
        # get wakefield
        Ezs, zs, rs = self.wakefield(beam)
        
        # get current profile
        if beam is not None:
            Is, ts = beam.currentProfile()
            zs0 = ts*SI.c
        else:
            zs0 = zs
            Is = np.zeros(len(zs))
        
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