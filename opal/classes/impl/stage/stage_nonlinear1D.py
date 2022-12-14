from opal import Stage
from opal.physicsmodels.wakeLu import wakefield_Lu
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import root, root_scalar
from opal.utilities.plasmaphysics import *
from opal.utilities import SI

class StageNonlinear1D(Stage):
    
    def __init__(self, deltaE=None, L=None, n0=None, kRb=1, sigt_jitter=0):
        self.deltaE = deltaE
        self.L = L
        self.n0 = n0
        self.kRb = kRb
        self.sigt_jitter = sigt_jitter
        self.enableBetatron = True
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # calculate energy gain for each particle
        EzFcn, rFcn = self.wakefieldFcn(beam)
        
        # remove particles beyond the wake radius
        beam.filterPhaseSpace(beam.rs() > rFcn(beam.zs()))
        
        # calculate energy change
        deltaEs = np.sign(beam.qs()) * EzFcn(beam.zs()) * self.L
        
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
        
        # get wakefield
        Ezs, zs, rs = wakefield_Lu(self.n0, self.kRb, beam)
        
        # add timing jitter
        dz = np.random.normal() * self.sigt_jitter * SI.c
        zs = zs + dz
        
        return Ezs, zs, rs
    
    # wakefield function (Lu equation)
    def wakefieldFcn(self, beam=None):
        Ezs, zs, rs = self.wakefield(beam)
        EzFcn = lambda z: np.interp(z, zs, Ezs)
        rFcn = lambda z: np.interp(z, zs, rs)
        return EzFcn, rFcn
    
    def plotWakefield(self, beam=None):
        
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
        axs[1].set_ylim(0, self.kRb/k_p(self.n0)*1.1e6)
        
        axs[2].plot(zs*1e6, Ezs/1e9, '-')
        axs[2].set_xlabel('z (um)')
        axs[2].set_ylabel('Electric field (GV/m)')
        axs[2].set_xlim(zlims)
        axs[2].set_ylim(-Ez_wavebreaking(self.n0)*self.kRb/1e9,0)
        
    def optimalInjectionPosition(self, source):
        
        # target field strength
        Ez_target = -self.deltaE/self.L
        
        # initial guess (no beam loading)
        z_guess = -1/k_p(self.n0)
        Ez0 = self.wakefieldFcn()
        Ez, zs, _ = self.wakefield()
        sol0 = root(lambda z: Ez0(z)-Ez_target, z_guess)
        
        # define optimizer function
        def fcn(z):
            source.z = z
            EzFcn = self.wakefieldFcn(source.track())
            return EzFcn(z)-Ez_target
        
        # find optimum
        sol = root_scalar(fcn, x0=sol0.x[0], bracket=[min(zs), max(zs)])
        z_opt = sol.root
        
        return z_opt
        
        
    def length(self):
        return self.L
    
    def energyGain(self):
        return self.deltaE