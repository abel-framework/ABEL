import numpy as np
from opal.utilities import SI
from opal.utilities.relativity import *
from opal.utilities.statistics import prct_clean, prct_clean2D
from opal.utilities.plasmaphysics import k_p
from opal.physicsmodels.betatronHills import evolveHillsEquation
from matplotlib import pyplot as plt

class Beam():
    
    # empty beam
    def __init__(self, phasespace = None, Npart = 1000):

        # the phase space variable is private
        if phasespace is not None:
            self.__phasespace = phasespace
        else:
            self.__phasespace = self.resetPhaseSpace(Npart)
            
        self.trackableNumber = 0
        self.stageNumber = 0
        self.location = 0        
    
    
    # reset phase space
    def resetPhaseSpace(self, Npart):
        self.__phasespace = np.zeros((8, Npart))
    
    
    # filter out macroparticles based on a mask (true means delete)
    def filterPhaseSpace(self, mask):
        if mask.any():
            self.__phasespace = np.delete(self.__phasespace, np.where(mask), 1)
            
        
    # set phase space
    def setPhaseSpace(self, Q, xs, ys, zs, wxs=None, wys=None, wzs=None, pxs=None, pys=None, pzs=None, xps=None, yps=None, Es=None):
        
        # make empty phase space
        Npart = len(xs)
        self.resetPhaseSpace(Npart)
        
        # add positions
        self.__setXs(xs)
        self.__setYs(ys)
        self.__setZs(zs)
        
        # add momenta
        if wzs is None:
            if pzs is not None:
                wzs = momentum2properVelocity(pzs)
            elif Es is not None:
                wzs = energy2properVelocity(Es)
        self.__phasespace[5,:] = wzs
        if wxs is None:
            if pxs is not None:
                wxs = momentum2properVelocity(pxs)
            elif xps is not None:
                wxs = xps * wzs
        self.__phasespace[3,:] = wxs
        if wys is None:
            if pys is not None:
                wys = momentum2properVelocity(pys)
            elif yps is not None:
                wys = yps * wzs
        self.__phasespace[4,:] = wys
        
        # charge
        self.__phasespace[6,:] = Q/Npart
            
        
    
    ## BEAM ARRAYS

    # number of macroparticles
    def Npart(self):
        return self.__phasespace.shape[1]
     
    # get phase space variables
    def xs(self):
        return self.__phasespace[0,:]
    def ys(self):
        return self.__phasespace[1,:]
    def zs(self):
        return self.__phasespace[2,:]
    def wxs(self):
        return self.__phasespace[3,:]
    def wys(self):
        return self.__phasespace[4,:]
    def wzs(self):
        return self.__phasespace[5,:]
    def qs(self):
        return self.__phasespace[6,:]
    
    # set phase space (private)
    def __setXs(self, xs):
        self.__phasespace[0,:] = xs
    def __setYs(self, ys):
        self.__phasespace[1,:] = ys
    def __setZs(self, zs):
        self.__phasespace[2,:] = zs
    def __setXps(self, xps):
        self.__setWxs(xps*self.wzs())
    def __setYps(self, yps):
        self.__setWys(yps*self.wzs())
        
    # set phase space (private)
    def __setWxs(self, wxs):
        self.__phasespace[3,:] = wxs
    def __setWys(self, wys):
        self.__phasespace[4,:] = wys
    def __setWzs(self, wzs):
        self.__phasespace[5,:] = wzs
        
    def __setQs(self, qs):
        self.__phasespace[6,:] = qs
    
    # copy another beam's macroparticle charge
    def copyParticleCharge(self, beam):
        self.__setQs(np.median(beam.qs()))
    
    
    def rs(self):
        return np.sqrt(self.xs()**2 + self.ys()**2)
    
    def pxs(self):
        return properVelocity2momentum(self.wxs())
    def pys(self):
        return properVelocity2momentum(self.wys())
    def pzs(self):
        return properVelocity2momentum(self.wzs())
    
    def xps(self):
        return self.wxs()/self.wzs()
    def yps(self):
        return self.wys()/self.wzs()

    def gammas(self):
        return properVelocity2gamma(self.wzs())
    def Es(self):
        return properVelocity2energy(self.wzs())
    def deltas(self, pz0=None):
        if pz0 is None:
            pz0 = np.mean(self.pzs())
        return self.pzs()/pz0 -1
    
    def ts(self):
        return self.zs()/SI.c
    
    def Ns(self):
        return self.qs()/SI.e
    
    # vector of transverse positions and angles: (x, x', y, y')
    def transverseVector(self):
        X = np.zeros((4,self.Npart()))
        X[0,:] = self.xs()
        X[1,:] = self.xps()
        X[2,:] = self.ys()
        X[3,:] = self.yps()
        return X
    
    # set phase space based on transverse vector: (x, x', y, y')
    def setTransverseVector(self, X):
        self.__setXs(X[0,:])
        self.__setXps(X[1,:])
        self.__setYs(X[2,:])
        self.__setYps(X[3,:])  
    
    
    ## BEAM STATISTICS
    
    def charge(self):
        return np.sum(self.qs())
    
    def absCharge(self):
        return abs(self.charge())
    
    def chargeSign(self):
        return self.charge()/self.absCharge()
    
    def energy(self, clean=True):
        return np.mean(prct_clean(self.Es(), clean))
    
    def gammaRelativistic(self, clean=True):
        return np.mean(prct_clean(self.gammas(), clean))
    
    def energySpread(self, clean=True):
        return np.std(prct_clean(self.Es(), clean))
    
    def relEnergySpread(self, clean=True):
        return self.energySpread(clean)/self.energy(clean)
    
    def offsetZ(self, clean=True):
        return np.mean(prct_clean(self.zs(), clean))
    
    def bunchLength(self, clean=True):
        return np.std(prct_clean(self.zs(), clean))
    
    def offsetX(self, clean=True):
        return np.mean(prct_clean(self.xs(), clean))
    
    def offsetY(self, clean=True):
        return np.mean(prct_clean(self.ys(), clean))
    
    def beamSizeX(self, clean=True):
        return np.std(prct_clean(self.xs(), clean))

    def beamSizeY(self, clean=True):
        return np.std(prct_clean(self.ys(), clean))
    
    def normEmittanceX(self, clean=True):
        xs, wxs = prct_clean2D(self.xs(), self.wxs(), clean)
        #return np.sqrt(np.linalg.det(np.cov(self.xs(), self.wxs()/SI.c)))
        return np.sqrt(np.linalg.det(np.cov(xs, wxs/SI.c)))
    
    def normEmittanceY(self, clean=True):
        ys, wys = prct_clean2D(self.ys(), self.wys(), clean)
        return np.sqrt(np.linalg.det(np.cov(ys, wys/SI.c)))
    
    def betaX(self, clean=True):
        xs, xps = prct_clean2D(self.xs(), self.xps(), clean)
        covx = np.cov(xs, xps)
        return covx[0,0]/np.sqrt(np.linalg.det(covx))
    
    def betaY(self, clean=True):
        ys, yps = prct_clean2D(self.ys(), self.yps(), clean)
        covy = np.cov(ys, yps)
        return covy[0,0]/np.sqrt(np.linalg.det(covy))
    
    
    ## BEAM PROJECTIONS
    
    def projectedDensity(self, fcn, bins=None):
        if bins is None:
            nsig = 5
            Nbins = int(np.sqrt(self.Npart())/2)
            bins = np.mean(fcn()) + nsig * np.std(fcn()) * np.arange(-1, 1, 2/Nbins)
        counts, edges = np.histogram(fcn(), weights=self.qs(), bins=bins)
        ctrs = (edges[0:-1] + edges[1:])/2
        proj = counts/np.diff(edges)
        return proj, ctrs
        
    def currentProfile(self, bins=None):
        return self.projectedDensity(self.ts, bins=bins)
        return dQdt, ts
    
    def longitudinalNumberDensity(self, bins=None):
        dQdz, zs = self.projectedDensity(self.zs, bins=bins)
        dNdz = dQdz / SI.e
        return dNdz, zs
    
    def energyProfile(self, bins=None):
        return self.projectedDensity(self.Es, bins=bins)
    
    def relEnergyProfile(self, E0=None, bins=None):
        if E0 is None:
            E0 = self.energy()
        return self.projectedDensity(lambda: self.Es()/E0-1, bins=bins)
    
    def transverseProfileX(self, bins=None):
        return self.projectedDensity(self.xs, bins=bins)
    
    def transverseProfileY(self, bins=None):
        return self.projectedDensity(self.ys, bins=bins)
    
    ## phase spaces
    
    def phaseSpaceDensity(self, hfcn, vfcn, hbins=None, vbins=None):
        nsig = 4
        Nbins = int(np.sqrt(self.Npart()))
        if hbins is None:
            hbins = np.mean(hfcn()) + nsig * np.std(hfcn()) * np.arange(-1, 1, 2/Nbins)
        if vbins is None:
            vbins = np.mean(vfcn()) + nsig * np.std(vfcn()) * np.arange(-1, 1, 2/Nbins)
        counts, hedges, vedges = np.histogram2d(hfcn(), vfcn(), weights=self.qs(), bins=(hbins, vbins))
        hctrs = (hedges[0:-1] + hedges[1:])/2
        vctrs = (vedges[0:-1] + vedges[1:])/2
        density = (counts/np.diff(vedges)).T/np.diff(hedges)
        return density, hctrs, vctrs
    
    def densityLPS(self, hbins=None, vbins=None):
        return self.phaseSpaceDensity(self.zs, self.Es, hbins=hbins, vbins=vbins)
    
    def densityTransverse(self, hbins=None, vbins=None):
        return self.phaseSpaceDensity(self.ys, self.xs, hbins=hbins, vbins=vbins)    
        
    
    ## PLOTTING
    def plotCurrentProfile(self):
        dQdt, ts = self.currentProfile()

        fig, ax = plt.subplots()
        fig.set_figwidth(6)
        fig.set_figheight(4)        
        ax.plot(ts*SI.c*1e6, -dQdt/1e3)
        ax.set_xlabel('z (um)')
        ax.set_ylabel('Beam current (kA)')
    
    def plotLPS(self):
        dQdzdE, zs, Es = self.densityLPS()

        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(5)  
        p = ax.pcolor(zs*1e6, Es/1e9, -dQdzdE*1e15, cmap='GnBu')
        ax.set_xlabel('z (um)')
        ax.set_ylabel('E (GeV)')
        ax.set_title('Longitudinal phase space')
        cb = fig.colorbar(p)
        cb.ax.set_ylabel('Charge density (pC/um/GeV)')
        
    
    ## CHANGE BEAM
    
    def accelerate(self, deltaE = 0, chirp = 0, z0 = 0):
        
        # add energy and chirp
        Es = self.Es() + deltaE 
        Es = Es + np.sign(self.qs()) * (self.zs()-z0) * chirp
        self.__setWzs(energy2properVelocity(Es))
        
        # remove particles with subzero energy
        self.filterPhaseSpace(Es < 0)
        self.filterPhaseSpace(np.isnan(Es))
        
    def compress(self, R56, E0):
        zs = self.zs() + (1-self.Es()/E0) * R56
        self.__setZs(zs)
        
    # betatron damping (must be done before acceleration)
    def betatronDamping(self, deltaE):
        gammasBoosted = energy2gamma(self.Es()+deltaE)
        factor = np.sqrt(self.gammas()/gammasBoosted)
        self.__setXs(self.xs() * factor)
        self.__setYs(self.ys() * factor)
        self.__setWxs(self.wxs() / factor)
        self.__setWys(self.wys() / factor)
        
    def flipTransversePhaseSpaces(self):
        self.__setWxs(-self.wxs())
        self.__setWys(-self.wys()) 
        
    def betatronMotion(self, L, n0, deltaEs):
        kps = lambda s: k_p(n0)
        xs, wxs, ys, wys = self.xs(), self.wxs(), self.ys(), self.wys()
        gamma0s = energy2gamma(self.Es())
        gammas = energy2gamma(self.Es()+deltaEs)
        for i in range(self.Npart()):
            gamma = lambda s: gamma0s[i] + (gammas[i]-gamma0s[i])*s/L
            xs[i], wxs[i], _ = evolveHillsEquation(xs[i], wxs[i], L, gamma, kps, fast=False)
            ys[i], wys[i], _ = evolveHillsEquation(ys[i], wys[i], L, gamma, kps, fast=False)
        self.__setXs(xs)
        self.__setWxs(wxs)
        self.__setYs(ys)
        self.__setWys(wys)
        
  
    ## SAVE AND LOAD BEAM
    
    # make filename for beam 
    def filename(self):
        return "beam_" + str(self.trackableNumber).zfill(3) + "_"  + str(self.stageNumber).zfill(3) + "_" + "{:012.6F}".format(self.location) + ".txt"
      
    # save beam
    def save(self, runnable):
        np.savetxt(runnable.runPath() + self.filename(), self.__phasespace)
    
    # load beam
    @classmethod
    def load(_, filename):
        
        # create beam from phase space
        beam = Beam(phasespace = np.loadtxt(filename))
            
        # find stage and location from filename
        parts0 = filename.split('/')
        parts = parts0[-1].split('_')
        beam.trackableNumber = int(parts[1])
        beam.stageNumber = int(parts[2])
        subparts = parts[3].split('.')
        beam.location = float(subparts[0] + '.' + subparts[1])
        
        return beam
      