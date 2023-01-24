import numpy as np
import openpmd_api as io
from datetime import datetime
from pytz import timezone
from opal.utilities import SI
from opal.utilities.relativity import *
from opal.utilities.statistics import prct_clean, prct_clean2D
from opal.utilities.plasmaphysics import k_p
from opal.physicsmodels.betatronHills import evolveHillsEquation_analytic
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
            
            # perform filtering and fix memory allocation
            self.__phasespace = np.ascontiguousarray(np.delete(self.__phasespace, np.where(mask), 1))
            
        
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
        
        # ids
        self.__phasespace[7,:] = np.arange(Npart)
            
    
    # add beam to phasespace (combining beams)
    def addBeam(self, beam):
        self.__phasespace = np.append(self.__phasespace, beam.__phasespace, axis=1)
    
    
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
    def ids(self):
        return self.__phasespace[7,:]
    
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
    def __setIds(self, ids):
        self.__phasespace[7,:] = ids
        
    def weightings(self):
        return self.__phasespace[6,:]/(self.chargeSign()*SI.e)
    
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
        qs = self.qs()
        return np.sum(qs[~np.isnan(qs)])
    
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
    
    def geomEmittanceX(self, clean=True):
        xs, xps = prct_clean2D(self.xs(), self.xps(), clean)
        return np.sqrt(np.linalg.det(np.cov(xs, xps)))
    
    def geomEmittanceY(self, clean=True):
        ys, yps = prct_clean2D(self.ys(), self.yps(), clean)
        return np.sqrt(np.linalg.det(np.cov(ys, yps)))
    
    def normEmittanceX(self, clean=True):
        xs, wxs = prct_clean2D(self.xs(), self.wxs(), clean)
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
    
    def alphaX(self, clean=True):
        xs, xps = prct_clean2D(self.xs(), self.xps(), clean)
        covx = np.cov(xs, xps)
        return -covx[1,0]/np.sqrt(np.linalg.det(covx))
    
    def alphaY(self, clean=True):
        ys, yps = prct_clean2D(self.ys(), self.yps(), clean)
        covy = np.cov(ys, yps)
        return -covy[1,0]/np.sqrt(np.linalg.det(covy))
    
    def gammaX(self, clean=True):
        xs, xps = prct_clean2D(self.xs(), self.xps(), clean)
        covx = np.cov(xs, xps)
        return covx[1,1]/np.sqrt(np.linalg.det(covx))
    
    def gammaY(self, clean=True):
        ys, yps = prct_clean2D(self.ys(), self.yps(), clean)
        covy = np.cov(ys, yps)
        return covy[1,1]/np.sqrt(np.linalg.det(covy))
    
    
    def peakDensity(self):
        return (self.charge()/SI.e)/(np.sqrt(2*np.pi)**3*self.beamSizeX()*self.beamSizeY()*self.bunchLength())
    
    
    ## BEAM PROJECTIONS
    
    def projectedDensity(self, fcn, bins=None):
        if bins is None:
            Nbins = int(np.sqrt(self.Npart()/2))
            bins = np.linspace(min(fcn()), max(fcn()), Nbins)
        counts, edges = np.histogram(fcn(), weights=self.qs(), bins=bins)
        ctrs = (edges[0:-1] + edges[1:])/2
        proj = counts/np.diff(edges)
        return proj, ctrs
        
    def currentProfile(self, bins=None):
        return self.projectedDensity(self.ts, bins=bins)
    
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
        if hbins is None:
            hbins = int(np.sqrt(self.Npart())/2)
        if vbins is None:
            vbins = int(np.sqrt(self.Npart())/2)
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
        xs, wxs, ys, wys = self.xs(), self.wxs(), self.ys(), self.wys()
        gamma0s = energy2gamma(self.Es())
        gammas = energy2gamma(self.Es()+deltaEs)
        dgamma_ds = (gammas-gamma0s)/L
        xs, wxs = evolveHillsEquation_analytic(self.xs(), self.wxs(), L, gamma0s, dgamma_ds, k_p(n0))
        ys, wys = evolveHillsEquation_analytic(self.ys(), self.wys(), L, gamma0s, dgamma_ds, k_p(n0))
        self.__setXs(xs)
        self.__setWxs(wxs)
        self.__setYs(ys)
        self.__setWys(wys)
        
  
    ## SAVE AND LOAD BEAM
    
    def filename(self, runnable):
        return runnable.shotPath() + "/beam_{:012.6F}".format(self.location) + ".h5"
    
    
    # save beam (to OpenPMD format)
    def save(self, runnable=None, beamName="beam", series=None):
        
        if self.Npart() == 0:
            return
        
        # make new file if not provided
        if series is None:
            
            # open a new file
            series = io.Series(self.filename(runnable), io.Access.create)
        
            # add metadata
            series.author = "OPAL (the Optimizable Plasma-Accelerator Linac code)"
            series.date = datetime.now(timezone('CET')).strftime('%Y-%m-%d %H:%M:%S %z')

        # make step (only one)
        iteration = series.iterations[0]
        
        # add attributes
        iteration.set_attribute("time", self.location/SI.c)
        for key, value in self.__dict__.items():
            if not "__phasespace" in key:
                iteration.set_attribute(key, value)
        
        # make beam record
        b = iteration.particles[beamName]
        
        # declare dataset structure
        dset = io.Dataset(self.ids().dtype, self.ids().shape)
        b["id"][io.Record_Component.SCALAR].reset_dataset(dset)
        b["weighting"][io.Record_Component.SCALAR].reset_dataset(dset)
        b["position"]["z"].reset_dataset(dset)
        b["position"]["x"].reset_dataset(dset)
        b["position"]["y"].reset_dataset(dset)
        b["momentum"]["z"].reset_dataset(dset)
        b["momentum"]["x"].reset_dataset(dset)
        b["momentum"]["y"].reset_dataset(dset)
        
        # save dataset structure to file
        series.flush()
        
        # add beam attributes
        b["charge"][io.Record_Component.SCALAR].set_attribute("value", self.chargeSign()*SI.e)
        b["mass"][io.Record_Component.SCALAR].set_attribute("value", SI.me)
        
        # store data
        b["id"][io.Record_Component.SCALAR].store_chunk(self.ids())
        b["weighting"][io.Record_Component.SCALAR].store_chunk(self.weightings())
        b["position"]["z"].store_chunk(self.zs())
        b["position"]["x"].store_chunk(self.xs())
        b["position"]["y"].store_chunk(self.ys())
        b["momentum"]["z"].store_chunk(self.wzs())
        b["momentum"]["x"].store_chunk(self.wxs())
        b["momentum"]["y"].store_chunk(self.wys())
        
        # save data to file
        series.flush()
        
        return series
        
        
    # load beam (from OpenPMD format)
    @classmethod
    def load(_, filename, beamName="beam"):    
        
        # load file
        series = io.Series(filename, io.Access.read_only)
        
        # get particle data
        b = series.iterations[0].particles[beamName]
        
        # get attributes
        charge = b["charge"][io.Record_Component.SCALAR].get_attribute("value")
        mass = b["mass"][io.Record_Component.SCALAR].get_attribute("value")
        
        # extract phase space
        ids = b["id"][io.Record_Component.SCALAR].load_chunk()
        weightings = b["weighting"][io.Record_Component.SCALAR].load_chunk()
        xs = b["position"]["x"].load_chunk()
        ys = b["position"]["y"].load_chunk()
        zs = b["position"]["z"].load_chunk()
        wxs = b["momentum"]["x"].load_chunk()
        wys = b["momentum"]["y"].load_chunk()
        wzs = b["momentum"]["z"].load_chunk()
        series.flush()
        
        # make beam
        beam = Beam()
        beam.setPhaseSpace(Q=np.sum(weightings*charge), xs=xs, ys=ys, zs=zs, wxs=wxs, wys=wys, wzs=wzs)
        
        # add metadata to beam
        beam.trackableNumber = series.iterations[0].get_attribute("trackableNumber")
        beam.stageNumber = series.iterations[0].get_attribute("stageNumber")
        beam.location = series.iterations[0].get_attribute("location")  
        
        return beam
      