import numpy as np
import openpmd_api as io
from datetime import datetime
from pytz import timezone
import scipy.constants as SI
from abel.utilities.relativity import energy2proper_velocity, proper_velocity2energy, momentum2proper_velocity, proper_velocity2momentum, proper_velocity2gamma, energy2gamma
from abel.utilities.statistics import prct_clean, prct_clean2d
from abel.utilities.plasma_physics import k_p
from abel.physics_models.hills_equation import evolve_hills_equation_analytic
from matplotlib import pyplot as plt

class Beam():
    
    def __init__(self, phasespace=None, num_particles=1000):

        # the phase space variable is private
        if phasespace is not None:
            self.__phasespace = phasespace
        else:
            self.__phasespace = self.reset_phase_space(num_particles)
            
        self.trackable_number = -1 # will increase to 0 after first tracking element
        self.stage_number = 0
        self.location = 0        
    
    
    # reset phase space
    def reset_phase_space(self, num_particles):
        self.__phasespace = np.zeros((8, num_particles))
    
    # filter out macroparticles based on a mask (true means delete)
    def __delitem__(self, indices):
        if hasattr(indices, 'len'):
            if len(indices) == len(self):
                indices = np.where(indices)
        self.__phasespace = np.ascontiguousarray(np.delete(self.__phasespace, indices, 1))
    
    # filter out nans
    def remove_nans(self):
        del self[np.isnan(self).any(axis=1)]
        
    # set phase space
    def set_phase_space(self, Q, xs, ys, zs, uxs=None, uys=None, uzs=None, pxs=None, pys=None, pzs=None, xps=None, yps=None, Es=None):
        
        # make empty phase space
        num_particles = len(xs)
        self.reset_phase_space(num_particles)
        
        # add positions
        self.set_xs(xs)
        self.set_ys(ys)
        self.set_zs(zs)
        
        # add momenta
        if uzs is None:
            if pzs is not None:
                uzs = momentum2proper_velocity(pzs)
            elif Es is not None:
                uzs = energy2proper_velocity(Es)
        self.__phasespace[5,:] = uzs
        
        if uxs is None:
            if pxs is not None:
                uxs = momentum2proper_velocity(pxs)
            elif xps is not None:
                uxs = xps * uzs
        self.__phasespace[3,:] = uxs
        
        if uys is None:
            if pys is not None:
                uys = momentum2proper_velocity(pys)
            elif yps is not None:
                uys = yps * uzs
        self.__phasespace[4,:] = uys
        
        # charge
        self.__phasespace[6,:] = Q/num_particles
        
        # ids
        self.__phasespace[7,:] = np.arange(num_particles)
       
    
    # addition operator (add two beams using the + operator)
    def __add__(self, beam):
        return Beam(phasespace = np.append(self.__phasespace, beam.__phasespace, axis=1))
        
    # in-place addition operator (add one beam to another using the += operator)
    def __iadd__(self, beam):
        if beam is not None:    
            self.__phasespace = np.append(self.__phasespace, beam.__phasespace, axis=1)
        return self
    
    # indexing operator (get single particle out)
    def __getitem__(self, index):
        return self.__phasespace[:,index]
    
    # "length" operator (number of macroparticles)
    def __len__(self):
        return self.__phasespace.shape[1]
    
    # string operator (called when printing)
    def __str__(self):
        return f"Beam: {len(self)} macroparticles, {self.charge()*1e9:.2f} nC, {self.energy()/1e9:.2f} GeV"
        
        
    
    
    ## BEAM ARRAYS

    # get phase space variables
    def xs(self):
        return self.__phasespace[0,:]
    def ys(self):
        return self.__phasespace[1,:]
    def zs(self):
        return self.__phasespace[2,:]
    def uxs(self):
        return self.__phasespace[3,:]
    def uys(self):
        return self.__phasespace[4,:]
    def uzs(self):
        return self.__phasespace[5,:]
    def qs(self):
        return self.__phasespace[6,:]
    def ids(self):
        return self.__phasespace[7,:]
    
    # set phase space variables
    def set_xs(self, xs):
        self.__phasespace[0,:] = xs
    def set_ys(self, ys):
        self.__phasespace[1,:] = ys
    def set_zs(self, zs):
        self.__phasespace[2,:] = zs
    def set_uxs(self, uxs):
        self.__phasespace[3,:] = uxs
    def set_uys(self, uys):
        self.__phasespace[4,:] = uys
    def set_uzs(self, uzs):
        self.__phasespace[5,:] = uzs
        
    def set_xps(self, xps):
        self.set_uxs(xps*self.uzs())
    def set_yps(self, yps):
        self.set_uys(yps*self.uzs())
    def set_Es(self, Es):
        self.set_uzs(energy2proper_velocity(Es))
        
    def set_qs(self, qs):
        self.__phasespace[6,:] = qs
    def set_ids(self, ids):
        self.__phasespace[7,:] = ids
        
    def weightings(self):
        return self.__phasespace[6,:]/(self.charge_sign()*SI.e)
    
    # copy another beam's macroparticle charge
    def copy_particle_charge(self, beam):
        self.set_qs(np.median(beam.qs()))
    
    
    def rs(self):
        return np.sqrt(self.xs()**2 + self.ys()**2)
    
    def pxs(self):
        return proper_velocity2momentum(self.uxs())
    def pys(self):
        return proper_velocity2momentum(self.uys())
    def pzs(self):
        return proper_velocity2momentum(self.uzs())
    
    def xps(self):
        return self.uxs()/self.uzs()
    def yps(self):
        return self.uys()/self.uzs()

    def gammas(self):
        return proper_velocity2gamma(self.uzs())
    def Es(self):
        return proper_velocity2energy(self.uzs())
    def deltas(self, pz0=None):
        if pz0 is None:
            pz0 = np.mean(self.pzs())
        return self.pzs()/pz0 -1
    
    def ts(self):
        return self.zs()/SI.c
    
    # vector of transverse positions and angles: (x, x', y, y')
    def transverse_vector(self):
        vector = np.zeros((4,len(self)))
        vector[0,:] = self.xs()
        vector[1,:] = self.xps()
        vector[2,:] = self.ys()
        vector[3,:] = self.yps()
        return vector
    
    # set phase space based on transverse vector: (x, x', y, y')
    def set_transverse_vector(self, vector):
        self.set_xs(vector[0,:])
        self.set_xps(vector[1,:])
        self.set_ys(vector[2,:])
        self.set_yps(vector[3,:])  
    
    
    ## BEAM STATISTICS
    
    def charge(self):
        qs = self.qs()
        return np.sum(qs[~np.isnan(qs)])
    
    def abs_charge(self):
        return abs(self.charge())
    
    def charge_sign(self):
        return self.charge()/abs(self.charge())
    
    def energy(self, clean=False):
        return np.mean(prct_clean(self.Es(), clean))
    
    def gamma(self, clean=False):
        return np.mean(prct_clean(self.gammas(), clean))
    
    def total_energy(self):
        return self.abs_charge()*self.energy()
    
    def energy_spread(self, clean=False):
        return np.std(prct_clean(self.Es(), clean))
    
    def rel_energy_spread(self, clean=False):
        return self.energy_spread(clean)/self.energy(clean)
    
    def z_offset(self, clean=False):
        return np.mean(prct_clean(self.zs(), clean))
    
    def bunch_length(self, clean=False):
        return np.std(prct_clean(self.zs(), clean))
    
    def x_offset(self, clean=False):
        return np.mean(prct_clean(self.xs(), clean))
    
    def y_offset(self, clean=False):
        return np.mean(prct_clean(self.ys(), clean))

    def x_angle(self, clean=False):
        return np.mean(prct_clean(self.xps(), clean))
    
    def y_angle(self, clean=False):
        return np.mean(prct_clean(self.yps(), clean))
        
    def ux_offset(self, clean=False):
        return np.mean(prct_clean(self.uxs(), clean))
    
    def uy_offset(self, clean=False):
        return np.mean(prct_clean(self.uys(), clean))
    
    def beam_size_x(self, clean=False):
        return np.std(prct_clean(self.xs(), clean))

    def beam_size_y(self, clean=False):
        return np.std(prct_clean(self.ys(), clean))
    
    def divergence_x(self, clean=False):
        return np.std(prct_clean(self.xps(), clean))

    def divergence_y(self, clean=False):
        return np.std(prct_clean(self.yps(), clean))
    
    def geom_emittance_x(self, clean=False):
        xs, xps = prct_clean2d(self.xs(), self.xps(), clean)
        return np.sqrt(np.linalg.det(np.cov(xs, xps)))
    
    def geom_emittance_y(self, clean=False):
        ys, yps = prct_clean2d(self.ys(), self.yps(), clean)
        return np.sqrt(np.linalg.det(np.cov(ys, yps)))
    
    def norm_emittance_x(self, clean=False):
        xs, uxs = prct_clean2d(self.xs(), self.uxs(), clean)
        return np.sqrt(np.linalg.det(np.cov(xs, uxs/SI.c)))
    
    def norm_emittance_y(self, clean=False):
        ys, uys = prct_clean2d(self.ys(), self.uys(), clean)
        return np.sqrt(np.linalg.det(np.cov(ys, uys/SI.c)))
    
    def beta_x(self, clean=False):
        xs, xps = prct_clean2d(self.xs(), self.xps(), clean)
        covx = np.cov(xs, xps)
        return covx[0,0]/np.sqrt(np.linalg.det(covx))
    
    def beta_y(self, clean=False):
        ys, yps = prct_clean2d(self.ys(), self.yps(), clean)
        covy = np.cov(ys, yps)
        return covy[0,0]/np.sqrt(np.linalg.det(covy))
    
    def alpha_x(self, clean=False):
        xs, xps = prct_clean2d(self.xs(), self.xps(), clean)
        covx = np.cov(xs, xps)
        return -covx[1,0]/np.sqrt(np.linalg.det(covx))
    
    def alpha_y(self, clean=False):
        ys, yps = prct_clean2d(self.ys(), self.yps(), clean)
        covy = np.cov(ys, yps)
        return -covy[1,0]/np.sqrt(np.linalg.det(covy))
    
    def gamma_x(self, clean=False):
        xs, xps = prct_clean2d(self.xs(), self.xps(), clean)
        covx = np.cov(xs, xps)
        return covx[1,1]/np.sqrt(np.linalg.det(covx))
    
    def gamma_y(self, clean=False):
        ys, yps = prct_clean2d(self.ys(), self.yps(), clean)
        covy = np.cov(ys, yps)
        return covy[1,1]/np.sqrt(np.linalg.det(covy))
    
    
    def peak_density(self):
        return (self.charge()/SI.e)/(np.sqrt(2*SI.pi)**3*self.beam_size_x()*self.beam_size_y()*self.bunch_length())
    
    def peak_current(self):
        Is, _ = self.current_profile()
        return max(abs(Is))
    

    ## BEAM HALO CLEANING (EXTREME OUTLIERS)
    def remove_halo_particles(self, nsigma=20):
        xfilter = np.abs(self.xs()-self.x_offset(clean=True)) > nsigma*self.beam_size_x(clean=True)
        xpfilter = np.abs(self.xps()-self.x_angle(clean=True)) > nsigma*self.divergence_x(clean=True)
        yfilter = np.abs(self.ys()-self.y_offset(clean=True)) > nsigma*self.beam_size_y(clean=True)
        ypfilter = np.abs(self.ys()-self.y_angle(clean=True)) > nsigma*self.divergence_y(clean=True)
        filter = np.logical_or(np.logical_or(xfilter, xpfilter), np.logical_or(yfilter, ypfilter))
        del self[filter]

    
    ## BEAM PROJECTIONS
    
    def projected_density(self, fcn, bins=None):
        if bins is None:
            Nbins = int(np.sqrt(len(self)/2))
            bins = np.linspace(min(fcn()), max(fcn()), Nbins)
        counts, edges = np.histogram(fcn(), weights=self.qs(), bins=bins)
        ctrs = (edges[0:-1] + edges[1:])/2
        proj = counts/np.diff(edges)
        return proj, ctrs
        
    def current_profile(self, bins=None):
        return self.projected_density(self.ts, bins=bins)
    
    def longitudinal_num_density(self, bins=None):
        dQdz, zs = self.projected_density(self.zs, bins=bins)
        dNdz = dQdz / SI.e
        return dNdz, zs
    
    def energy_spectrum(self, bins=None):
        return self.projected_density(self.Es, bins=bins)
    
    def rel_energy_spectrum(self, nom_energy=None, bins=None):
        if nom_energy is None:
            nom_energy = self.energy()
        return self.projected_density(lambda: self.Es()/nom_energy-1, bins=bins)
    
    def transverse_profile_x(self, bins=None):
        return self.projected_density(self.xs, bins=bins)
    
    def transverse_profile_y(self, bins=None):
        return self.projected_density(self.ys, bins=bins)
    
    ## phase spaces
    
    def phase_space_density(self, hfcn, vfcn, hbins=None, vbins=None):
        self.remove_nans()
        if hbins is None:
            hbins = round(np.sqrt(len(self))/2)
        if vbins is None:
            vbins = round(np.sqrt(len(self))/2)
        counts, hedges, vedges = np.histogram2d(hfcn(), vfcn(), weights=self.qs(), bins=(hbins, vbins))
        hctrs = (hedges[0:-1] + hedges[1:])/2
        vctrs = (vedges[0:-1] + vedges[1:])/2
        density = (counts/np.diff(vedges)).T/np.diff(hedges)
        return density, hctrs, vctrs
    
    def density_lps(self, hbins=None, vbins=None):
        return self.phase_space_density(self.zs, self.Es, hbins=hbins, vbins=vbins)
    
    def density_transverse(self, hbins=None, vbins=None):
        return self.phase_space_density(self.xs, self.ys, hbins=hbins, vbins=vbins)    
        
    
    ## PLOTTING
    def plot_current_profile(self):
        dQdt, ts = self.current_profile()

        fig, ax = plt.subplots()
        fig.set_figwidth(6)
        fig.set_figheight(4)        
        ax.plot(ts*SI.c*1e6, -dQdt/1e3)
        ax.set_xlabel('z (um)')
        ax.set_ylabel('Beam current (kA)')
    
    def plot_lps(self):
        dQdzdE, zs, Es = self.density_lps()

        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(5)  
            
        p = ax.pcolor(zs*1e6, Es/1e9, -dQdzdE*1e15, cmap='GnBu', shading='auto')
        ax.set_xlabel('z (um)')
        ax.set_ylabel('E (GeV)')
        ax.set_title('Longitudinal phase space')
        cb = fig.colorbar(p)
        cb.ax.set_ylabel('Charge density (pC/um/GeV)')
        
    def plot_trace_space_x(self):
        dQdxdxp, xs, xps = self.phase_space_density(self.xs, self.xps)

        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(5)  
        p = ax.pcolor(xs*1e6, xps*1e3, -dQdxdxp*1e3, cmap='GnBu', shading='auto')
        ax.set_xlabel('x (um)')
        ax.set_ylabel('x'' (mrad)')
        ax.set_title('Horizontal trace space')
        cb = fig.colorbar(p)
        cb.ax.set_ylabel('Charge density (pC/um/mrad)')
        
    def plot_trace_space_y(self):
        dQdydyp, ys, yps = self.phase_space_density(self.ys, self.yps)

        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(5)  
        p = ax.pcolor(ys*1e6, yps*1e3, -dQdydyp*1e3, cmap='GnBu', shading='auto')
        ax.set_xlabel('y (um)')
        ax.set_ylabel('y'' (mrad)')
        ax.set_title('Vertical trace space')
        cb = fig.colorbar(p)
        cb.ax.set_ylabel('Charge density (pC/um/mrad)')
        
        
    
    ## CHANGE BEAM
    
    def accelerate(self, energy_gain=0, chirp=0, z_offset=0):
        
        # add energy and chirp
        Es = self.Es() + energy_gain 
        Es = Es + np.sign(self.qs()) * (self.zs()-z_offset) * chirp
        self.set_uzs(energy2proper_velocity(Es))
        
        # remove particles with subzero energy
        del self[Es < 0]
        
    def compress(self, R_56, nom_energy):
        zs = self.zs() + (1-self.Es()/nom_energy) * R_56
        self.set_zs(zs)
        
    def scale_to_length(self, bunch_length):
        z_mean = self.z_offset()
        zs_scaled = z_mean + (self.zs()-z_mean)*bunch_length/self.bunch_length()
        self.set_zs(zs_scaled)
        
    # betatron damping (must be done before acceleration)
    def apply_betatron_damping(self, deltaE):
        
        # remove particles with subzero energy
        del self[self.Es() < 0]
        del self[np.isnan(self.Es())]
        
        gammasBoosted = energy2gamma(abs(self.Es()+deltaE))
        betamag = np.sqrt(self.gammas()/gammasBoosted)
        self.magnify_beta_function(betamag)
        
    
    # magnify beta function (increase beam size, decrease divergence)
    def magnify_beta_function(self, beta_mag, axis_defining_beam=None):
        
        # calculate beam (not beta) magnification
        mag = np.sqrt(beta_mag)

        if axis_defining_beam is None:
            x_offset, y_offset = 0, 0
            ux_offset, uy_offset = 0, 0
        else:
            x_offset, y_offset = axis_defining_beam.x_offset(), axis_defining_beam.y_offset()
            ux_offset, uy_offset = axis_defining_beam.ux_offset(), axis_defining_beam.ux_offset()
        
        self.set_xs((self.xs()-x_offset) * mag + x_offset)
        self.set_ys(self.ys() * mag)
        self.set_uxs(self.uxs() / mag)
        self.set_uys(self.uys() / mag)
        
        
    def flip_transverse_phase_spaces(self, flip_momenta=True, flip_positions=False):
        if flip_momenta:
            self.set_uxs(-self.uxs())
            self.set_uys(-self.uys())
        elif flip_positions:
            self.set_xs(-self.xs())
            self.set_ys(-self.ys())
        
    def apply_betatron_motion(self, L, n0, deltaEs, x0_driver=0, y0_driver=0):
        
        # remove particles with subzero energy
        del self[self.Es() < 0]
        del self[np.isnan(self.Es())]
        
        # determine initial and final Lorentz factor
        gamma0s = energy2gamma(self.Es())
        gammas = energy2gamma(abs(self.Es()+deltaEs))
        dgamma_ds = (gammas-gamma0s)/L
        
        # calculate final positions and angles after betatron motion
        xs, uxs = evolve_hills_equation_analytic(self.xs()-x0_driver, self.uxs(), L, gamma0s, dgamma_ds, k_p(n0))
        ys, uys = evolve_hills_equation_analytic(self.ys()-y0_driver, self.uys(), L, gamma0s, dgamma_ds, k_p(n0))
        
        # set new beam positions and angles (shift back driver offsets)
        self.set_xs(xs+x0_driver)
        self.set_uxs(uxs)
        self.set_ys(ys+y0_driver)
        self.set_uys(uys)
        
        
  
    ## SAVE AND LOAD BEAM
    
    def filename(self, runnable, beam_name):
        return runnable.shot_path() + "/" + beam_name + "_" + str(self.trackable_number).zfill(3) + "_{:012.6F}".format(self.location) + ".h5"
    
    
    # save beam (to OpenPMD format)
    def save(self, runnable=None, filename=None, beam_name="beam", series=None):
        
        if len(self) == 0:
            return
        
        # make new file if not provided
        if series is None:

            # open a new file
            if runnable is not None:
                filename = self.filename(runnable, beam_name)
            
            # open a new file
            series = io.Series(filename, io.Access.create)
            
        
            # add metadata
            series.author = "ABEL (the Advanced Beams and Extreme Linacs code)"
            series.date = datetime.now(timezone('CET')).strftime('%Y-%m-%d %H:%M:%S %z')

        # make step (only one)
        index = 0
        iteration = series.iterations[index]
        
        # add attributes
        iteration.set_attribute("time", self.location/SI.c)
        for key, value in self.__dict__.items():
            if not "__phasespace" in key:
                iteration.set_attribute(key, value)
                
        # make beam record
        particles = iteration.particles[beam_name]
       
        # generate datasets
        dset_z = io.Dataset(self.zs().dtype, extent=self.zs().shape)
        dset_x = io.Dataset(self.xs().dtype, extent=self.xs().shape)
        dset_y = io.Dataset(self.ys().dtype, extent=self.ys().shape)
        dset_zoff = io.Dataset(np.dtype('float64'), extent=[1])
        dset_xoff = io.Dataset(np.dtype('float64'), extent=[1])
        dset_yoff = io.Dataset(np.dtype('float64'), extent=[1])
        dset_uz = io.Dataset(self.uzs().dtype, extent=self.uzs().shape)
        dset_ux = io.Dataset(self.uxs().dtype, extent=self.uxs().shape)
        dset_uy = io.Dataset(self.uys().dtype, extent=self.uys().shape)
        dset_w = io.Dataset(self.weightings().dtype, extent=self.weightings().shape)
        dset_id = io.Dataset(self.ids().dtype, extent=self.ids().shape)
        dset_q = io.Dataset(np.dtype('float64'), extent=[1])
        dset_m = io.Dataset(np.dtype('float64'), extent=[1])
        
        # prepare for writing
        particles['position']['z'].reset_dataset(dset_z)
        particles['position']['x'].reset_dataset(dset_x)
        particles['position']['y'].reset_dataset(dset_y)
        particles['positionOffset']['z'].reset_dataset(dset_zoff)
        particles['positionOffset']['x'].reset_dataset(dset_xoff)
        particles['positionOffset']['y'].reset_dataset(dset_yoff)
        particles['momentum']['z'].reset_dataset(dset_uz)
        particles['momentum']['x'].reset_dataset(dset_ux)
        particles['momentum']['y'].reset_dataset(dset_uy)
        particles['weighting'][io.Record_Component.SCALAR].reset_dataset(dset_w)
        particles['id'][io.Record_Component.SCALAR].reset_dataset(dset_id)        
        particles['charge'][io.Record_Component.SCALAR].reset_dataset(dset_q)
        particles['mass'][io.Record_Component.SCALAR].reset_dataset(dset_m)
        
        # store data
        particles['position']['z'].store_chunk(self.zs())
        particles['position']['x'].store_chunk(self.xs())
        particles['position']['y'].store_chunk(self.ys())
        particles['positionOffset']['x'].make_constant(0.)
        particles['positionOffset']['y'].make_constant(0.)
        particles['positionOffset']['z'].make_constant(0.)
        particles['momentum']['z'].store_chunk(self.uzs())
        particles['momentum']['x'].store_chunk(self.uxs())
        particles['momentum']['y'].store_chunk(self.uys())
        particles['weighting'][io.Record_Component.SCALAR].store_chunk(self.weightings())
        particles['id'][io.Record_Component.SCALAR].store_chunk(self.ids())
        particles["charge"][io.Record_Component.SCALAR].make_constant(self.charge_sign()*SI.e)
        particles["mass"][io.Record_Component.SCALAR].make_constant(SI.m_e)

        # set SI units (scaling factor)
        particles['momentum']['z'].unit_SI = SI.m_e
        particles['momentum']['x'].unit_SI = SI.m_e
        particles['momentum']['y'].unit_SI = SI.m_e
        
        # set dimensional units
        particles['position'].unit_dimension = {io.Unit_Dimension.L: 1}
        particles['positionOffset'].unit_dimension = {io.Unit_Dimension.L: 1}
        particles['momentum'].unit_dimension = {io.Unit_Dimension.L: 1, io.Unit_Dimension.M: 1, io.Unit_Dimension.T: -1}
        particles['charge'].unit_dimension = {io.Unit_Dimension.T: 1, io.Unit_Dimension.I: 1}
        particles['mass'].unit_dimension = {io.Unit_Dimension.M: 1}
        
        # save data to file
        series.flush()
        
        return series
        
        
    # load beam (from OpenPMD format)
    @classmethod
    def load(_, filename, beam_name="beam"):
        
        # load file
        series = io.Series(filename, io.Access.read_only)
        
        # find index (use last one)
        *_, index = series.iterations
        
        # get particle data
        particles = series.iterations[index].particles[beam_name]
        
        # get attributes
        charge = particles["charge"][io.Record_Component.SCALAR].get_attribute("value")
        mass = particles["mass"][io.Record_Component.SCALAR].get_attribute("value")
        
        # extract phase space
        ids = particles["id"][io.Record_Component.SCALAR].load_chunk()
        weightings = particles["weighting"][io.Record_Component.SCALAR].load_chunk()
        xs = particles['position']['x'].load_chunk()
        ys = particles['position']['y'].load_chunk()
        zs = particles['position']['z'].load_chunk()
        pxs_unscaled = particles['momentum']['x'].load_chunk()
        pys_unscaled = particles['momentum']['y'].load_chunk()
        pzs_unscaled = particles['momentum']['z'].load_chunk()
        series.flush()
        
        # apply SI scaling
        pxs = pxs_unscaled * particles['momentum']['x'].unit_SI
        pys = pys_unscaled * particles['momentum']['y'].unit_SI
        pzs = pzs_unscaled * particles['momentum']['z'].unit_SI
        
        # make beam
        beam = Beam()
        beam.set_phase_space(Q=np.sum(weightings*charge), xs=xs, ys=ys, zs=zs, pxs=pxs, pys=pys, pzs=pzs)
        
        # add metadata to beam
        try: 
            beam.trackable_number = series.iterations[index].get_attribute("trackable_number")
            beam.stage_number = series.iterations[index].get_attribute("stage_number")
            beam.location = series.iterations[index].get_attribute("location")  
        except:
            beam.trackable_number = None
            beam.stage_number = None
            beam.location = None
        
        return beam
      