from abel.classes.plasma_lens.plasma_lens import PlasmaLens
import numpy as np
import scipy.constants as SI

class PlasmaLensImpactX(PlasmaLens):
    
    def __init__(self, length=None, radius=None, current=None, rel_nonlinearity=0):

        super().__init__(length, radius, current)
        
        # set nonlinearity (defined as R/Dx)
        self.rel_nonlinearity = rel_nonlinearity
        
    
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        import impactx
        from abel.wrappers.impactx.impactx_wrapper import beam2particle_container, particle_container2beam
        
        # initialize AMReX
        verbose_debug = False

        # make simulation object
        sim = impactx.ImpactX()

        # serial run on one CPU core
        sim.omp_threads = 1

        # set ImpactX verbosity
        sim.verbose = int(verbose_debug)
        sim.tiny_profiler = verbose_debug
        
        # convert to ImpactX particle container
        _, sim = beam2particle_container(beam0, sim)

        # add beam diagnostics
        monitor = impactx.elements.BeamMonitor("monitor", backend="h5")

        # TODO: include the interstage optic
        # TODO: print the evolution (i.e., put the values into the evolution namespace, and add plotting functions in the base class)
        
        # design the accelerator lattice
        ns = 25  # number of slices per ds in the element
        
        # specify thick tapered plasma lens element
        num_cuts = 10
        k0 = -self.get_focusing_gradient() * SI.c / beam0.energy()
        dtaper = self.rel_nonlinearity / self.radius  # 1/(horizontal dispersion in m)
        ds = self.length / num_cuts
        dk = k0 / num_cuts
        pl = impactx.elements.TaperedPL(k=dk, taper=dtaper, units=0)
        
        # drifts appearing the drift-kick sequence
        drift = impactx.elements.Drift(ds=ds/2, nslice=ns)
        
        # define the lens segments
        thick_lens = []
        for _ in range(0, num_cuts):
            thick_lens.extend([drift, pl, drift])
        
        # assign the lattice
        sim.lattice.extend(thick_lens)
        
        # run simulation
        sim.evolve()
        
        # convert back to ABEL beam
        beam = particle_container2beam(sim.particle_container())

        # clean shutdown
        sim.finalize()

        # copy meta data from input beam (will be iterated by super)
        beam.trackable_number = beam0.trackable_number
        beam.stage_number = beam0.stage_number
        beam.location = beam0.location
        
        return super().track(beam, savedepth, runnable, verbose)  


    def get_focusing_gradient(self):
        return SI.mu_0 * self.current / (2*np.pi * self.radius**2)
    
