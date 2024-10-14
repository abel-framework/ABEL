from abel import Interstage
import impactx
import amrex.space3d as amr
from abel.apis.impactx.impactx_api import beam2particle_container, particle_container2beam
import contextlib, os

class InterstageImpactX(Interstage):
    
    def __init__(self, nom_energy=None, length=None):
        super().__init__()
        
        self.nom_energy = nom_energy
        self.length = length

    
     # lattice length
    def get_length(self):
        return self.length
        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # initialize AMReX
        amr.initialize(["amrex.omp_threads=1", f"amrex.verbose={int(verbose)}"])

        # make simulation object
        sim = impactx.ImpactX()
        sim.verbose = int(verbose)
        
        # convert to ImpactX particle container
        _, sim = beam2particle_container(beam0, sim)

        # add beam diagnostics
        monitor = impactx.elements.BeamMonitor("monitor", backend="h5")

        # TODO: include the interstage optic
        # TODO: print the evolution (i.e., put the values into the evolution namespace, and add plotting functions in the base class)
        
        # design the accelerator lattice
        num_slice = 25  # number of slices per ds in the element
        lattice = [monitor, impactx.elements.Drift(ds=self.length, nslice=num_slice), monitor]
        
        # assign the lattice
        sim.lattice.extend(lattice)
        
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
        