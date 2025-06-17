import numpy as np
from abel.classes.bds.bds import BeamDeliverySystem
import scipy.constants as SI

class DriverDelaySystem(BeamDeliverySystem):

    def __init__(self, E_nom, delay_per_stage, length_stage, num_stages, oscillation_amplitude, kx=None, ky=None, B_bend=1):
        super().__init__()
        self.E_nom = E_nom
        self.delay_per_stage = delay_per_stage
        self.num_stages = num_stages
        self.A_oscillation = oscillation_amplitude
        self.length_stage = length_stage
        self.kx = kx
        self.ky = ky
        self.B_bend = B_bend

    def get_lattice_single_stage(self, kx, ky):
        """
        Oscillating Chicanes:
            4 dipoles, 70% fill factor, quadrupoles of the crests and troughs. Start at the trough.
            Max dipole field, 1.2.
            Assume no drift between dipoles and quads, for now.

        Input:
            kx: focusing strength in x
            ky: focusing strength in y
            B_bend: dipole field [Tesla]
        Returns:
            list of lattice elements
        """
        l_quads = 0.2 # m

        fill_factor = 0.7
        # Length of section without bends
        L_single_bend = (self.length_stage + SI.c*self.delay_per_stage)*fill_factor/4
        L_straight = (SI.c*self.delay_per_stage + self.length_stage - 4*L_single_bend)/2-l_quads

        p = self.E_nom / SI.e /1e9 # SI to GeV
        r = p/0.3*self.B_bend # p in GeV
        bend_angle = L_single_bend/r

        import impactx

        lattice = []

        ns = 25 # Number of slices

        # Quads
        quad_x_half = impactx.elements.ExactQuad(name="quad_x", k=kx/2, ds = l_quads/2, nslice=ns)
        quad_y = impactx.elements.ExactQuad(name="quad_y", k=ky, ds = l_quads, nslice=ns)

        # Bend
        bend_up = impactx.elements.ExactSbend(name="bends", ds=L_single_bend, phi=bend_angle)
        bend_down = impactx.elements.ExactSbend(name="bends", ds=L_single_bend, phi=-bend_angle)

        # Drift
        drift = impactx.elements.ExactDrift(name="drift", ds=L_straight, nslice=ns)

        # Make lattice
        lattice.append(quad_x_half)
        lattice.append(bend_up)
        lattice.append(drift)
        lattice.append(bend_down)
        lattice.append(quad_y)
        lattice.append(bend_down)
        lattice.append(drift)
        lattice.append(bend_up)
        lattice.append(quad_x_half)

        return lattice
    
    def get_lattice(self, kx, ky):
        lattice = self.get_lattice_single_stage(kx, ky)*self.num_stages
        return lattice
    
    def match_quads(self):
        import impactx
        sim = impactx.ImpactX()
        p = self.E_nom / SI.e /1e6 # SI to MeV
        ref = sim.particle_container().ref_particle()
        ref.set_charge_qe(-1.0).set_mass_MeV(0.510998950).set_kin_energy_MeV(p)

        dist = impactx.distribution_input_helpers.twiss

        def minimize_periodic(params):
            kx = params[0]
            ky = params[1]
            lattice = self.get_lattice_single_stage(kx, ky)
            sim.lattice.extend(lattice)
            sim.track_envelope()
            sim.finalize()
    
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        from abel.apis.impactx.impactx_api import run_impactx

        # Get lattice
        if self.kx is None or self.ky is None:
            ks = self.match_quads()
            self.kx = ks[0]
            self.ky = ks[1]

        lattice = self.get_lattice(self.kx, self.ky)
        
        # run ImpactX
        beam, evol = run_impactx(lattice, beam0, verbose=False, runnable=runnable, keep_data=self.keep_data, space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)
        
        # save evolution
        self.evolution = evol
        
        return super().track(beam, savedepth, runnable, verbose)









