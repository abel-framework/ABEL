from abel.classes.spectrometer.quad_imaging import SpectrometerQuadImaging
from types import SimpleNamespace
from abel.utilities.other import check_kwargs
import scipy.constants as SI
import numpy as np

class SpectrometerQuadImagingImpactX(SpectrometerQuadImaging):
    
    def __init__(self, imaging_energy_x=None, imaging_energy_y=None, object_plane_x=0.0, object_plane_y=None, magnification_x=-8.0, magnification_y=None):

        super().__init__(imaging_energy_x=imaging_energy_x, imaging_energy_y=imaging_energy_y, object_plane_x=object_plane_x, object_plane_y=object_plane_y, magnification_x=magnification_x, magnification_y=magnification_y)

        self.num_slices = 50
        self.enable_space_charge = False
        self.enable_csr = True
        self.enable_isr = True
    
        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        "Track plasma-lens-based interstage using ImpactX."
        
        # re-perform the matching
        self.set_imaging()

        # get the lattice
        lattice = self.get_impactx_lattice()
        
        # run ImpactX
        from abel.apis.impactx.impactx_api import run_impactx
        beam, self.evolution = run_impactx(lattice, beam0, nom_energy=self.nom_energy, verbose=False, runnable=runnable, space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def get_impactx_lattice(self):
        "Set up the ImpactX plasma-lens-based interstage lattice."
        
        from impactx import elements
        
        # initialize lattice
        lattice = []

        ls, inv_rhos, ks, _, _ = self.matrix_lattice()

        for i in range(len(ls)):

            if abs(inv_rhos[i]) > 0:
                phi_dip = self.length_dipole*inv_rhos[i]
                B_dip = inv_rhos[i]*self.nom_energy/SI.c
                dipole = elements.ExactSbend(ds=self.length_dipole, phi=np.rad2deg(phi_dip), B=B_dip, nslice=self.num_slices, rotation=90)
                lattice.append(dipole)
            elif abs(ks[i]) > 0:
                quad = elements.ExactQuad(ds=self.length_quad, k=ks[i], nslice=self.num_slices)
                lattice.append(quad)
            else:
                drift = elements.ExactDrift(ds=ls[i], nslice=1)
                lattice.append(drift)

        return lattice
                