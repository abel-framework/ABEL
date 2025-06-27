from abel.classes.interstage.quads import InterstageQuads
import numpy as np
import os
from types import SimpleNamespace
import scipy.constants as SI

class InterstageQuadsImpactX(InterstageQuads):
    
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0, cancel_chromaticity=True, cancel_sec_order_dispersion=True, enable_csr=True, enable_isr=True, enable_space_charge=False, num_slices=50, use_monitors=False):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, length_dipole=length_dipole, field_dipole=field_dipole, R56=R56,
                         cancel_chromaticity=cancel_chromaticity, cancel_sec_order_dispersion=cancel_sec_order_dispersion,
                         enable_csr=enable_csr, enable_isr=enable_isr, enable_space_charge=enable_space_charge)

        # simulation options
        self.num_slices = num_slices
        self.use_monitors = use_monitors

    
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        "Track quad-based interstage using ImpactX."
        
        # re-perform the matching
        self.match()
        
        # get lattice
        lattice = self.get_impactx_lattice()
        
        # run ImpactX
        from abel.apis.impactx.impactx_api import run_impactx
        beam, self.evolution = run_impactx(lattice, beam0, nom_energy=self.nom_energy, verbose=False, runnable=runnable, save_beams=self.use_monitors, 
                                           space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)
        
        return super().track(beam, savedepth, runnable, verbose)

        
    def get_impactx_lattice(self):
        "Set up the ImpactX  quad-based interstage lattice."

        from impactx import elements
        
        # initialize lattice
        lattice = []

        # calculate the momentum
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        
        # add monitor (before and after gaps, and in the middle)
        if self.use_monitors:
            from abel.apis.impactx.impactx_api import initialize_amrex
            initialize_amrex()
            monitor = elements.BeamMonitor(name='monitor', backend='h5', encoding='g')
        else:
            monitor = []
        
        # gap drift (with monitors)
        gap = []
        quad_gap = []
    
        if self.use_monitors:
            gap.append(monitor)
        gap.append(elements.ExactDrift(ds=self.length_gap, nslice=1))
        if self.use_monitors:
            gap.append(monitor)
        
        # define dipole
        B_dip = self.field_dipole
        phi_dip = self.length_dipole*B_dip*SI.e/p0
        dipole = elements.ExactSbend(ds=self.length_dipole, phi=np.rad2deg(phi_dip), B=B_dip, nslice=self.num_slices)

        # define quads
        quad1 = elements.ExactQuad(ds=self.length_quadrupole, k=self.strength_quadrupole1/self.length_quadrupole, nslice=self.num_slices)
        quad2 = elements.ExactQuad(ds=self.length_quadrupole, k=self.strength_quadrupole2/self.length_quadrupole, nslice=self.num_slices)
        quad3 = elements.ExactQuad(ds=self.length_quadrupole, k=self.strength_quadrupole3/self.length_quadrupole, nslice=self.num_slices)
        
        # define first chicane dipole
        B_chic1 = self.field_chicane_dipole1
        phi_chic1 = self.length_chicane_dipole*B_chic1*(SI.e/p0)
        if abs(B_chic1) > 0:
            chicane_dipole1 = elements.ExactSbend(ds=self.length_chicane_dipole, phi=np.rad2deg(phi_chic1), B=B_chic1, nslice=self.num_slices)
        else:
            chicane_dipole1 = elements.ExactDrift(ds=self.length_chicane_dipole, nslice=self.num_slices)

        # define second chicane dipole
        B_chic2 = self.field_chicane_dipole2
        phi_chic2 = self.length_chicane_dipole*B_chic2*(SI.e/p0)
        if abs(B_chic2) > 0:
            chicane_dipole2 = elements.ExactSbend(ds=self.length_chicane_dipole, phi=np.rad2deg(phi_chic2), B=B_chic2, nslice=self.num_slices)
        else:
            chicane_dipole2 = elements.ExactDrift(ds=self.length_chicane_dipole, nslice=self.num_slices)

        # first sextupole
        if abs(self.strength_sextupole1) > 0:
            quad_gap_or_sextupole1 = elements.ExactMultipole(ds=self.length_quad_gap_or_sextupole, k_normal=[0.,0.,self.strength_sextupole1/self.length_quad_gap_or_sextupole], k_skew=[0.,0.,0.], nslice=self.num_slices)
        else:
            quad_gap_or_sextupole1 = elements.ExactDrift(ds=self.length_quad_gap_or_sextupole, nslice=1)

        # second sextupole
        if abs(self.strength_sextupole2) > 0:
            quad_gap_or_sextupole2 = elements.ExactMultipole(ds=self.length_quad_gap_or_sextupole, k_normal=[0.,0.,self.strength_sextupole2/self.length_quad_gap_or_sextupole], k_skew=[0.,0.,0.], nslice=self.num_slices)
        else:
            quad_gap_or_sextupole2 = elements.ExactDrift(ds=self.length_quad_gap_or_sextupole, nslice=1)
        
        # define sextupole (or gap)
        if abs(self.strength_sextupole3) > 0:
            half_central_gap_or_sextupole = elements.ExactMultipole(ds=self.length_central_gap_or_sextupole/2, k_normal=[0.,0.,self.strength_sextupole3/self.length_central_gap_or_sextupole], k_skew=[0.,0.,0.], nslice=self.num_slices)
        else:
            half_central_gap_or_sextupole = elements.ExactDrift(ds=self.length_central_gap_or_sextupole/2, nslice=1)
        
        
        # specify the lattice sequence
        lattice.extend(gap)
        lattice.append(dipole)
        lattice.extend(gap)
        lattice.append(quad1)
        lattice.extend(gap)
        lattice.append(quad2)
        lattice.extend(gap)
        lattice.append(quad_gap_or_sextupole1)
        lattice.extend(gap)
        lattice.append(quad3)
        lattice.extend(gap)
        lattice.append(quad_gap_or_sextupole2)
        lattice.extend(gap)
        lattice.append(chicane_dipole1)
        lattice.extend(gap)
        lattice.append(chicane_dipole2)
        lattice.extend(gap)
        lattice.append(half_central_gap_or_sextupole)
        if self.use_monitors:
            lattice.append(monitor)
        lattice.append(half_central_gap_or_sextupole)
        lattice.extend(gap)
        lattice.append(chicane_dipole2)
        lattice.extend(gap)
        lattice.append(chicane_dipole1)
        lattice.extend(gap)
        lattice.append(quad_gap_or_sextupole2)
        lattice.extend(gap)
        lattice.append(quad3)
        lattice.extend(gap)
        lattice.append(quad_gap_or_sextupole1)
        lattice.extend(gap)
        lattice.append(quad2)
        lattice.extend(gap)
        lattice.append(quad1)
        lattice.extend(gap)
        lattice.append(dipole)
        lattice.extend(gap)

        # remove first and last monitor
        if lattice[0] == monitor:
            del lattice[0]
        if lattice[-1] == monitor:
            del lattice[-1]
                
        return lattice
    