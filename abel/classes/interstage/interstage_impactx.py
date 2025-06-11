from abel.classes.interstage import Interstage
import numpy as np
import os
from types import SimpleNamespace
import scipy.constants as SI

class InterstageImpactX(Interstage):
    
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0,
                 use_nonlinearity=True, use_chicane=True, use_sextupole=True, use_gaps=True, use_thick_lenses=True,
                 enable_csr=True, enable_isr=True, enable_space_charge=False, num_slices=50, use_monitors=False, keep_data=False):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, length_dipole=length_dipole, field_dipole=field_dipole, R56=R56, 
                         use_nonlinearity=use_nonlinearity, use_chicane=use_chicane, 
                         use_sextupole=use_sextupole, use_gaps=use_gaps, use_thick_lenses=use_thick_lenses, 
                         enable_csr=enable_csr, enable_isr=enable_isr, enable_space_charge=enable_space_charge)

        # simulation options
        self.num_slices = num_slices
        self.use_monitors = use_monitors
        self.keep_data = keep_data

    
    def get_lattice(self):

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
        if self.use_gaps:
            if self.use_monitors:
                gap.append(monitor)
            gap.append(elements.ExactDrift(ds=self.length_gap, nslice=1))
            if self.use_monitors:
                gap.append(monitor)
        
        # define dipole
        B_dip = self.field_dipole
        phi_dip = self.length_dipole*B_dip*SI.e/p0
        dipole = elements.ExactSbend(ds=self.length_dipole, phi=np.rad2deg(phi_dip), B=B_dip, nslice=self.num_slices)

        # define plasma lens
        kl_lens = self.strength_plasma_lens
        tau_lens = self.nonlinearity_plasma_lens
        plasma_lens = []

        pl_aperture = elements.Aperture(aperture_x=self.lens_radius, aperture_y=self.lens_radius, shape="elliptical")
        if self.use_apertures: # add aperture to the plasma lens
            plasma_lens.append(pl_aperture)
            
        if self.use_thick_lenses:
            drift_slice_pl = elements.ExactDrift(ds=self.length_plasma_lens/(self.num_slices+1), nslice=1)
            pl_slice = elements.TaperedPL(k=kl_lens/self.num_slices, taper=tau_lens)
            plasma_lens.extend([drift_slice_pl, pl_slice]*self.num_slices)
            plasma_lens.append(drift_slice_pl)
        else:
            plasma_lens.append(elements.TaperedPL(k=kl_lens, taper=tau_lens))

        if self.use_apertures: # add another one at the end of the lens
            plasma_lens.append(pl_aperture)

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

        # define sextupole
        ml_sext = self.strength_sextupole
        half_sextupole = []
        if self.use_thick_lenses:
            drift_slice_sext = elements.ExactDrift(ds=self.length_sextupole/(self.num_slices+1)/2, nslice=1)
            sext_slice = elements.Multipole(multipole=3, K_normal=ml_sext/self.num_slices, K_skew=0)
            half_sextupole.extend([drift_slice_sext, sext_slice]*self.num_slices)
            half_sextupole.append(drift_slice_sext)
            # TODO: upgrade to class impactx.elements.ExactMultipole when available
        else:
            half_sextupole.append(elements.Multipole(multipole=3, K_normal=ml_sext, K_skew=0))

        
        # specify the lattice sequence
        lattice.extend(gap)
        lattice.append(dipole)
        lattice.extend(gap)
        lattice.extend(plasma_lens)
        lattice.extend(gap)
        lattice.append(chicane_dipole1)
        lattice.extend(gap)
        lattice.append(chicane_dipole2)
        lattice.extend(gap)
        lattice.extend(half_sextupole)
        if self.use_monitors:
            lattice.append(monitor)
        lattice.extend(half_sextupole)
        lattice.extend(gap)
        lattice.append(chicane_dipole2)
        lattice.extend(gap)
        lattice.append(chicane_dipole1)
        lattice.extend(gap)
        lattice.extend(plasma_lens)
        lattice.extend(gap)
        lattice.append(dipole)
        lattice.extend(gap)

        # remove first and last monitor
        if lattice[0] == monitor:
            del lattice[0]
        if lattice[-1] == monitor:
            del lattice[-1]
                
        return lattice

        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # re-perform the matching
        self.match_all()
        
        # get lattice
        lattice = self.get_lattice()
        
        # run ImpactX
        from abel.apis.impactx.impactx_api import run_impactx
        beam, evol = run_impactx(lattice, beam0, nom_energy=self.nom_energy, verbose=False, runnable=runnable, keep_data=self.keep_data, save_beams=self.use_monitors,
                                 space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)
        self.evolution = evol
        
        return super().track(beam, savedepth, runnable, verbose)
        