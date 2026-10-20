# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.bds.impl.plasma_lens import BeamDeliverySystemPlasmaLens
import numpy as np
import scipy.constants as SI

class BeamDeliverySystemPlasmaLensImpactX(BeamDeliverySystemPlasmaLens):
    
    def __init__(self, nom_energy=None, beta0=None, alpha0=0, beta_star=None, L_star=2.0, length_scale=2.0, Dpx_star=0.04, charge_sign=-1, cancel_chromaticity=True, enable_csr=True, enable_isr=True, num_slices=50, use_monitors=False, use_apertures=False, isr_on_ref_part=True):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, alpha0=alpha0, beta_star=beta_star, L_star=L_star, length_scale=length_scale, Dpx_star=Dpx_star, charge_sign=charge_sign, cancel_chromaticity=cancel_chromaticity, enable_csr=enable_csr, enable_isr=enable_isr)
        
        # simulation options
        self.num_slices = num_slices
        self.use_monitors = use_monitors
        self.use_apertures = use_apertures
        self.isr_on_ref_part = isr_on_ref_part


    # ==================================================
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        "Track plasma-lens-based BDS using ImpactX."
        
        # re-perform the matching
        self.match()

        # get the lattice
        lattice = self.get_impactx_lattice()
        
        # run ImpactX
        from abel.wrappers.impactx.impactx_wrapper import run_impactx
        beam, self.evolution = run_impactx(lattice, beam0, nom_energy=self.nom_energy, verbose=False, runnable=runnable, save_beams=self.use_monitors, space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr, isr_on_ref_part=self.isr_on_ref_part)
        
        return super().track(beam, savedepth, runnable, verbose)
    

    # ==================================================
    def get_impactx_lattice(self):
        "Set up the ImpactX plasma-lens-based BDS lattice."
        
        from impactx import elements
        from abel.utilities.relativity import energy2momentum
        
        # initialize lattice
        lattice = []
        
        # add monitor (before and after gaps, and in the middle)
        if self.use_monitors:
            from abel.wrappers.impactx.impactx_wrapper import initialize_amrex
            initialize_amrex()
            monitor = elements.BeamMonitor(name='monitor', backend='h5', encoding='g')
        else:
            monitor = []
        
        # gap drift (with monitors)
        gap = []
        if self.use_monitors:
            gap.append(monitor)
        gap.append(elements.ExactDrift(ds=self.length_gap, nslice=1))
        if self.use_monitors:
            gap.append(monitor)
        
        # define dipoles
        phi1 = self.length_dipole1*self.field_dipole1*SI.e/energy2momentum(self.nom_energy)
        dipole1 = elements.ExactSbend(ds=self.length_dipole1, phi=np.rad2deg(phi1), B=self.field_dipole1, nslice=self.num_slices)

        phi2 = self.length_dipole2*self.field_dipole2*SI.e/energy2momentum(self.nom_energy)
        dipole2 = elements.ExactSbend(ds=self.length_dipole2, phi=np.rad2deg(phi2), B=self.field_dipole2, nslice=self.num_slices)

        phi3 = self.length_dipole3*self.field_dipole3*SI.e/energy2momentum(self.nom_energy)
        dipole3 = elements.ExactSbend(ds=self.length_dipole3, phi=np.rad2deg(phi3), B=self.field_dipole3, nslice=self.num_slices)

        phi4 = self.length_dipole4*self.field_dipole4*SI.e/energy2momentum(self.nom_energy)
        dipole4 = elements.ExactSbend(ds=self.length_dipole4, phi=np.rad2deg(phi4), B=self.field_dipole4, nslice=self.num_slices)
            
        # define plasma lenses
        ds_pl = self.length_plasma_lens/(self.num_slices+1)
        drift_slice_pl = elements.ExactDrift(ds=ds_pl, nslice=1)
        plasma_lens1 = [drift_slice_pl]
        plasma_lens2 = [drift_slice_pl]
        plasma_lens3 = [drift_slice_pl]
        for i in range(self.num_slices):
            plasma_lens1.append(elements.TaperedPL(k=self.strength_plasma_lens1/self.num_slices, taper=self.nonlinearity_plasma_lens1))
            plasma_lens1.append(drift_slice_pl)
            
            plasma_lens2.append(elements.TaperedPL(k=self.strength_plasma_lens2/self.num_slices, taper=self.nonlinearity_plasma_lens2))
            plasma_lens2.append(drift_slice_pl)
            
            plasma_lens3.append(elements.TaperedPL(k=self.strength_plasma_lens3/self.num_slices, taper=0.0))
            plasma_lens3.append(drift_slice_pl)
        
        # add another one at the end of the lens
        if self.use_apertures:
            aperture = elements.Aperture(aperture_x=self.lens_radius, aperture_y=self.lens_radius, shape="elliptical")
            
            pl1 = [aperture]
            pl1.extend(plasma_lens1)
            pl1.append(aperture)
            plasma_lens1 = pl1
            
            pl2 = [aperture]
            pl2.extend(plasma_lens2)
            pl2.append(aperture)
            plasma_lens2 = pl2
            
            pl3 = [aperture]
            pl3.extend(plasma_lens3)
            pl3.append(aperture)
            plasma_lens3 = pl3

        # define sextupole (or gap)
        if abs(self.strength_sextupole) > 0.0:
            sextupole = elements.ExactMultipole(ds=self.length_sextupole, k_normal=[0.,0.,self.strength_sextupole/self.length_sextupole], k_skew=[0.,0.,0.], nslice=self.num_slices)
        else:
            sextupole = elements.ExactDrift(ds=self.length_sextupole, nslice=1)

        # final drift (L*)
        drift_Lstar = []
        if self.use_monitors:
            drift_Lstar.append(monitor)
        drift_Lstar.append(elements.ExactDrift(ds=self.L_star, nslice=10))
        if self.use_monitors:
            drift_Lstar.append(monitor)
        
        # specify the lattice sequence
        lattice.extend(gap)
        lattice.append(dipole4)
        lattice.extend(gap)
        lattice.extend(plasma_lens3)
        lattice.extend(gap)
        lattice.append(dipole3)
        lattice.extend(gap)
        lattice.extend(plasma_lens2)
        lattice.extend(gap)
        lattice.append(dipole2)
        lattice.extend(gap)
        lattice.append(sextupole)
        lattice.extend(gap)
        lattice.append(dipole1)
        lattice.extend(gap)
        lattice.extend(plasma_lens1)
        lattice.extend(drift_Lstar)

        # remove first and last monitor
        if lattice[0] == monitor:
            del lattice[0]
        if lattice[-1] == monitor:
            del lattice[-1]
                
        return lattice

        