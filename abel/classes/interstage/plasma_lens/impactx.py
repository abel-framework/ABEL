# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.interstage.plasma_lens import InterstagePlasmaLens
import numpy as np
import os
from types import SimpleNamespace
import scipy.constants as SI

class InterstagePlasmaLensImpactX(InterstagePlasmaLens):
    """
    Interstage modul using ImpactX [1]_ for full 3D particle tracking through an 
    interstage lattice with optional physics effects.

    This subclass of :class:`InterstagePlasmaLens` enables realistic beam 
    tracking with support for Coherent Synchrotron Radiation (CSR), Incoherent 
    Synchrotron Radiation (ISR), and space-charge modeling.

    Inherits all attributes from :class:`InterstagePlasmaLens`.

    Attributes
    ----------
    num_slices : int
        Number of longitudinal slices per beamline element in the ImpactX 
        simulation. Defaults to 50.

    use_monitors : bool
        If ``True``, inserts ImpactX ``BeamMonitor`` in the lattice for 
        recording intermediate beam states. Defaults to ``False``.

    enable_isr_on_ref_part : bool, optional
        Flag for applying `ISR to the reference particle <https://impactx.readthedocs.io/en/latest/usage/python.html#impactx.ImpactX.isr_on_ref_part>`_.
        Note that this does not have any effect if 
        :attr:`self.enable_isr <abel.Interstage.enable_isr>` is ``False``. 
        Defaults to ``True``.

    keep_data : bool
        If ``True``, retains raw data files produced by ImpactX. Defaults to 
        ``False``.

    References
    ----------
    .. [1] ImpactX documentation: https://impactx.readthedocs.io/en/latest/
    """
    # TODO: keep_data is not used.
    
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0, cancel_chromaticity=True, cancel_sec_order_dispersion=True,
                       enable_csr=True, enable_isr=True, enable_space_charge=False, num_slices=50, use_monitors=False, enable_isr_on_ref_part=True, keep_data=False):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, length_dipole=length_dipole, field_dipole=field_dipole, R56=R56, 
                         cancel_chromaticity=cancel_chromaticity, cancel_sec_order_dispersion=cancel_sec_order_dispersion, 
                         enable_csr=enable_csr, enable_isr=enable_isr, enable_space_charge=enable_space_charge)
        
        # simulation options
        self.num_slices = num_slices
        self.use_monitors = use_monitors
        self.isr_on_ref_part = enable_isr_on_ref_part


    # ==================================================
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        "Track plasma-lens-based interstage using ImpactX."
        
        # re-perform the matching
        self.match()

        # get the lattice
        lattice = self.get_impactx_lattice()
        
        # run ImpactX
        from abel.wrappers.impactx.impactx_wrapper import run_impactx
        beam, self.evolution = run_impactx(lattice, beam0, nom_energy=self.nom_energy, verbose=False, runnable=runnable, save_beams=self.use_monitors, space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr, isr_on_ref_part=self.isr_on_ref_part)
        
        return super().track(beam, savedepth, runnable, verbose)
    

    # ==================================================
    @property
    def isr_on_ref_part(self) -> bool | None:
        """
        Whether `ISR will be applied to the reference particle <https://impactx.readthedocs.io/en/latest/usage/python.html#impactx.ImpactX.isr_on_ref_part>`_.
        to cause the reference particle to lose energy due to radiation. This 
        should be activated when the lattice optics, magnet settings, etc. 
        are chosen to account for radiative energy loss. This can prevent beam 
        centroid kicks.

        Only ``True`` if 
        :attr:`self.enable_isr <abel.Interstage.enable_isr>` and 
        :attr:`self.enable_isr_on_ref_part <abel.InterstagePlasmaLensImpactX.enable_isr_on_ref_part>` 
        are ``True``. 
        """
        return self._isr_on_ref_part
    @isr_on_ref_part.setter
    def isr_on_ref_part(self, enable_isr_on_ref_part : bool | None):
        if self.enable_isr and enable_isr_on_ref_part:
            self._isr_on_ref_part = True
        else:
            self._isr_on_ref_part = False
    _isr_on_ref_part = None
    

    # ==================================================
    def get_impactx_lattice(self):
        "Set up the ImpactX plasma-lens-based interstage lattice."
        
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
        
        # define dipole
        B_dip = self.field_dipole
        phi_dip = self.length_dipole*B_dip*SI.e/energy2momentum(self.nom_energy)
        dipole = elements.ExactSbend(ds=self.length_dipole, phi=np.rad2deg(phi_dip), B=B_dip, nslice=self.num_slices)

        # add lens offset for ISR mitigation
        if self.enable_isr and self.cancel_isr_kicks:
            dx_isr = self.lens_offset_isr_kick_mitigation()
        else:
            dx_isr = 0
            
        # define plasma lens
        ds_pl = self.length_plasma_lens/(self.num_slices+1)
        drift_slice_pl = elements.ExactDrift(ds=ds_pl, nslice=1)
        plasma_lens1 = [drift_slice_pl]
        plasma_lens2 = [drift_slice_pl]
        kl_lens = self.strength_plasma_lens
        tau_lens = self.nonlinearity_plasma_lens
        dxs = dx_isr*np.array([1, -1]) + np.array([self.lens1_offset_x, self.lens2_offset_x]) + np.random.normal(scale=self.jitter.lens_offset_x, size=2)
        dxps = np.random.normal(scale=self.jitter.lens_angle_x, size=2)
        dys = np.array([self.lens1_offset_y, self.lens2_offset_y]) + np.random.normal(scale=self.jitter.lens_offset_y, size=2)
        dyps = np.random.normal(scale=self.jitter.lens_angle_y, size=2)
        for i in range(self.num_slices):
            dl = ds_pl*(i+1) - self.length_plasma_lens/2
            plasma_lens1.append(elements.TaperedPL(k=kl_lens/self.num_slices, taper=tau_lens, dx=dxs[0]+dl*dxps[0], dy=dys[0]+dl*dyps[0]))
            plasma_lens1.append(drift_slice_pl)
            plasma_lens2.append(elements.TaperedPL(k=kl_lens/self.num_slices, taper=tau_lens, dx=dxs[1]+dl*dxps[1], dy=dys[1]+dl*dyps[1]))
            plasma_lens2.append(drift_slice_pl)
        
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
            plasma_lens1 = pl2

        # define first chicane dipole
        B_chic1 = self.field_chicane_dipole1
        phi_chic1 = self.length_chicane_dipole*B_chic1*(SI.e/energy2momentum(self.nom_energy))
        if abs(B_chic1) > 0:
            chicane_dipole1 = elements.ExactSbend(ds=self.length_chicane_dipole, phi=np.rad2deg(phi_chic1), B=B_chic1, nslice=self.num_slices)
        else:
            chicane_dipole1 = elements.ExactDrift(ds=self.length_chicane_dipole, nslice=self.num_slices)

        # define second chicane dipole
        B_chic2 = self.field_chicane_dipole2
        phi_chic2 = self.length_chicane_dipole*B_chic2*(SI.e/energy2momentum(self.nom_energy))
        if abs(B_chic2) > 0:
            chicane_dipole2 = elements.ExactSbend(ds=self.length_chicane_dipole, phi=np.rad2deg(phi_chic2), B=B_chic2, nslice=self.num_slices)
        else:
            chicane_dipole2 = elements.ExactDrift(ds=self.length_chicane_dipole, nslice=self.num_slices)

        # define sextupole (or gap)
        if self.cancel_sec_order_dispersion:
            dx_sext = np.random.normal(scale=self.jitter.sextupole_offset_x)
            dy_sext = np.random.normal(scale=self.jitter.sextupole_offset_y)
            half_sextupole_or_gap = elements.ExactMultipole(ds=self.length_central_gap_or_sextupole/2, k_normal=[0.,0.,self.strength_sextupole/self.length_central_gap_or_sextupole], k_skew=[0.,0.,0.], nslice=self.num_slices, dx=dx_sext, dy=dy_sext)
        else:
            half_sextupole_or_gap = elements.ExactDrift(ds=self.length_central_gap_or_sextupole/2, nslice=1)
        
        # specify the lattice sequence
        lattice.extend(gap)
        lattice.append(dipole)
        lattice.extend(gap)
        lattice.extend(plasma_lens1)
        lattice.extend(gap)
        lattice.append(chicane_dipole1)
        lattice.extend(gap)
        lattice.append(chicane_dipole2)
        lattice.extend(gap)
        lattice.append(half_sextupole_or_gap)
        if self.use_monitors:
            lattice.append(monitor)
        lattice.append(half_sextupole_or_gap)
        lattice.extend(gap)
        lattice.append(chicane_dipole2)
        lattice.extend(gap)
        lattice.append(chicane_dipole1)
        lattice.extend(gap)
        lattice.extend(plasma_lens2)
        lattice.extend(gap)
        lattice.append(dipole)
        lattice.extend(gap)

        # remove first and last monitor
        if lattice[0] == monitor:
            del lattice[0]
        if lattice[-1] == monitor:
            del lattice[-1]
                
        return lattice

        