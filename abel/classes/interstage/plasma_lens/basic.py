# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.interstage.plasma_lens import InterstagePlasmaLens
import scipy.constants as SI
import numpy as np

class InterstagePlasmaLensBasic(InterstagePlasmaLens):
    """
    Basic implementation of a plasma-lens-based interstage providing analytical
    phase-space rotation and longitudinal compression.

    The ``InterstagePlasmaLensBasic`` class extends :class:`InterstagePlasmaLens`
    to provide a minimal model for studying beam envelope evolution and compression
    effects without performing full lattice beam tracking. It applies a phase advance
    rotation to both transverse planes and a linear compression in longitudinal phase
    space by applying longitudinal dispersion using the specified ``R56`` value.

    Inherits all attributes from ``InterstagePlasmaLens``.

    Attributes
    ----------
    phase_advance : [rad] float
        Total phase-space rotation angle to apply to both transverse planes.
        Defaults to ``2*np.pi`` (one full betatron oscillation).
    """

    
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0, cancel_chromaticity=True, cancel_sec_order_dispersion=True, enable_csr=False, enable_isr=False, enable_space_charge=False, phase_advance=2*np.pi):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, length_dipole=length_dipole, field_dipole=field_dipole, R56=R56, cancel_chromaticity=cancel_chromaticity, cancel_sec_order_dispersion=cancel_sec_order_dispersion, enable_csr=enable_csr, enable_isr=enable_isr, enable_space_charge=enable_space_charge)
        
        self.phase_advance = phase_advance
    
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        """
        Track the input beam through the interstage lattice.

        The beam undergoes:
        
        1. Longitudinal compression using ``self.R56`` and nominal energy.
        2. Transverse phase-space rotation in both x and y planes by a total
           phase advance ``self.phase_advance`` radians, based on 
           ``self.beta0``.
        """
        
        # compress beam
        beam.compress(R_56=self.R56, nom_energy=self.nom_energy)
        
        # rotate transverse phase spaces (assumed achromatic)
        if callable(self.beta0):
            betas = self.beta0(beam.Es())
        else:
            betas = self.beta0
        theta = self.phase_advance
        xs_rotated = beam.xs()*np.cos(theta) + betas*beam.xps()*np.sin(theta)
        xps_rotated = -beam.xs()*np.sin(theta)/betas + beam.xps()*np.cos(theta)
        beam.set_xs(xs_rotated)
        beam.set_xps(xps_rotated)
        ys_rotated = beam.ys()*np.cos(theta) + betas*beam.yps()*np.sin(theta)
        yps_rotated = -beam.ys()*np.sin(theta)/betas + beam.yps()*np.cos(theta)
        beam.set_ys(ys_rotated)
        beam.set_yps(yps_rotated)
        
        return super().track(beam, savedepth, runnable, verbose)
        