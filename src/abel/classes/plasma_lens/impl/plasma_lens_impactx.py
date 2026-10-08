# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abel.classes.plasma_lens.plasma_lens import PlasmaLens
from abel.utilities.relativity import energy2momentum
import numpy as np
import scipy.constants as SI


class PlasmaLensImpactX(PlasmaLens):
    """
    Active plasma lens tracked with ImpactX.

    The lens is modelled as a drift-kick sequence of thin, transversely tapered
    plasma-lens elements (``impactx.elements.TaperedPL``), following the same
    construction as :class:`abel.InterstagePlasmaLensImpactX`.

    Parameters
    ----------
    length : [m] float
        Length of the plasma lens.

    radius : [m] float
        Radius of the plasma-lens capillary.

    current : [A] float
        Current through the plasma lens.

    rel_nonlinearity : float, optional
        Relative nonlinearity of the transverse field profile, defined as
        ``radius / Dx`` where ``Dx`` is the targeted horizontal dispersion.
        Defaults to 0 (a purely linear lens).

    num_slices : int, optional
        Number of thin-lens kicks used to represent the thick lens. Defaults
        to 30.

    use_apertures : bool, optional
        If ``True``, an elliptical aperture of half-width :attr:`radius` is
        applied at each end of the lens, so that charge outside the capillary
        is removed. Defaults to ``False``, which reproduces the behaviour of
        the other ``PlasmaLens`` implementations' unclipped field expansion.

    Notes
    -----
    ``TaperedPL`` applies a polynomial expansion of the plasma-lens field. It
    is only physical inside the capillary; particles outside :attr:`radius`
    receive an unphysical kick unless ``use_apertures=True``.
    """

    def __init__(self, length=None, radius=None, current=None, rel_nonlinearity=0, num_slices=30, use_apertures=False):

        super().__init__(length, radius, current)

        # set nonlinearity (defined as R/Dx)
        self.rel_nonlinearity = rel_nonlinearity

        # simulation options
        self.num_slices = num_slices
        self.use_apertures = use_apertures


    # ==================================================
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        "Track the plasma lens using ImpactX."

        # get the lattice
        lattice = self.get_impactx_lattice(beam0)

        # run ImpactX
        from abel.wrappers.impactx.impactx_wrapper import run_impactx
        beam, self.evolution = run_impactx(lattice, beam0, nom_energy=beam0.energy(), verbose=False, runnable=runnable)

        return super().track(beam, savedepth, runnable, verbose)


    # ==================================================
    def get_impactx_lattice(self, beam0):
        "Set up the ImpactX plasma-lens lattice."

        from impactx import elements

        # integrated focusing strength [1/m], signed by the beam charge
        # (k = L * g / (magnetic rigidity), see the ImpactX TaperedPL docs)
        strength = beam0.charge_sign() * self.get_focusing_gradient() * self.length * SI.e / energy2momentum(beam0.energy())

        # horizontal taper parameter [1/m], i.e. the inverse target dispersion
        taper = self.rel_nonlinearity / self.radius

        # drift-kick sequence: num_slices kicks separated by num_slices+1 drifts
        ds = self.length / (self.num_slices + 1)
        drift = elements.ExactDrift(ds=ds, nslice=1)

        lattice = [drift]
        for _ in range(self.num_slices):
            lattice.append(elements.TaperedPL(k=strength/self.num_slices, taper=taper, unit=0, dx=self.offset_x, dy=self.offset_y))
            lattice.append(drift)

        # clip charge outside the capillary
        if self.use_apertures:
            aperture = elements.Aperture(aperture_x=self.radius, aperture_y=self.radius, shape='elliptical')
            lattice = [aperture] + lattice + [aperture]

        return lattice


    # ==================================================
    def get_focusing_gradient(self):
        "Plasma-lens field gradient [T/m]."
        return SI.mu_0 * self.current / (2*np.pi * self.radius**2)
