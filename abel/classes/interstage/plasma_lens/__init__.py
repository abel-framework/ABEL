# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# Authors: C.A.Lindstrøm(1), J.B.B.Chen(1), O.G.Finnerud(1), D.Kalvik(1), E.Hørlyk(1), A.Huebl(2), K.N.Sjobak(1), E.Adli(1)
# Affiliations: 1) University of Oslo, 2) LBNL
# License: GPL-3.0-or-later

from abc import abstractmethod, ABC
from abel.classes.interstage import Interstage
from types import SimpleNamespace
import numpy as np
import scipy.constants as SI

class InterstagePlasmaLens(Interstage, ABC):
    """
    Abstract subclass of :class:`Interstage` implementing an achromatic 
    interstage lattice that uses nonlinear plasma lenses as the focusing 
    elements.

    This class defines parameters, matching procedures, and lattice composition 
    for its subclasses. It handles the optical and field-level configuration of 
    the lattice components and provides matching functions to ensure proper beam
    transport and chromatic correction.

    The layout of the first half of interstage lattice is: 

    [drift, dipole, drift, plasma lens, drift, chicane dipole 1, drift, 
    chicane dipol 2, drift, sextupole 3].

    The lattice is then repeated in the opposite order (excluding sextupole 3) 
    to form a mirror symmetric lattice.

    Inherits all attributes from :class:`Interstage`.

    Attributes
    ----------
    lens_radius : [m] float
        Plasma lens physical aperture (radius). Defaults to 2e-3.

    lens1_offset_x : [m] float
        x-offset of the first plasma lens. Defaults to 0.

    lens2_offset_x : [m] float
        x-offset of the second plasma lens. Defaults to 0.

    lens1_offset_y : [m] float
        y-offset of the first plasma lens. Defaults to 0.

    lens2_offset_y : [m] float
        y-offset of the second plasma lens. Defaults to 0.

    cancel_isr_kicks : bool
        Flag for mitigating ISR (incoherent synchrotron radiation) kicks on beam 
        centroid by offseting the plasma lenses. Defaults to ``False``.
    """

    # TODO: shouldn't use_apertures be passed to the constructor of the parent class?
    
    @abstractmethod
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0, lens_radius=2e-3, charge_sign=-1,
                 cancel_chromaticity=True, cancel_sec_order_dispersion=False, use_apertures=True, enable_csr=True, enable_isr=True, enable_space_charge=False):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, length_dipole=length_dipole, field_dipole=field_dipole, R56=R56, cancel_chromaticity=cancel_chromaticity, cancel_sec_order_dispersion=cancel_sec_order_dispersion, enable_csr=enable_csr, enable_isr=enable_isr, enable_space_charge=enable_space_charge, uses_plasma_lenses=True)

        # lens parameters
        self.lens_radius = lens_radius
        self.lens1_offset_x = 0
        self.lens2_offset_x = 0
        self.lens1_offset_y = 0
        self.lens2_offset_y = 0

        self.cancel_isr_kicks = False
        
        # length ratios
        self.length_ratio_gap = 0.025
        self.length_ratio_plasma_lens = 0.05
        self.length_ratio_chicane_dipole = 0.85
        self.length_ratio_central_gap_or_sextupole = 0.25

        # derivable (but also settable) parameters
        self._field_ratio_chicane_dipole1 = None
        self._field_ratio_chicane_dipole2 = None
        self._strength_plasma_lens = None # [1/m]
        self._nonlinearity_plasma_lens = None # [1/m]
        self._strength_sextupole = None # [1/m^2]

        
    
    ## OVERALL LENGTH
    
    # lattice length
    def get_length(self):
        """
        Compute the total length of the interstage by summing all lattice 
        elements.

        Returns
        -------
        total_length : [m] float
            Total geometric length of the plasma-lens interstage. Returns 
            ``None`` if ``length_dipole`` is not defined.
        """
        if self.length_dipole is not None:
            ls, *_ = self.matrix_lattice(k_lens=0, tau_lens=0, B_chic1=0, B_chic2=0, m_sext=0, half_lattice=False)
            return np.sum(ls)
        else:
            return None
    
    
    ## RATIO-DEFINED LENGTHS
    
    @property
    def length_gap(self) -> float:
        """
        The length of a drift section [m].
        """
        return self.length_dipole * self.length_ratio_gap
        
    @property
    def length_plasma_lens(self) -> float:
        """
        The length of a plasma lens [m].
        """
        return self.length_dipole * self.length_ratio_plasma_lens

    @property
    def length_chicane_dipole(self) -> float:
        """
        The length of a chicane dipole [m].
        """
        return self.length_dipole * self.length_ratio_chicane_dipole

    @property
    def length_central_gap_or_sextupole(self) -> float:
        """
        Length of the central element, either a gap or a sextupole [m].
        """
        return self.length_dipole * self.length_ratio_central_gap_or_sextupole

    
    ## STRENGTH VALUES

    @property
    def strength_plasma_lens(self) -> float:
        """
        Effective integrated focusing strength of the plasma lens (equivalent to 
        gql/p, where g [T/m] is the gradient of the focusing magnetic fields, q 
        is the particle charge, l is the length of the lens, and p [kg m/s] is 
        the nominal particle momentum).


        Returns
        -------
        strength_plasma_lens : [1/m] float
            Effective integrated focusing strength of the plasma lens, matched via 
            :meth:`InterstagePlasmaLens.match_beta_function`.
        """
        if self._strength_plasma_lens is None:
            self.match_beta_function()
        return self._strength_plasma_lens

    @property
    def nonlinearity_plasma_lens(self) -> float:
        """
        Plasma lens nonlinearity (transverse taper coefficient).

        Returns
        -------
        nonlinearity_plasma_lens : [1/m] float
            Plasma lens nonlinearity focusing term, matched via 
            :meth:`InterstagePlasmaLens.match_chromatic_amplitude`.
        """
        if self._nonlinearity_plasma_lens is None:
            self.match_chromatic_amplitude()
        return self._nonlinearity_plasma_lens
        
    @property
    def strength_sextupole(self) -> float:
        """
        Sextupole strength.

        Returns
        -------
        strength_sextupole : [m^-2] float
            Sextupole strength, matched via 
            :meth:`InterstagePlasmaLens.match_second_order_dispersion`.
        """
        if self._strength_sextupole is None:
            self.match_second_order_dispersion()
        return self._strength_sextupole


    
    ## FIELD VALUES
    
    @property
    def field_gradient_plasma_lens(self) -> float:
        "Plasma-lens field gradient [T/m]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.charge_sign*self.strength_plasma_lens*p0/(SI.e*self.length_plasma_lens)

    @property
    def field_chicane_dipole1(self) -> float:
        """
        Field of the outer chicane dipoles.

        Returns
        -------
        field_chicane_dipole1 : [T] float
            Magnetic field strength of chicane dipole 1, determined via 
            :meth:`InterstagePlasmaLens.match_dispersion_and_R56`.
        """
        if self._field_ratio_chicane_dipole1 is None:
            self.match_dispersion_and_R56()
        return self.field_dipole * self._field_ratio_chicane_dipole1

    @property
    def field_chicane_dipole2(self) -> float:
        """
        Field of the inner chicane dipoles, matched to cancel dispersion.

        Returns
        -------
        field_chicane_dipole2 : [T] float
            Magnetic field strength of chicane dipole 2, determined via 
            :meth:`InterstagePlasmaLens.match_dispersion_and_R56`.
        """
        if self._field_ratio_chicane_dipole2 is None:
            self.match_dispersion_and_R56()
        return self.field_dipole * self._field_ratio_chicane_dipole2

    @property
    def field_gradient_sextupole(self) -> float:
        "Sextupole field gradient [T/m^2]."
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.charge_sign*self.strength_sextupole*p0/(SI.e*self.length_central_gap_or_sextupole)


    ## ISR KICK MITIGATION
    
    def lens_offset_isr_kick_mitigation(self):
        """
        Estimate the transverse lens offset required to cancel ISR (Incoherent 
        Synchrotron Radiation) kicks.

        The computed offset value applies to the first plasma lens, with the 
        second lens offset being in the opposite direction.

        Returns
        -------
        dx : [m] float
            Estimated lens offset for ISR kick mitigation.
        """
        pfit = [2.36256046e-08, 1.09612466e-07, 2.70442278e-07, -1.47004050e-07, 5.08857498e-08]
        R56_scaling = SI.c**2*(self.field_dipole**2*self.length_dipole**3/self.nom_energy**2)
        dx_scaling = self.field_dipole**3*self.length_dipole**3
        dx = np.polyval(pfit, self.R56/R56_scaling)*dx_scaling
        
        return dx
    
    
    ## MATRIX LATTICE

    # full lattice 
    def matrix_lattice(self, k_lens=None, tau_lens=None, B_chic1=None, B_chic2=None, m_sext=None, half_lattice=False, orbit_only=False):
        """
        Return the optical lattice representation of the interstage.

        Parameters
        ----------
        k_lens : [m^-2] float, optional
            Effective focusing strength of the plasma lens (equivalent to 
            gq/p, where g [T/m] is the gradient of the focusing magnetic fields, 
            q is the particle charge, and p [kg m/s] is the nominal particle 
            momentum). Defaults to 
            :attr:`InterstagePlasmaLens.strength_plasma_lens` / :attr:`InterstagePlasmaLens.length_plasma_lens`.

        tau_lens : [m^-1] float, optional
            Transverse taper coefficient representing plasma lens nonlinearity.
            Defaults to :attr:`InterstagePlasmaLens.nonlinearity_plasma_lens`.

        B_chic1 : [T] float, optional
            Field strength of the outer chicane dipoles. Defaults to 
            :attr:`InterstagePlasmaLens.field_chicane_dipole1`.

        B_chic2 : [T] float, optional
            Field strength of the inner chicane dipoles. Defaults to 
            :attr:`InterstagePlasmaLens.field_chicane_dipole2`.

        m_sext : [m^-3] float, optional
            Sextupole normalized strength. Defaults to 
            :attr:`InterstagePlasmaLens.strength_sextupole` / :attr:`InterstagePlasmaLens.length_central_gap_or_sextupole`.

        half_lattice : bool, optional
            If ``True``, returns only half of the symmetric lattice. Defaults to 
            ``False``.

        orbit_only : bool, optional
            If ``True``, sets the plasma lens transverse taper coefficient 
            ``taus`` and sextupole strength arrays ``ms`` to all zeros. Defaults 
            to ``False``.

        Returns
        -------
        ls : [m] 1D float ndarray
            Lattice element lengths.

        inv_rhos : [m^-1] 1D float ndarray
            Inverse bending radii.

        ks : [m^-2] 1D float ndarray
            Plasma lens focusing strengths equivalent to gq/p, where g [T/m] is 
            the gradient of the focusing magnetic fields, q is the particle 
            charge, and p [kg m/s] is the nominal particle momentum.

        ms : [m^-3] 1D float ndarray
            Sextupole strengths.

        taus : [m^-1] 1D float ndarray
            Plasma lens transverse taper coefficients.
        """
        
        # fast solution for orbit only
        if orbit_only:
            tau_lens, m_sext = 0.0, 0.0
            
        # element length array
        dL = self.length_gap
        ls = np.array([dL, self.length_dipole, dL, self.length_plasma_lens, dL, 
                       self.length_chicane_dipole, dL, self.length_chicane_dipole, dL, self.length_central_gap_or_sextupole/2])
        
        # bending strength array
        if B_chic1 is None:
            B_chic1 = self.field_chicane_dipole1
        if B_chic2 is None:
            B_chic2 = self.field_chicane_dipole2
        Bs = np.array([0, self.field_dipole, 0, 0, 0, B_chic1, 0, B_chic2, 0, 0])
        
        from abel.utilities.relativity import energy2momentum
        inv_rhos = -self.charge_sign * Bs * SI.e / energy2momentum(self.nom_energy)
        
        # focusing strength array
        if k_lens is None:
            k_lens = self.strength_plasma_lens/self.length_plasma_lens
        ks = np.array([0, 0, 0, k_lens, 0, 0, 0, 0, 0, 0])
        
        # sextupole strength array
        if m_sext is None:
            if self.cancel_sec_order_dispersion:
                m_sext = self.strength_sextupole/self.length_central_gap_or_sextupole
            else:
                m_sext = 0
        ms = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, m_sext])

        # plasma-lens transverse taper array
        if tau_lens is None:
            tau_lens = self.nonlinearity_plasma_lens
        taus = np.array([0, 0, 0, tau_lens, 0, 0, 0, 0, 0, 0])
        
        # mirror symmetrize the lattice
        if not half_lattice:
            ls = np.append(np.append(ls[:-1], 2*ls[-1]), np.flip(ls[:-1]))
            inv_rhos = np.append(np.append(inv_rhos[:-1], inv_rhos[-1]), np.flip(inv_rhos[:-1]))
            ks = np.append(np.append(ks[:-1], ks[-1]), np.flip(ks[:-1]))
            ms = np.append(np.append(ms[:-1], ms[-1]), np.flip(ms[:-1]))
            taus = np.append(np.append(taus[:-1], taus[-1]), np.flip(taus[:-1]))
        
        return ls, inv_rhos, ks, ms, taus
        

    
    ## MATCHING
    
    def match_beta_function(self):
        """
        Match the beta function by adjusting the plasma-lens focusing strength.

        Returns
        -------
        None
            Updates ``self._strength_plasma_lens`` in place.
        """
        
        # minimizer function for beta matching (central alpha function is zero)
        from abel.utilities.beam_physics import evolve_beta_function
        def minfun_beta(params):
            ls, _, ks, _, _ = self.matrix_lattice(k_lens=params[0], tau_lens=0, B_chic1=0, B_chic2=0, m_sext=0, half_lattice=True)
            _, alpha, _ = evolve_beta_function(ls, ks, self.beta0, fast=True) 
            return alpha**2
    
        # initial guess for the lens strength
        f = 1/(1/(self.length_dipole + self.length_plasma_lens/2 + 2*self.length_gap) + 1/(2*self.length_chicane_dipole + self.length_central_gap_or_sextupole/2 + self.length_plasma_lens/2 + 3*self.length_gap))
        k_lens0 = 1/(f*self.length_plasma_lens)
        
        # match the beta function
        from scipy.optimize import minimize
        result_beta = minimize(minfun_beta, k_lens0, tol=1e-16, options={'maxiter': 200})
        self._strength_plasma_lens = result_beta.x[0]*self.length_plasma_lens

    
    def match_dispersion_and_R56(self, high_res=False):
        """
        Cancelling the dispersion and matching the R56 by adjusting the chicane 
        dipole fields.

        Parameters
        ----------
        high_res : bool, optional
            Enables higher-resolution computation of ``R56`` and dispersion evolution.
            Defaults to ``False``.

        Returns
        -------
        None :
            Updates ``self._field_ratio_chicane_dipole1`` and 
            ``self._field_ratio_chicane_dipole2``.
        """
        
        nom_R56 = self.R56
            
        # normalizing scale for the merit function
        Dpx_scale = self.length_dipole*self.field_dipole*SI.c/self.nom_energy
        R56_scale = self.length_dipole**3*self.field_dipole**2*SI.c**2/self.nom_energy**2
        
        # minimizer function for dispersion (central dispersion prime is zero)
        from abel.utilities.beam_physics import evolve_dispersion, evolve_R56
        def minfun_dispersion_R56(params):
            ls, inv_rhos, ks, _, _ = self.matrix_lattice(tau_lens=0, B_chic1=params[0], B_chic2=params[1], m_sext=0, half_lattice=True)
            _, Dpx_mid, evolution_disp = evolve_dispersion(ls, inv_rhos, ks, fast=True) 
            R56_mid, _ = evolve_R56(ls, inv_rhos, ks, high_res=high_res, fast=True, evolution_disp=evolution_disp)
            return (Dpx_mid/Dpx_scale)**2 + ((R56_mid - nom_R56/2)/R56_scale)**2

        # initial guess for the chicane dipole fields
        B_chic1_guess = self.field_dipole/4
        B_chic2_guess = -self.field_dipole/4
        
        # match the beta function
        from scipy.optimize import minimize
        result_dispersion_R56 = minimize(minfun_dispersion_R56, [B_chic1_guess, B_chic2_guess], tol=1e-8, options={'maxiter': 50})
        self._field_ratio_chicane_dipole1 = result_dispersion_R56.x[0]/self.field_dipole
        self._field_ratio_chicane_dipole2 = result_dispersion_R56.x[1]/self.field_dipole
    
    
    def match_chromatic_amplitude(self):
        """
        Match the chromatic amplitude by tuning the plasma-lens transverse taper 
        coefficient.

        Returns
        -------
        None : 
            Updates ``self._nonlinearity_plasma_lens`` in place. If 
            :attr:`Interstage.cancel_chromaticity` is ``False``, 
            ``self._nonlinearity_plasma_lens`` is set to zero.
        """
        
        # stop if nonlinearity is turned off
        if not self.cancel_chromaticity:
            self._nonlinearity_plasma_lens = 0.0
            return
        
        # normalizing scale for the merit function
        W_scale = 2*self.length_dipole/self.beta0
        
        # minimizer function for beta matching (central alpha function is zero)
        from abel.utilities.beam_physics import evolve_chromatic_amplitude
        def minfun_W(params):
            ls, inv_rhos, ks, ms, taus = self.matrix_lattice(tau_lens=params[0], m_sext=0, half_lattice=True)
            W_mid, _ = evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, self.beta0, fast=True) 
            return (W_mid/W_scale)**2
        
        # calculate dispersion and dispersion prime in the lens
        from abel.utilities.relativity import energy2momentum
        Dpx_lens = -self.charge_sign*self.length_dipole*self.field_dipole*SI.e/energy2momentum(self.nom_energy)
        Dx_lens = Dpx_lens*(self.length_dipole/2 + self.length_gap + self.length_plasma_lens/2)
        tau_lens0 = -1/Dx_lens
            
        # match the beta function
        from scipy.optimize import minimize
        result_W = minimize(minfun_W, tau_lens0, tol=1e-16, options={'maxiter': 100})
        self._nonlinearity_plasma_lens = result_W.x[0]
    
        
    def match_second_order_dispersion(self):
        """
        Match and cancel the second-order dispersion by adjusting the sextupole strength.

        Returns
        -------
        None : 
            Updates ``self._strength_sextupole`` in place. If 
            :attr:`Interstage.cancel_sec_order_dispersion` is ``False``, sets 
            ``self._strength_sextupole`` to zero.
        """

        # stop if nonlinearity is turned off
        if not self.cancel_sec_order_dispersion:
            self._strength_sextupole = 0.0
            return

        # normalizing scale for the merit function
        DDpx_scale = self.length_dipole*self.field_dipole*SI.c/self.nom_energy
        
        # minimizer function for second-order dispersion (central second-order dispersion prime is zero)
        from abel.utilities.beam_physics import evolve_second_order_dispersion
        def minfun_second_order_dispersion(params):
            ls, inv_rhos, ks, ms, taus = self.matrix_lattice(m_sext=params[0], half_lattice=True)
            _, DDpx, _ = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=True) 
            return (DDpx/DDpx_scale)**2

        # guesstimate the sextupole strength (starting point for optimization)
        Dx_scale = -self.length_dipole**2*self.field_dipole*SI.c/self.nom_energy/2
        m_sext0 = 1.11/(Dx_scale*self.length_central_gap_or_sextupole*self.length_dipole)
    
        # match the beta function
        from scipy.optimize import minimize
        result_dispersion = minimize(minfun_second_order_dispersion, m_sext0, method='Nelder-Mead', tol=1e-20, options={'maxiter': 50})
        self._strength_sextupole = result_dispersion.x[0]*self.length_central_gap_or_sextupole
        
    
    
    ## PRINT INFO

    def print_summary(self):
        """
        Print a formatted summary of the interstage configuration, including all
        optical element lengths, fields, and total lattice parameters.

        Returns
        -------
        None : 
            Prints the formatted lattice summary to the console.
        """
        
        print('------------------------------------------------')
        print(f'Main dipole (2x):          {self.length_dipole:.3f} m,  B = {self.field_dipole:.2f} T')
        print(f'Plasma lens (2x):          {self.length_plasma_lens:.3f} m,  g = {self.field_gradient_plasma_lens:.1f} T/m')
        print(f'Outer chicane dipole (2x): {self.length_chicane_dipole:.3f} m,  B = {self.field_chicane_dipole1:.3f} T')
        print(f'Inner chicane dipole (2x): {self.length_chicane_dipole:.3f} m,  B = {self.field_chicane_dipole2:.3f} T')
        if self.cancel_sec_order_dispersion:
            print(f'Sextupole:                 {self.length_central_gap_or_sextupole:.3f} m,  m = {self.field_gradient_sextupole:.1f} T/m^2')
            print(f'Gaps (10x):                {self.length_gap:.3f} m')
        else:
            print(f'Central gap:               {self.length_central_gap_or_sextupole + 2*self.length_gap:.3f} m')
            print(f'Other gaps (8x):           {self.length_gap:.3f} m')
        
        print('------------------------------------------------')
        print(f'             Total length: {self.get_length():.3f} m')
        print(f'         Total bend angle:           {np.rad2deg(self.total_bend_angle()):.2f} deg')
        print('------------------------------------------------')

    