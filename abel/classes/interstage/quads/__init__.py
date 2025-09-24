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

class InterstageQuads(Interstage, ABC):
    
    @abstractmethod
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0, polarity_quads=1, beta_ratio_central = 2, 
                 use_apertures=True, cancel_chromaticity=True, cancel_sec_order_dispersion=True,
                 enable_csr=True, enable_isr=True, enable_space_charge=False, charge_sign=-1):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, length_dipole=length_dipole, field_dipole=field_dipole, R56=R56, cancel_chromaticity=cancel_chromaticity, cancel_sec_order_dispersion=cancel_sec_order_dispersion, enable_csr=enable_csr, enable_isr=enable_isr, enable_space_charge=enable_space_charge, uses_plasma_lenses=False)

        # main parameters
        self.polarity_quads = polarity_quads
        self.beta_ratio_central = beta_ratio_central

        # length ratios
        self.length_ratio_gap = 0.025
        self.length_ratio_quad_gap_or_sextupole = 0.35
        self.length_ratio_central_gap_or_sextupole = 0.35
        self.length_ratio_quadrupole = 0.50
        self.length_ratio_chicane_dipole = 1.2
        
        # derivable (but also settable) parameters
        self._field_ratio_chicane_dipole1 = None
        self._field_ratio_chicane_dipole2 = None
        self._strength_quadrupole1 = None # [1/m]
        self._strength_quadrupole2 = None # [1/m]
        self._strength_quadrupole3 = None # [1/m]
    
        self._strength_sextupole1 = None # [1/m]
        self._strength_sextupole2 = None # [1/m]
        self._strength_sextupole3 = None # [1/m]
        

    ## OVERALL LENGTH
    
    # lattice length
    def get_length(self):
        if self.length_dipole is not None:
            ls, *_ = self.matrix_lattice(k1=0, k2=0, k3=0, B_chic1=0, B_chic2=0, m1=0, m2=0, m3=0, half_lattice=False)
            return np.sum(ls)
        else:
            return None
    
    
    ## RATIO-DEFINED PARAMETERS
    
    @property
    def length_gap(self):
        return self.length_dipole * self.length_ratio_gap

    @property
    def length_quad_gap_or_sextupole(self):
        return self.length_dipole * self.length_ratio_quad_gap_or_sextupole

    @property
    def length_central_gap_or_sextupole(self):
        return self.length_dipole * self.length_ratio_central_gap_or_sextupole
        
    @property
    def length_quadrupole(self):
        return self.length_dipole * self.length_ratio_quadrupole

    @property
    def length_chicane_dipole(self) -> float:
        return self.length_dipole * self.length_ratio_chicane_dipole

    ## DERIVED PARAMETERS

    @property
    def field_chicane_dipole1(self) -> float:
        if self._field_ratio_chicane_dipole1 is None:
            self.match_dispersion_and_R56()
        return self.field_dipole * self._field_ratio_chicane_dipole1

    @property
    def field_chicane_dipole2(self) -> float:
        if self._field_ratio_chicane_dipole2 is None:
            self.match_dispersion_and_R56()
        return self.field_dipole * self._field_ratio_chicane_dipole2

        
    @property
    def strength_quadrupole1(self) -> float:
        if self._strength_quadrupole1 is None:
            self.match_beta_function()
        return self._strength_quadrupole1

    @property
    def field_gradient_quadrupole1(self) -> float:
        "First quadrupole field gradient [T/m]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_quadrupole1*p0/(SI.e*self.length_quadrupole)
   
    @property
    def strength_quadrupole2(self) -> float:
        if self._strength_quadrupole2 is None:
            self.match_beta_function()
        return self._strength_quadrupole2

    @property
    def field_gradient_quadrupole2(self) -> float:
        "Second quadrupole field gradient [T/m]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_quadrupole2*p0/(SI.e*self.length_quadrupole)

    @property
    def strength_quadrupole3(self) -> float:
        if self._strength_quadrupole3 is None:
            self.match_beta_function()
        return self._strength_quadrupole3

    @property
    def field_gradient_quadrupole3(self) -> float:
        "Third quadrupole field gradient [T/m]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_quadrupole3*p0/(SI.e*self.length_quadrupole)


    @property
    def strength_sextupole1(self) -> float:
        if self._strength_sextupole1 is None:
            self.match_chromatic_amplitude()
        return self._strength_sextupole1

    @property
    def strength_sextupole2(self) -> float:
        if self._strength_sextupole2 is None:
            self.match_chromatic_amplitude()
        return self._strength_sextupole2
        
    @property
    def strength_sextupole3(self) -> float:
        if self._strength_sextupole3 is None:
            self.match_second_order_dispersion()
        return self._strength_sextupole3
    
    @property
    def field_gradient_sextupole1(self) -> float:
        "Sextupole 1 field gradient [T/m^2]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_sextupole1*p0/(SI.e*self.length_quad_gap_or_sextupole)

    @property
    def field_gradient_sextupole2(self) -> float:
        "Sextupole 2 field gradient [T/m^2]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_sextupole2*p0/(SI.e*self.length_quad_gap_or_sextupole)

    @property
    def field_gradient_sextupole3(self) -> float:
        "Central sextupole field gradient [T/m^2]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.strength_sextupole3*p0/(SI.e*self.length_central_gap_or_sextupole)

    
    ## MATRIX LATTICE

    # full lattice 
    def matrix_lattice(self, k1=None, k2=None, k3=None, B_chic1=None, B_chic2=None, m1=None, m2=None, m3=None, half_lattice=False, orbit_only=False):
        
        # fast solution for orbit only
        if orbit_only:
            m1, m2, m3 = 0.0, 0.0, 0.0
            
        # element length array
        dL = self.length_gap
        ls = np.array([dL, self.length_dipole, dL, self.length_quadrupole, dL, self.length_quadrupole, dL, self.length_quad_gap_or_sextupole, dL, self.length_quadrupole, dL, self.length_quad_gap_or_sextupole, dL, self.length_chicane_dipole, dL, self.length_chicane_dipole, dL, self.length_central_gap_or_sextupole/2])
        
        # bending strength array
        if B_chic1 is None:
            B_chic1 = self.field_chicane_dipole1
        if B_chic2 is None:
            B_chic2 = self.field_chicane_dipole2
        Bs = np.array([0, self.field_dipole, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, B_chic1, 0, B_chic2, 0, 0])
        
        from abel.utilities.relativity import energy2momentum
        inv_rhos = -self.charge_sign*Bs * SI.e / energy2momentum(self.nom_energy)
        
        # focusing strength array
        if k1 is None:
            k1 = self.strength_quadrupole1/self.length_quadrupole
        if k2 is None:
            k2 = self.strength_quadrupole2/self.length_quadrupole
        if k3 is None:
            k3 = self.strength_quadrupole3/self.length_quadrupole
        ks = np.array([0, 0, 0, k1, 0, k2, 0, 0, 0, k3, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # sextupole strength array
        if m1 is None:
            m1 = self.strength_sextupole1/self.length_quad_gap_or_sextupole
        if m2 is None:
            m2 = self.strength_sextupole2/self.length_quad_gap_or_sextupole
        if m3 is None:
            m3 = self.strength_sextupole3/self.length_central_gap_or_sextupole
        ms = np.array([0, 0, 0, 0, 0, 0, 0, m1, 0, 0, 0, m2, 0, 0, 0, 0, 0, m3])

        # plasma-lens transverse taper array
        taus = np.zeros_like(ls)
        
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
        "Matching the beta function by adjusting the quadrupole strengths."
        
        # minimizer function for beta matching (central alpha function is zero)
        from abel.utilities.beam_physics import evolve_beta_function
        def minfun_beta(params):
            ls, inv_rhos, ks, _, _ = self.matrix_lattice(k1=params[0], k2=params[1], k3=params[2], B_chic1=0, B_chic2=0, m1=0, m2=0, m3=0, half_lattice=True)
            beta_x, alpha_x, _ = evolve_beta_function(ls, ks, self.beta0, inv_rhos=inv_rhos, fast=True) 
            beta_y, alpha_y, _ = evolve_beta_function(ls, -ks, self.beta0, fast=True)
            return alpha_x**2 + alpha_y**2 + (beta_x/beta_y-float(self.beta_ratio_central)**np.sign(self.polarity_quads))**2+ (max(beta_x/10, self.beta0)/self.beta0-1)**2 + (max(beta_y/10, self.beta0)/self.beta0-1)**2

        # initial guess for the quad strength
        k1_guess = 2*self.polarity_quads/(self.length_dipole*self.length_quadrupole)
        k2_guess = -2*self.polarity_quads/(self.length_dipole*self.length_quadrupole)
        k3_guess = 0.8*self.polarity_quads/(self.length_dipole*self.length_quadrupole)
        
        # match the beta function
        from scipy.optimize import minimize
        result_beta = minimize(minfun_beta, [k1_guess, k2_guess, k3_guess], tol=1e-20, options={'maxiter': 200})
        self._strength_quadrupole1 = result_beta.x[0]*self.length_quadrupole
        self._strength_quadrupole2 = result_beta.x[1]*self.length_quadrupole
        self._strength_quadrupole3 = result_beta.x[2]*self.length_quadrupole

    
    def match_dispersion_and_R56(self, high_res=False):
        "Cancelling the dispersion and matching the R56 by adjusting the chicane dipoles."
        
        # assume negative R56
        nom_R56 = self.R56

        # normalizing scale for the merit function
        Dpx_scale = self.length_dipole*self.field_dipole*SI.c/self.nom_energy
        R56_scale = self.length_dipole**3*self.field_dipole**2*SI.c**2/self.nom_energy**2
        
        # minimizer function for dispersion (central dispersion prime is zero)
        from abel.utilities.beam_physics import evolve_dispersion, evolve_R56
        def minfun_dispersion_R56(params):
            ls, inv_rhos, ks, _, _ = self.matrix_lattice(B_chic1=params[0], B_chic2=params[1], m1=0, m2=0, m3=0, half_lattice=True)
            _, Dpx_mid, _ = evolve_dispersion(ls, inv_rhos, ks, fast=True)
            R56_mid, _ = evolve_R56(ls, inv_rhos, ks, high_res=high_res)
            return (Dpx_mid/Dpx_scale)**2 + ((R56_mid - nom_R56/2)/R56_scale)**2

        # initial guess for the chicane dipole fields
        B_chic1_guess = self.field_dipole/2
        B_chic2_guess = -self.field_dipole/2
        
        # match the beta function
        from scipy.optimize import minimize
        result_dispersion_R56 = minimize(minfun_dispersion_R56, [B_chic1_guess, B_chic2_guess], tol=1e-16, options={'maxiter': 50})
        self._field_ratio_chicane_dipole1 = result_dispersion_R56.x[0]/self.field_dipole
        self._field_ratio_chicane_dipole2 = result_dispersion_R56.x[1]/self.field_dipole

    
    def match_chromatic_amplitude(self):
        "Cancelling the chromaticity with the two first sextupoles."
        
        # stop if nonlinearity is turned off
        if not self.cancel_chromaticity:
            self._strength_sextupole1 = 0.0
            self._strength_sextupole2 = 0.0
            return
        
        # normalizing scale for the merit function
        W_scale = 2*self.length_dipole/self.beta0
        
        # minimizer function for beta matching (central alpha function is zero)
        from abel.utilities.beam_physics import evolve_chromatic_amplitude
        def minfun_Wxy(params):
            ls, inv_rhos, ks, ms, taus = self.matrix_lattice(m1=params[0], m2=params[1], m3=0, half_lattice=False)
            Wx_mid, _ = evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, self.beta0, fast=True, bending_plane=True) 
            Wy_mid, _ = evolve_chromatic_amplitude(ls, inv_rhos, ks, ms, taus, self.beta0, fast=True, bending_plane=False) 
            return (Wx_mid/W_scale)**2 + (Wy_mid/W_scale)**2
        
        # make estimate
        m1_factor = 90 - 32*(1-self.polarity_quads)/2
        m2_factor = 90 + 36*(1-self.polarity_quads)/2
        m1_guess = m1_factor*self.polarity_quads/(self.length_quad_gap_or_sextupole*self.length_dipole*self.field_dipole)
        m2_guess = -m2_factor*self.polarity_quads/(self.length_quad_gap_or_sextupole*self.length_dipole*self.field_dipole)
        
        # match the beta function
        from scipy.optimize import minimize
        result_Wxy = minimize(minfun_Wxy, [m1_guess, m2_guess], method='Nelder-Mead', 
                              options={'maxiter': 200, 'fatol': 1e-10, 'xatol': m1_guess*1e-4})
        self._strength_sextupole1 = result_Wxy.x[0]*self.length_quad_gap_or_sextupole
        self._strength_sextupole2 = result_Wxy.x[1]*self.length_quad_gap_or_sextupole
            
    
    def match_second_order_dispersion(self):
        "Cancelling the second-order dispersion with the central sextupole."
        
        # stop if nonlinearity is turned off
        if not self.cancel_sec_order_dispersion:
            self._strength_sextupole3 = 0.0
            return

        # normalizing scale for the merit function
        DDpx_scale = self.length_dipole*self.field_dipole*SI.c/self.nom_energy
        
        # minimizer function for second-order dispersion (central second-order dispersion prime is zero)
        from abel.utilities.beam_physics import evolve_second_order_dispersion
        def minfun_second_order_dispersion(params):
            ls, inv_rhos, ks, ms, taus = self.matrix_lattice(m3=params[0], half_lattice=True)
            _, DDpx, _ = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus, fast=True) 
            return (DDpx/DDpx_scale)**2
    
        # guesstimate the sextupole strength (starting point for optimization)
        m3_guess = 40*self.polarity_quads/(self.length_central_gap_or_sextupole*self.length_dipole*self.field_dipole)

        # match the beta function
        from scipy.optimize import minimize
        result_dispersion = minimize(minfun_second_order_dispersion, m3_guess, method='Nelder-Mead', 
                                     options={'maxiter': 100, 'fatol': 1e-10, 'xatol': m3_guess*1e-4})
        self._strength_sextupole3 = result_dispersion.x[0]*self.length_central_gap_or_sextupole
        
    
    
    ## PRINT INFO

    def print_summary(self):
        print('------------------------------------------------')
        print(f'Main dipoles (2x):          {self.length_dipole:.3f} m,  B = {self.field_dipole:.2f} T')
        print(f'Outer chicane dipoles (2x): {self.length_chicane_dipole:.3f} m,  B = {self.field_chicane_dipole1:.3f} T')
        print(f'Inner chicane dipoles (2x): {self.length_chicane_dipole:.3f} m,  B = {self.field_chicane_dipole2:.3f} T')
        print(f'Outer quadrupoles (2x):     {self.length_quadrupole:.3f} m,  g = {self.field_gradient_quadrupole1:.1f} T/m')
        print(f'Middle quadrupoles (2x):    {self.length_quadrupole:.3f} m,  g = {self.field_gradient_quadrupole2:.1f} T/m')
        print(f'Inner quadrupoles (2x):     {self.length_quadrupole:.3f} m,  g = {self.field_gradient_quadrupole3:.1f} T/m')
        if abs(self.field_gradient_sextupole1)>0 or abs(self.field_gradient_sextupole2)>0:
            print(f'Outer sextupoles (2x):      {self.length_quad_gap_or_sextupole:.3f} m,  m = {self.field_gradient_sextupole1:.1f} T/m^2')
            print(f'Inner sextupoles (2x):      {self.length_quad_gap_or_sextupole:.3f} m,  m = {self.field_gradient_sextupole2:.1f} T/m^2')
        else:
            print(f'Large gaps (4x):            {self.length_quad_gap_or_sextupole:.3f} m')
        if abs(self.field_gradient_sextupole3)>0:
            print(f'Central sextupole (1x):     {self.length_central_gap_or_sextupole:.3f} m,  m = {self.field_gradient_sextupole3:.1f} T/m^2')
        else:
            print(f'Central gap (1x):           {self.length_central_gap_or_sextupole:.3f} m')
        print(f'Other gaps (18x):           {self.length_gap:.3f} m')
        
        print('------------------------------------------------')
        print(f'             Total length: {self.get_length():.3f} m')
        print(f'         Total bend angle:           {np.rad2deg(self.total_bend_angle()):.2f} deg')
        print('------------------------------------------------')
    