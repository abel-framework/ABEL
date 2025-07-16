from abc import abstractmethod, ABC
from abel.classes.interstage import Interstage
from types import SimpleNamespace
import numpy as np
import scipy.constants as SI

class InterstagePlasmaLens(Interstage, ABC):
    
    @abstractmethod
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, R56=0, lens_radius=2e-3, charge_sign=-1,
                 cancel_chromaticity=True, cancel_sec_order_dispersion=False, use_apertures=True, enable_csr=True, enable_isr=True, enable_space_charge=False):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, length_dipole=length_dipole, field_dipole=field_dipole, R56=R56, cancel_chromaticity=cancel_chromaticity, cancel_sec_order_dispersion=cancel_sec_order_dispersion, enable_csr=enable_csr, enable_isr=enable_isr, enable_space_charge=enable_space_charge, uses_plasma_lenses=True)

        # lens parameters
        self.lens_radius = lens_radius
        
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
        if self.length_dipole is not None:
            ls, *_ = self.matrix_lattice(k_lens=0, tau_lens=0, B_chic1=0, B_chic2=0, m_sext=0, half_lattice=False)
            return np.sum(ls)
        else:
            return None
    
    
    ## RATIO-DEFINED LENGTHS
    
    @property
    def length_gap(self) -> float:
        return self.length_dipole * self.length_ratio_gap
        
    @property
    def length_plasma_lens(self) -> float:
        return self.length_dipole * self.length_ratio_plasma_lens

    @property
    def length_chicane_dipole(self) -> float:
        return self.length_dipole * self.length_ratio_chicane_dipole

    @property
    def length_central_gap_or_sextupole(self) -> float:
        return self.length_dipole * self.length_ratio_central_gap_or_sextupole

    
    ## STRENGTH VALUES

    @property
    def strength_plasma_lens(self) -> float:
        if self._strength_plasma_lens is None:
            self.match_beta_function()
        return self._strength_plasma_lens

    @property
    def nonlinearity_plasma_lens(self) -> float:
        if self._nonlinearity_plasma_lens is None:
            self.match_chromatic_amplitude()
        return self._nonlinearity_plasma_lens
        
    @property
    def strength_sextupole(self) -> float:
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
        if self._field_ratio_chicane_dipole1 is None:
            self.match_dispersion_and_R56()
        return self.field_dipole * self._field_ratio_chicane_dipole1

    @property
    def field_chicane_dipole2(self) -> float:
        if self._field_ratio_chicane_dipole2 is None:
            self.match_dispersion_and_R56()
        return self.field_dipole * self._field_ratio_chicane_dipole2

    @property
    def field_gradient_sextupole(self) -> float:
        "Sextupole field gradient [T/m^2]"
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        return self.charge_sign*self.strength_sextupole*p0/(SI.e*self.length_central_gap_or_sextupole)

    
    
    ## MATRIX LATTICE

    # full lattice 
    def matrix_lattice(self, k_lens=None, tau_lens=None, B_chic1=None, B_chic2=None, m_sext=None, half_lattice=False, orbit_only=False):
        
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
        "Matching the beta function by adjusting the plasma-lens strength."
        
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
        result_beta = minimize(minfun_beta, k_lens0, tol=1e-20, options={'maxiter': 200})
        self._strength_plasma_lens = result_beta.x[0]*self.length_plasma_lens

    
    def match_dispersion_and_R56(self, high_res=False):
        "Cancelling the dispersion and matchign the R56 by adjusting the chicane dipoles."
        
        nom_R56 = self.R56
            
        # normalizing scale for the merit function
        Dpx_scale = self.length_dipole*self.field_dipole*SI.c/self.nom_energy
        R56_scale = self.length_dipole**3*self.field_dipole**2*SI.c**2/self.nom_energy**2
        
        # minimizer function for dispersion (central dispersion prime is zero)
        from abel.utilities.beam_physics import evolve_dispersion, evolve_R56
        def minfun_dispersion_R56(params):
            ls, inv_rhos, ks, _, _ = self.matrix_lattice(tau_lens=0, B_chic1=params[0], B_chic2=params[1], m_sext=0, half_lattice=True)
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
        "Matching the chroaticity of function by adjusting the plasma-lens nonlinearity."

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
        "Cancelling the second-order dispersion by adjusting the sextupole strength."

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

    