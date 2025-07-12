import numpy as np
from abel.classes.bds.bds import BeamDeliverySystem
import scipy.constants as SI
from scipy.optimize import minimize

class DriverDelaySystem(BeamDeliverySystem):

    def __init__(self, E_nom=1, delay_per_stage=1, length_stage=10, num_stages=2, ks=[], \
                 keep_data=False, enable_space_charge=False, enable_csr=False, enable_isr=False, \
                 focus_at_trough = "x", fill_factor=0.7, mirrored=False, layoutnr = 1, use_monitors=False):
        # E_nom in eV, delay in ns
        super().__init__()
        self._E_nom = E_nom #backing variable 
        self._delay_per_stage = delay_per_stage/1e9
        self.num_stages = num_stages
        self._length_stage = length_stage
        self.keep_data = keep_data
        self.enable_space_charge = enable_space_charge
        self.enable_csr = enable_csr
        self.enable_isr = enable_isr
        self.ks = ks
        self.mirrored = mirrored
        self.layoutnr = layoutnr
        self.use_monitors = use_monitors

        # Lattice-elements lengths
        self.l_quads = 0.1
        self.fill_factor = fill_factor
        self.l_dipole, self.l_straight, self.l_drift = self.get_lattice_lengths(self.delay_per_stage)
        # 2*cell_length = L_tot = 2*L_stage+2*c*delay_per_stage

        # Use lengths to get the required B-field
        self.B_dipole = self.get_B_field(E_nom/1e9)

        self.cell_length = self.get_length()

        self.initial_focus_plane = focus_at_trough

    @property
    def E_nom(self):
        return self._E_nom
    
    @property
    def delay_per_stage(self):
        return self._delay_per_stage
    
    @property
    def length_stage(self):
        return self._length_stage
    
    @length_stage.setter
    def length_stage(self, value):
        self._length_stage = value
        self.l_dipole, self.l_straight, self.l_drift = self.get_lattice_lengths(self.delay_per_stage/1e9)
        self.cell_length = self.get_length()/4
        self.B_dipole = self.get_B_field(self.E_nom/1e9)

    @delay_per_stage.setter
    def delay_per_stage(self, value):
        self._delay_per_stage = value
        self.l_dipole, self.l_straight, self.l_drift = self.get_lattice_lengths(value/1e9)
        self.cell_length = self.get_length()/4
        self.B_dipole=self.get_B_field(self.E_nom/1e9)


    # Setting the new E_nom
    @E_nom.setter
    def E_nom(self, value):
        self._E_nom = value
        # Recalculate B if new energy is set
        print("Recalculating B-field")
        self.B_dipole=self.get_B_field(value/1e9)

    def get_B_field(self, p):
        # p in GeV, B given in Tesla
        func = lambda B: 0 #### To be implemented ####
        
        from scipy.optimize import root_scalar
        res = root_scalar(func, x0=0.5)
        print("B-field calculation converged: ", res.converged)
        print("B-field: ", res.root)
        return res.root

    def get_lattice_lengths(self, delay):
        pass #### To be implemented ####
    
    
    def list2lattice(self, ls: list, ks: list, phis: list):
        import impactx
        lattice=[]
        if self.use_monitors:
            from abel.apis.impactx.impactx_api import initialize_amrex
            initialize_amrex()
            monitor = [impactx.elements.BeamMonitor(name='monitor', backend='h5', encoding='g')]
        else:
            monitor=[]
        
        lattice.extend(monitor)
        ns = 100
        for i, l in enumerate(ls):
            if phis[i]!=0:
                element = impactx.elements.ExactSbend(ds=l, phi=np.rad2deg(phis[i]), nslice=ns)
            elif ks[i]!=0:
                element = impactx.elements.ExactQuad(ds=l, k=ks[i], nslice=ns)
            else:
                element = impactx.elements.ExactDrift(ds=l, nslice=ns)
            
            lattice.append(element)

            lattice.extend(monitor)

        return lattice
    
    def get_ls_n_phis(self):
        L_tot = self.length_stage + self.delay_per_stage * SI.c
        L_diag = 3 # m
        L_gap = 0.5 # m
        L_kick = 1 # m
        L_dipoles_guess = (L_tot - L_diag - L_gap/2 - L_kick/2)/5 # the dipoles lengths. guess all but the first to be the same
        L_dipole_1st = L_dipoles_guess*2

        B_max = 1.4 # T
        p = self.E_nom/SI.c # eV/c, also: beam rigidity 
        inv_r = B_max/p # B in Tesla
        r = 1/inv_r
        theta = L_dipoles_guess*inv_r
        thetha_1st = L_dipole_1st*inv_r
        sines = r*np.sin(theta)
        
        # the equation that must be 0 for the projected length to be the same as the lentght of the stage
        const = lambda thetas: L_kick/2 + r*np.sin(thetha_1st) + 2*sines*np.cos(theta/2) + L_diag*np.cos(thetha_1st-theta) + 2*sines*np.cos(theta/2) + \
                                sines + L_gap/2 - self.length_stage
        pass

        ##### TO BE CONTINUED ##### solve this equation with root and see what the initial guesses for theta is. 
        # Start optimizing for dispersion and R56 under constraint of delay and length to hold

    def inv_r2phi(self, inv_rs: list):
        phis=[]
        for inv_r in inv_rs:
            phi = self.l_dipole*inv_r
            phis.append(phi)
        return phis
    
    def phi2inv_r(self, phis: list):
        """
        phis given in rads
        """
        inverse_rs = []
        for phi in phis:
            inv_r = phi/self.l_dipole
            inverse_rs.append(inv_r)
        return inverse_rs

    
    def get_lattice2stages(self, ks, return_lists=False):
        """
        Make list of ls, ks, and phis (basically predetermined), and list2lattice, to get final lattice
        """
        p = self.E_nom/SI.c # eV/c, also: beam rigidity 
        inv_r = self.B_dipole/p # B in Tesla
        bend_angle = self.l_dipole*inv_r
        bend_angle = np.rad2deg(bend_angle)

        #### To be implemented ####

        match self.layoutnr: # Different layouts for the delay chicane
            case 1:
                pass

            case 2:
                pass
            case 3:
                pass
            case _:
                raise ValueError("Invalid value for attribute: layoutnr")
        """
        if return_lists:
            return self.list2lattice(ls=ls, ks=ks_full, phis=phis), ls, ks_full, phis
        else:
            return self.list2lattice(ls=ls, ks=ks_full, phis=phis)
        """
        
    def get_lattice(self, ks):
        return self.get_lattice2stages(ks)*(self.num_stages//2)
    
    def match_quads(self):
        from abel.utilities.beam_physics import evolve_dispersion, evolve_R56
        # Initial guess
        ff = min([self.l_straight, self.l_dipole])
        if self.initial_focus_plane=="x":
            ff=ff
        elif self.initial_focus_plane=="y":
            ff=-ff
        else:
            raise ValueError("Invalid choice of initial focus plane. Must be x or y (as a string).")
        fd = -ff
        kx0 = 1/ff/self.l_quads/2
        ky0 = 1/fd/self.l_quads/2

        x0 = [kx0, ky0] #### To be implemented ####
            
        def minimize_periodic(params):
            _, ls, ks, phis = self.get_lattice2stages(params, return_lists=True) # phis given in degs, cause that's what the impactx lattice takes

            phis = np.deg2rad(phis)
            inverse_rs = self.phi2inv_r(phis)
            """
            Match dispersion and R56.
            params = [k1, k2, ...]
            """

            D, _, _ = evolve_dispersion(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, fast=True, plot=False, high_res=False)

            R56, _ = evolve_R56(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, fast=True, plot=False, high_res=False)

            # D´ is unitless, R56 is m/E. Normalise with energy per length
            R56_norm = self.E_nom/1e8 / self.cell_length
            #print(D**2, "  ", (R56*R56_norm)**2)
            return (R56*R56_norm)**2 + (D/self.cell_length)**2 #+ Dp**2
        

        res = minimize(minimize_periodic, x0=x0)
        print(res)
        self.ks = res.x
    
    def match_betas(self):
        """
        Use the quad-values found earlier, to find periodic beta/alpha-function
        """
        if not list(self.ks):
            self.match_quads()

        from abel.utilities.beam_physics import evolve_beta_function
        # Initial guess
        if self.initial_focus_plane=="x":
            beta0 = [10, 20] # betas
        else:
            beta0 = [20, 10] # betas

        def minimize_periodic(params):
            """
            params = [beta_x, beta_y]
            """
            _, ls, ks, _ = self.get_lattice2stages(self.ks, return_lists=True)                

            beta_x, alpha_x, _ = evolve_beta_function(ls, np.array(ks), beta0=params[0], alpha0=0, fast=True)
            beta_y, alpha_y, _ = evolve_beta_function(ls, -np.array(ks), beta0=params[1], alpha0=0, fast=True)

            return (beta_x-params[1])**2/self.cell_length**2 + (beta_y-params[0])**2/self.cell_length**2 + alpha_x**2 + alpha_y**2
        

        res = minimize(minimize_periodic, x0=beta0, bounds=[(1e-5, 100)]*2)

        print(res)
        betas = res.x
        return betas
    
    def match_quads_and_betas(self):
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_R56, evolve_second_order_dispersion

        ff = self.l_straight

        if self.initial_focus_plane=="x":
            ff=ff
            init = [10, 20] # betas
        else:
            ff=-ff
            init = [20, 10] # betas

        fd = -ff
        kx0 = 1/ff/self.l_quads/2
        ky0 = 1/fd/self.l_quads/2

        #### To be implemented ####
        # init.extend(ks)

        def minimize_periodic(params):
            beta_x0 = params[0]
            beta_y0 = params[1]

            ks = params[2:]

            _, ls, ks, phis = self.get_lattice2stages(ks=ks, return_lists=True) # phis given in degs

            phis = np.deg2rad(phis)

            inverse_rs = self.phi2inv_r(phis)

            D, _, _ = evolve_dispersion(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, fast=True, plot=False, high_res=False)
            #DD, _, _ = evolve_second_order_dispersion(np.array(ls), inv_rhos=np.array(inverse_rs), ks=np.array(ks),\
            #                                            ms=ms, taus=taus, fast=True)
            R56, _ = evolve_R56(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, fast=True, plot=False, high_res=False)

            # D´ is unitless, R56 is m/E. Normalise with energy per length
            R56_norm = self.E_nom/1e7 / self.cell_length

            beta_x, alpha_x, _ = evolve_beta_function(ls, np.array(ks), beta0=beta_x0, alpha0=0, fast=True)
            beta_y, alpha_y, _ = evolve_beta_function(ls, -np.array(ks), beta0=beta_y0, alpha0=0, fast=True)

            Dp_weigth = 1
            #(Dp*Dp_weigth)**2 + (DD/self.cell_length)**2 + (D/self.cell_length)**2 + (DD/self.cell_length)**2 +
            return (R56*R56_norm)**2  + (D/self.cell_length)**2 + alpha_y**2 + alpha_x**2 + \
                (beta_x-beta_x0)**2/self.cell_length**2 + (beta_y-beta_y0)**2/self.cell_length**2
        

        res = minimize(minimize_periodic, x0 = [], options={"maxiter": 5000})
        print(res)
        self.ks = res.x[2:]
        return res.x[0], res.x[1]

    
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        from abel.apis.impactx.impactx_api import run_impactx
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_R56, evolve_second_order_dispersion

        # Get lattice
        if not list(self.ks):
            print("matching quads")
            self.match_quads()
        
        self.lattice = self.get_lattice(self.ks)
        print(self.lattice)
        """
        Evolve beta function here
        """
        
        ls, ks, phis = self.get_element_attributes(self.lattice) # phis given in rads
        inverse_rs = self.phi2inv_r(phis)

        ms = [0]*len(ls)
        taus = [0]*len(ls)

        _, _, evox = evolve_beta_function(ls, ks, beta0=beam0.beta_x(), alpha0=beam0.alpha_x(), plot=True)
        _, _, evoy = evolve_beta_function(ls, -np.array(ks), beta0=beam0.beta_y(), alpha0=beam0.alpha_y(), plot=True)
        _, _, evoD = evolve_dispersion(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, plot=True)
        _, _, evoDD = evolve_second_order_dispersion(np.array(ls), inv_rhos=np.array(inverse_rs), ks=np.array(ks),\
                                                        ms=ms, taus=taus, plot=True)
        _, _ = evolve_R56(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, plot=True)

        # run ImpactX
        beam, evol = run_impactx(self.lattice, beam0, verbose=False, runnable=runnable, keep_data=self.use_monitors,\
                                  space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)
        
        # save evolution
        self.evolution = evol
        
        return super().track(beam, savedepth, runnable, verbose)
    
    def get_length(self):
        return 2*(self.l_quads + 2*self.l_dipole + self.l_straight)
    
    def get_nom_energy(self):
        return self.E_nom
    
    def get_element_attributes(self, lattice):
        ls = []
        ks = []
        phis = []
        for element in lattice:
                if hasattr(element, "ds"):
                    ls.append(element.ds)
                if hasattr(element, "k"):
                    ks.append(element.k)
                else:
                    ks.append(0)
                if hasattr(element, "phi"):
                    phi = np.deg2rad(element.phi)
                    phis.append(phi)
                else:
                    phis.append(0)
        return ls, ks, phis
    
    def get_quad_bounds(self, mirrored=None):
        k_max_dipole = 1/self.l_quads/self.l_dipole
        k_bound_dipole = (-k_max_dipole, k_max_dipole)
        if mirrored is None:
            mirrored=self.mirrored
        match self.layoutnr:
            case 1:
                drift = self.l_straight-2*self.l_quads
                k_max_drift = 1/self.l_quads/drift

                k_bound_drift = (-k_max_drift, k_max_drift)

                k_bounds = [k_bound_dipole, k_bound_drift]
                k_bounds.extend([k_bound_dipole]*2)
                
                if not mirrored:
                    k_bounds.extend([k_bound_drift, k_bound_dipole])

            case 2:
                drift = (self.l_straight-3*self.l_quads)/2
                k_max_drift = 1/self.l_quads/drift

                k_bound_drift = (-k_max_drift, k_max_drift)

                k_bounds = [k_bound_dipole]
                k_bounds.extend([k_bound_drift]*2)
                k_bounds.extend([k_bound_dipole]*2)
                
                if not mirrored:
                    k_bounds.extend([k_bound_drift]*2)
                    k_bounds.append(k_bound_dipole)
            case 3:
                drift = (self.l_straight-3*self.l_quads)/2
                drift_dipole = 2*self.l_dipole + self.l_quads
                k_max_drift = 1/self.l_quads/drift
                k_max_dipole = 1/self.l_quads/drift_dipole

                k_bound_drift = (-k_max_drift, k_max_drift)
                k_bound_dipole = (-k_max_dipole, k_max_dipole)

                k_bounds = [k_bound_drift]
                k_bounds.append(k_bound_dipole)
                k_bounds.extend([k_bound_drift]*2)
                
                if not mirrored:
                    k_bounds.extend([k_bound_dipole, k_bound_drift])
            case _:
                raise ValueError("Invalid choice of layoutnr")
            
        return k_bounds
    
    def plot_evolution(self):

        from matplotlib import pyplot as plt
        
        evol = self.evolution
        
        # prepare plot
        fig, axs = plt.subplots(3,3)
        fig.set_figwidth(20)
        fig.set_figheight(12)
        col0 = "tab:gray"
        col1 = "tab:blue"
        col2 = "tab:orange"
        long_label = 'Location [m]'
        long_limits = [min(evol.location), max(evol.location)]

        # plot energy
        axs[0,0].plot(evol.location, evol.energy / 1e9, color=col1)
        axs[0,0].set_ylabel('Energy [GeV]')
        axs[0,0].set_xlabel(long_label)
        axs[0,0].set_xlim(long_limits)
        
        # plot charge
        axs[0,1].plot(evol.location, abs(evol.charge[0]) * np.ones(evol.location.shape) * 1e9, ':', color=col0)
        axs[0,1].plot(evol.location, abs(evol.charge) * 1e9, color=col1)
        axs[0,1].set_ylabel('Charge [nC]')
        axs[0,1].set_xlim(long_limits)
        axs[0,1].set_ylim(0, abs(evol.charge[0]) * 1.3 * 1e9)
        
        # plot normalized emittance
        axs[0,2].plot(evol.location, evol.emit_ny*1e6, color=col2)
        axs[0,2].plot(evol.location, evol.emit_nx*1e6, color=col1)
        axs[0,2].set_ylabel('Emittance, rms [mm mrad]')
        axs[0,2].set_xlim(long_limits)
        axs[0,2].set_yscale('log')
        
        # plot energy spread
        axs[1,0].plot(evol.location, evol.rel_energy_spread*1e2, color=col1)
        axs[1,0].set_ylabel('Energy spread, rms [%]')
        axs[1,0].set_xlabel(long_label)
        axs[1,0].set_xlim(long_limits)
        axs[1,0].set_yscale('log')

        # plot bunch length
        axs[1,1].plot(evol.location, evol.bunch_length*1e6, color=col1)
        axs[1,1].set_ylabel(r'Bunch length, rms [$\mathrm{\mu}$m]')
        axs[1,1].set_xlabel(long_label)
        axs[1,1].set_xlim(long_limits)

        # plot beta function
        axs[1,2].plot(evol.location, evol.beta_y, color=col2)  
        axs[1,2].plot(evol.location, evol.beta_x, color=col1)
        axs[1,2].set_ylabel('Beta function [m]')
        axs[1,2].set_xlabel(long_label)
        axs[1,2].set_xlim(long_limits)
        axs[1,2].set_yscale('log')
        
        # plot transverse offset
        axs[2,0].plot(evol.location, evol.y*1e6, color=col2)  
        axs[2,0].plot(evol.location, evol.x*1e6, color=col1)
        axs[2,0].set_ylabel(r'Transverse offset [$\mathrm{\mu}$m]')
        axs[2,0].set_xlabel(long_label)
        axs[2,0].set_xlim(long_limits)
        
        # plot dispersion
        axs[2,1].plot(evol.location, evol.dispersion_y*1e3, color=col2)  
        axs[2,1].plot(evol.location, evol.dispersion_x*1e3, color=col1)
        #axs[2,1].plot(evol.location, evol.second_order_dispersion_x*1e3, ':', color=col1)
        axs[2,1].set_ylabel('First-order dispersion [mm]')
        axs[2,1].set_xlabel(long_label)
        axs[2,1].set_xlim(long_limits)

        # plot beam size
        axs[2,2].plot(evol.location, evol.beam_size_y*1e6, color=col2)  
        axs[2,2].plot(evol.location, evol.beam_size_x*1e6, color=col1)
        axs[2,2].set_ylabel(r'Beam size, rms [$\mathrm{\mu}$m]')
        axs[2,2].set_xlabel(long_label)
        axs[2,2].set_xlim(long_limits)
        
        
        plt.show()