import numpy as np
from abel.classes.bds.bds import BeamDeliverySystem
import scipy.constants as SI
from scipy.optimize import minimize, root_scalar, NonlinearConstraint

class DriverDelaySystem_M(BeamDeliverySystem):

    def __init__(self, E_nom=2, delay_per_stage=1, length_stage=12, num_stages=2, ks=[], B_dipole=0.8, \
                 keep_data=False, enable_space_charge=False, enable_csr=False, enable_isr=False, layoutnr = 1,\
                 l_diag = 2, use_monitors=False):
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
        self.layoutnr = layoutnr
        self.use_monitors = use_monitors

        # Lattice-elements lengths
        self.l_quads = 0.1
        self.l_kick = 2
        self.l_gap = 0.5
        self.l_diag = l_diag
        # 2*cell_length = L_tot = 2*L_stage+2*c*delay_per_stage

        self.B_dipole1 = B_dipole

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

    @delay_per_stage.setter
    def delay_per_stage(self, value):
        self._delay_per_stage = value


    # Setting the new E_nom
    
    @E_nom.setter
    def E_nom(self, value):
        self._E_nom = value

    

    def get_dipole_lengths(self):
        from abel.utilities.beam_physics import evolve_dispersion, evolve_R56

        p = self.E_nom/SI.c # eV/c, also: beam rigidity 
        length_fraction = 2

        # Initial guess for diagonal length
        l_diag = 1 #m
        ### Create constraints (delay and length of lattice). The function returns 2 values that should be kept around 0 while optimizing ###
        def constraint(ls_B):
            """
            ls = [l_dipole1, l_dipole2, l_dipole3, l_diag, B1, B2], the lengths (and B-fields) to be optimized. Same input as in the optimize-function
            """
            r1 = p/ls_B[-2]
            r2 = p/ls_B[-1]

            l_diag = ls_B[3]
            theta1, theta2, theta3 = ls_B[0]/r1, -ls_B[1]/r1, -ls_B[2]/r2 # dipole 2 and 3 bend opposite way, giving a change in sign
            theta4 = -(theta1 + theta2 + theta3)

            constraint1 = self.l_kick/2 + r1*np.sin(abs(theta1)) + 2*r1*np.sin(abs(theta2/2))*np.cos(theta1+theta2/2) + \
                    l_diag*np.cos(theta1 + theta2) + 2*r2*np.sin(abs(theta3)/2)*np.cos(theta1+theta2+theta3/2) + \
                        r2*np.sin(abs(theta4)) + self.l_gap/2 - self.length_stage # this must be 0 for our lattice to be valid (following the stages w. interstages)
            constraint2 = self.l_kick/2 + self.l_gap/2 + np.sum(ls_B[:4]) + theta4*r2 + l_diag - self.length_stage - SI.c*self.delay_per_stage # must be 0.
            """
            With these constraints we can optimize for ls (see above) to make the dispersion prime and R56 zero.
            The constraints ensure the lengths give the desired delay, and makes sure the longitudinal length (projected length) is the same as the length of the stage.
            """
            return [constraint1, constraint2]
        
        constraint_scipy = NonlinearConstraint(constraint, lb=[-1e-4]*2, ub=[1e-4]*2) #Making the constraint with the bounds around 0. multiply list by 2 to get 2 entries (for 2 constraints)

        ### Create initial guess that conforms to these constraints (try setting all lengths equal (maybe vary the l_diag)) ###
        def get_length(x):
            L = (self.length_stage+SI.c*self.delay_per_stage-self.l_kick/2-self.l_gap/2-l_diag)/((1+1/x)*2) # Gets a length that conforms to the required delay
            return L
        
        L_dipoles = get_length(length_fraction) # Just a starting point for the optimization (equal B-fields are default)

        def get_L_proj(x, B=None):
            if B is None:
                B = self.B_dipole
            else:
                r = p/B
            L = get_length(x)
            theta1, theta2, theta3, theta4 = L/r, -L/r/x, -L/r, L/r/x
            L_proj = self.l_kick/2 + r*np.sin(abs(theta1)) + 2*r*np.sin(abs(theta2/2))*np.cos(theta1+theta2/2) + \
                            l_diag*np.cos(theta1 + theta2) + 2*r*np.sin(abs(theta3)/2)*np.cos(theta1+theta2+theta3/2) + \
                                r*np.sin(abs(theta4)) + self.l_gap/2
            return L_proj
        
        # Find B-field to make x=length_fraction
        B_calc = lambda B: get_L_proj(length_fraction, B) - self.length_stage

        opt_dipole = root_scalar(B_calc, x0=self.B_dipole1)

        L0 = [L_dipoles, L_dipoles/length_fraction, L_dipoles, l_diag, opt_dipole.root, opt_dipole.root] # Initial guess in optimizing for Dispersion-prime and R56
        

        def optimize_dipole_lengths(ls_B):
            D0 = 0
            Dp0 = 0
            B1 = ls_B[-2]
            B2 = ls_B[-1]

            ks, fun, success = self.get_quad_strengths(ls_B)

            if not success:
                return fun*1e3

            ls_list, ks_list, inv_rs_list = self.get_elements_arrays(ls_B, ks) # Get expanded numpy arrays

            _, Dpx, _ = evolve_dispersion(ls=ls_list, ks=ks_list, inv_rhos=inv_rs_list)

            R56, _ = evolve_R56(ls=ls_list, ks=ks_list, inv_rhos=inv_rs_list)

            return Dpx**2 + (R56/self.length_stage)**2
        
        L_bounds = [(0.5,3.5)]*4
        L_bounds[3] = (1,3) # Set another boundary for the straight section (where the triplet is)

        B_bounds = [(0.1,1.3)]*2
        L_bounds.extend(B_bounds)
        opt = minimize(optimize_dipole_lengths, x0=L0, bounds=L_bounds, constraints=constraint_scipy)
        print(opt)
            
        return opt.x
    
    def get_dipole_fields(self):
        from abel.utilities.beam_physics import evolve_dispersion, evolve_R56

        p = self.E_nom/SI.c # eV/c, also: beam rigidity 
        length_fraction = 2

        # Initial guess for diagonal length
        l_diag = 2 #m
        ### Create constraints (delay and length of lattice). The function returns 2 values that should be kept around 0 while optimizing ###
        def constraint(Bs):
            """
            Bs= [B1, B2, B3], the B-fields to be optimized. Same input as in the optimize-function
            """
            B4 = -(Bs[0] + Bs[1] + Bs[2])

            r1 = p/Bs[0]
            r2 = p/Bs[1]
            r3 = p/Bs[2]
            r4 = p/B4

            L1, L2, L3, L4 = self.get_dipole_lengths()

            constraint1 = self.l_kick/2 + r1*np.sin(abs(theta1)) + 2*r1*np.sin(abs(theta2/2))*np.cos(theta1+theta2/2) + \
                    l_diag*np.cos(theta1 + theta2) + 2*r2*np.sin(abs(theta3)/2)*np.cos(theta1+theta2+theta3/2) + \
                        r2*np.sin(abs(theta4)) + self.l_gap/2 - self.length_stage # this must be 0 for our lattice to be valid (following the stages w. interstages)
            """
            With these constraints we can optimize for ls (see above) to make the dispersion prime and R56 zero.
            The constraints ensure the lengths give the desired delay, and makes sure the longitudinal length (projected length) is the same as the length of the stage.
            """
            return constraint1
        
        constraint_scipy = NonlinearConstraint(constraint, lb=[-1e-4]*2, ub=[1e-4]*2) #Making the constraint with the bounds around 0. multiply list by 2 to get 2 entries (for 2 constraints)

        ### Create initial guess that conforms to these constraints (try setting all lengths equal (maybe vary the l_diag)) ###
        def get_length(x):
            L = (self.length_stage+SI.c*self.delay_per_stage-self.l_kick/2-self.l_gap/2-l_diag)/((1+1/x)*2) # Gets a length that conforms to the required delay
            return L
        
        L_dipoles = get_length(length_fraction) # Just a starting point for the optimization (equal B-fields are default)

        def get_L_proj(x, B=None):
            if B is None:
                B = self.B_dipole
            else:
                r = p/B
            L = get_length(x)
            theta1, theta2, theta3, theta4 = L/r, -L/r/x, -L/r, L/r/x
            L_proj = self.l_kick/2 + r*np.sin(abs(theta1)) + 2*r*np.sin(abs(theta2/2))*np.cos(theta1+theta2/2) + \
                            l_diag*np.cos(theta1 + theta2) + 2*r*np.sin(abs(theta3)/2)*np.cos(theta1+theta2+theta3/2) + \
                                r*np.sin(abs(theta4)) + self.l_gap/2
            return L_proj
        
        # Find B-field to make x=length_fraction
        B_calc = lambda B: get_L_proj(length_fraction, B) - self.length_stage

        opt_dipole = root_scalar(B_calc, x0=self.B_dipole1)

        L0 = [L_dipoles, L_dipoles/length_fraction, L_dipoles, l_diag, opt_dipole.root, opt_dipole.root] # Initial guess in optimizing for Dispersion-prime and R56
        

        def optimize_dipole_lengths(ls_B):
            D0 = 0
            Dp0 = 0
            B1 = ls_B[-2]
            B2 = ls_B[-1]

            ks, fun, success = self.get_quad_strengths(ls_B)

            if not success:
                return fun*1e3

            ls_list, ks_list, inv_rs_list = self.get_elements_arrays(ls_B, ks) # Get expanded numpy arrays

            _, Dpx, _ = evolve_dispersion(ls=ls_list, ks=ks_list, inv_rhos=inv_rs_list)

            R56, _ = evolve_R56(ls=ls_list, ks=ks_list, inv_rhos=inv_rs_list)

            return Dpx**2 + (R56/self.length_stage)**2
        
        L_bounds = [(0.5,3.5)]*4
        L_bounds[3] = (1,3) # Set another boundary for the straight section (where the triplet is)

        B_bounds = [(0.1,1.3)]*2
        L_bounds.extend(B_bounds)
        opt = minimize(optimize_dipole_lengths, x0=L0, bounds=L_bounds, constraints=constraint_scipy)
        print(opt)
            
        return opt.x
    
    def get_quad_strengths(self, ls_B):
        ### Optimize for alpha_x/y to get quad-strengths ###
        # ls_B = [l_dipole1, l_dipole2, l_dipole3, l_diag, B1, B2]
        from abel.utilities.beam_physics import evolve_beta_function
        from scipy.optimize import minimize

        beta0_x = 10
        beta0_y = 10
        ls_list, _ = self.get_elements_arrays(ls_B) # Get expanded numpy arrays

        def optimize_quads(ks_betas):
            """
            Assume 0 bend while optimizing for the quads. This means the triplets are in the middle
            ks_betas = [k1, k2, k3, betax, betay]
            """
            ls_list, ks_list, _ = self.get_elements_arrays(ls_B, ks_betas[:3]) # Get expanded numpy arrays
            beta0_x = ks_betas[3]
            beta0_y = ks_betas[4]

            _, alpha_x, _ = evolve_beta_function(ls_list, ks_list, beta0=beta0_x, alpha0=0)
            _, alpha_y, _ = evolve_beta_function(ls_list, ks_list, beta0=beta0_y, alpha0=0)

            return alpha_x**2 + alpha_y**2
        
        f0_1 = ls_list[5]
        f0_2 = f0_1
        f0_3 = np.sum(ls_list[-4:])*2 # distance from last quad in triplet, to first quad in next triplet
        k0 = [1/f0_1/self.l_quads, -1/f0_2/self.l_quads, 1/f0_3/self.l_quads, beta0_x, beta0_y]

        opt = minimize(optimize_quads, x0=k0)
        print(opt.success)

        return opt.x, opt.fun, opt.success

    
    def get_elements_arrays(self, ls_B, ks=None):
        # ls = [l_dipole1, l_dipole2, l_dipole3, l_diag, B1, B2]
        p = self.E_nom/SI.c
        inv_r1 = ls_B[-2]/p
        inv_r2 = ls_B[-1]/p

        theta1, theta2, theta3 = ls_B[0]*inv_r1, -ls_B[1]*inv_r1, -ls_B[2]*inv_r2 # dipole 2 and 3 bend opposite way, giving a change in sign
        theta4 = -(theta1 + theta2 + theta3)
        l4 = theta4/inv_r2

        # Make equally space triplet at the diagonal. The drift lengths will then be:
        l_drift_diag = (ls_B[3]-self.l_quads*3)/4

        ls_list = [self.l_kick/2, ls_B[0], ls_B[1], l_drift_diag, self.l_quads, l_drift_diag, self.l_quads, l_drift_diag, self.l_quads, l_drift_diag, ls_B[2], l4, self.l_gap/2]
        inv_rs_list = [0, inv_r1, -inv_r1, 0, 0, 0, 0, 0, 0, 0, -inv_r2, inv_r2, 0]

        if ks is not None:
            ks_list = [0, 0, 0, 0, ks[0], 0, ks[1], 0, ks[2], 0, 0, 0, 0]
            return np.array(ls_list), np.array(ks_list), np.array(inv_rs_list)
        else:
            return np.array(ls_list), np.array(inv_rs_list)



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
            pass
        #phis=[]
        #for inv_r in inv_rs:
            #phi = self.l_dipole*inv_r
            #phis.append(phi)
            """
            phis returned as radians
            """
        #return phis
    
    def phi2inv_r(self, phis: list):
        """
        phis given in rads
        """
        #inverse_rs = []
        #for phi in phis:
        #    inv_r = phi/self.l_dipole
        #    inverse_rs.append(inv_r)
        #return inverse_rs
        pass
    
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