import numpy as np
from abel.classes.bds.bds import BeamDeliverySystem
import scipy.constants as SI
from scipy.optimize import minimize, root_scalar, NonlinearConstraint

class DriverDelaySystem_M(BeamDeliverySystem):

    def __init__(self, E_nom=2, delay_per_stage=1, length_stage=12, num_stages=2, ks=[], B_dipole=1, \
                 keep_data=False, enable_space_charge=False, enable_csr=False, enable_isr=False, layoutnr = 1,\
                 l_diag = 2, use_monitors=False, x0=10):
        # E_nom in eV, delay in ns
        super().__init__()
        self._E_nom = E_nom #backing variable 
        self._delay_per_stage = delay_per_stage*1e-9
        self.num_stages = num_stages
        self._length_stage = length_stage
        self.keep_data = keep_data
        self.enable_space_charge = enable_space_charge
        self.enable_csr = enable_csr
        self.enable_isr = enable_isr
        self.ks = ks
        self.layoutnr = layoutnr
        self.use_monitors = use_monitors
        self.x0=x0

        # Lattice-elements lengths
        self.l_quads = 0.1
        self.l_kick = 2
        self.l_gap = 0.25
        self.l_diag = l_diag
        # 2*cell_length = L_tot = 2*L_stage+2*c*delay_per_stage

        self.B_dipole = B_dipole

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
        self._delay_per_stage = value*1e-9


    # Setting the new E_nom
    
    @E_nom.setter
    def E_nom(self, value):
        self._E_nom = value

    

    def get_dipole_lengths(self):
        """
        Sets the lengths of the dipoles such that the delay is achieved
        """
        L = (self.length_stage + SI.c*self.delay_per_stage - self.l_gap/2 - self.l_kick/2 - self.l_diag)/4

        return L
    
    def get_dipole_fields(self):
        from abel.utilities.beam_physics import evolve_dispersion, evolve_R56

        p = self.E_nom/SI.c # eV/c, also: beam rigidity 

        ### Create constraints (delay and length of lattice). The function returns 2 values that should be kept around 0 while optimizing ###
        def constraint(Bs):
            """
            Bs= [B1, B2, B3], the B-fields to be optimized. Same input as in the optimize-function
            """
            B4 = -(Bs[0] + Bs[1] + Bs[2]) # Write the signs implicitly

            inv_r1 = Bs[0]/p
            inv_r2 = Bs[1]/p
            inv_r3 = Bs[2]/p
            inv_r4 = B4/p

            L_dipoles = self.get_dipole_lengths()

            # Calculate the bend angles
            theta1, theta2, theta3, theta4 = L_dipoles*inv_r1, L_dipoles*inv_r2, L_dipoles*inv_r3, L_dipoles*inv_r4

            # first dipole:
            if theta1==0:
                const1 = L_dipoles
            else:
                const1 = np.sin(abs(theta1))/inv_r1
            # second dipole
            if theta2==0:
                const2 = L_dipoles
            else:
                const2 = np.sin(abs(theta2/2))/inv_r2
            # third dipole
            if theta3==0:
                const3 = L_dipoles
            else:
                const3 = np.sin(abs(theta3/2))/inv_r3
            # fourth dipole
            if theta4==0:
                const4 = L_dipoles
            else:
                const4 = np.sin(abs(theta4))/inv_r4

            # calculate projected length and subtract L_stage to make the constraint 0
            constraint1 = self.l_kick/2 + const1 + 2*const2*np.cos(theta1+theta2/2) + \
                    self.l_diag*np.cos(theta1 + theta2) + 2*const3*np.cos(theta1+theta2+theta3/2) + \
                        const4 + self.l_gap/2 - self.length_stage # this must be 0 for our lattice to be valid (following the stages w. interstages)
            """
            With these constraints we can optimize for ls (see above) to make the dispersion prime and R56 zero.
            The constraints ensure the lengths give the desired delay, and makes sure the longitudinal length (projected length) is the same as the length of the stage.
            """
            return constraint1
        
        constraint_scipy = NonlinearConstraint(constraint, lb=-1e-9, ub=1e-9) #Making the constraint with the bounds around 0

        ### Create initial guess that conforms to this constraint (setting all lengths equal) ###
        x0 = self.x0

        def get_L_proj(x, B):
            # 1 B-value, positive
            r13 = p/B
            r24 = p/(B/x)
            L = self.get_dipole_lengths()

            theta1, theta2, theta3, theta4 = L/r13, -L/r24, -L/r13, L/r24

            L_proj = self.l_kick/2 + r13*np.sin(abs(theta1)) + 2*r24*np.sin(abs(theta2/2))*np.cos(theta1+theta2/2) + \
                            self.l_diag*np.cos(theta1 + theta2) + 2*r13*np.sin(abs(theta3)/2)*np.cos(theta1+theta2+theta3/2) + \
                                r24*np.sin(abs(theta4)) + self.l_gap/2
            return L_proj

        # Find B-field to make x = length_fraction
        B_calc = lambda B: get_L_proj(x0, B) - self.length_stage

        opt_dipole = root_scalar(B_calc, x0=self.B_dipole)

        B0 = [opt_dipole.root, -opt_dipole.root, -opt_dipole.root] # Initial guess in optimizing for Dispersion-prime and R56, manually set nr 2 and 3 negative
        

        def optimize_dipole_fields(Bs):
            D0 = 0
            Dp0 = 0

            if not self.ks:
                self.get_quad_strengths()

            ls_list, inv_rs_list, ks_list = self.get_elements_arrays(Bs=Bs, ks=self.ks) # Get expanded numpy arrays

            _, Dpx, _ = evolve_dispersion(ls=ls_list, ks=ks_list, inv_rhos=inv_rs_list)

            R56, _ = evolve_R56(ls=ls_list, ks=ks_list, inv_rhos=inv_rs_list)

            return Dpx**2 + (R56*1e2)**2
        
        B_bounds = [(-1.3,1.3)]*3 # Let B-values be negative

        opt = minimize(optimize_dipole_fields, x0=B0, bounds=B_bounds, constraints=constraint_scipy, options={'maxiter': 1000})
        print(opt)
            
        return opt.x
    
    def get_quad_strengths(self):
        ### Optimize for alpha_x/y to get quad-strengths ###
        # ls_B = [l_dipole1, l_dipole2, l_dipole3, l_diag, B1, B2]
        from abel.utilities.beam_physics import evolve_beta_function
        from scipy.optimize import minimize

        beta0_x = 5
        beta0_y = 5
        ls_list, = self.get_elements_arrays() # Get expanded numpy array of lengths


        def optimize_quads(ks_betas, plot=False):
            """
            Assume 0 bend while optimizing for the quads. This means the triplets are in the middle
            ks_betas = [k1, k2, k3, betax, betay]
            """
            ls_list, ks_list = self.get_elements_arrays(ks=ks_betas[:3]) # Get expanded numpy arrays
            if plot:
                print(ls_list)
                print(ks_list)


            _, alpha_x, _ = evolve_beta_function(ls_list, ks_list, beta0=ks_betas[3], alpha0=0, plot=plot)
            _, alpha_y, _ = evolve_beta_function(ls_list, ks_list, beta0=ks_betas[4], alpha0=0, plot=plot)

            return alpha_x**2 + alpha_y**2
        
        f0_1 = 2 * ls_list[5]  # drift between focusing quads in triplet
        f0_2 = np.sum(ls_list[-6:])*2 # distance from mid quad to next mid quad (ish)
        f0_3 = f0_2
        k0 = [1/f0_1/self.l_quads, -1/f0_2/self.l_quads, 1/f0_3/self.l_quads, beta0_x, beta0_y] # initial guess
        k_bounds = [(0,50)]*3
        k_bounds[1] = (-50,0)
        k_bounds.extend([(0.1,100)]*2)

        opt = minimize(optimize_quads, x0=k0, bounds=k_bounds)
        print(opt)
        #assert opt.fun<1e-3
        self.ks = list(opt.x[:3]) # set the quad-strengths in the lattice
        optimize_quads(opt.x, plot=True)
        return opt.x[3:]

    
    def get_elements_arrays(self, Bs=None, ks=None):
        # ls = [l_dipole1, l_dipole2, l_dipole3, l_diag, B1, B2]
        p = self.E_nom/SI.c

        L_dipoles = self.get_dipole_lengths()
        L_to_first_quad = 0.1 #m
        L_drift_diag = (self.l_diag - 3*self.l_quads - 2*L_to_first_quad)/2

        ls_list = [self.l_kick/2, L_dipoles, L_dipoles, L_to_first_quad, self.l_quads, L_drift_diag, self.l_quads, L_drift_diag, self.l_quads, L_to_first_quad, L_dipoles, L_dipoles, self.l_gap/2]
        assert abs(np.sum(ls_list) - (self.length_stage + self.delay_per_stage*SI.c)) < 1e-14
        yield np.array(ls_list)

        if Bs is not None:
            assert len(Bs)==3
            B4 = -(Bs[1] + Bs[2] + Bs[0]) # Explicitly using the signs of the field-values

            inv_r1 = Bs[0]/p
            inv_r2 = Bs[1]/p
            inv_r3 = Bs[2]/p
            inv_r4 = B4/p
            inv_rs_list = [0, inv_r1, inv_r2, 0, 0, 0, 0, 0, 0, 0, inv_r3, inv_r4, 0]

            yield np.array(inv_rs_list)
            
        if ks is not None:
            assert len(ks)==3
            ks_list = [0, 0, 0, 0, ks[0], 0, ks[1], 0, ks[2], 0, 0, 0, 0]
            yield np.array(ks_list)




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