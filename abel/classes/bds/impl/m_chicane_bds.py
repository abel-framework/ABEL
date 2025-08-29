import numpy as np
from abel.classes.bds.bds import BeamDeliverySystem
import scipy.constants as SI
from scipy.optimize import minimize, root_scalar, NonlinearConstraint

class DriverDelaySystem_M(BeamDeliverySystem):

    def __init__(self, E_nom=2, delay_per_stage=1, length_stage=12, num_stages=2, ks=[], B_dipole=1, \
                 keep_data=False, enable_space_charge=False, enable_csr=False, enable_isr=False, layoutnr = 1,\
                 l_diag = 2, use_monitors=False, x0=10, lattice=[], ns=50, beta0_x=1, beta0_y=1):
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
        self.x0 = x0
        self.lattice = lattice
        self.ns = ns
        self.beta0_x = beta0_x
        self.beta0_y = beta0_y

        # Lattice-elements lengths
        self.l_quads = 0.4
        self.l_kick = 2
        self.l_gap = 0.25
        self.l_diag = l_diag
        self.l_dipole = self.get_dipole_lengths()
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

    ### Create constraints (delay and length of lattice). The function returns 2 values that should be kept around 0 while optimizing ###
    # If I find the root of this function (I.e where it is 0, as it should be) by 
    def constraint(self, params=None, B_multiplier=1, plot=False):
        from abel.utilities.beam_physics import evolve_orbit
        """
        params= [k1, k2, k3, k4, B1, B2, B3], the B-fields to be optimized. Same input as in the optimize-function
        """

        ls, inv_rs = self.get_elements_arrays_from_lattice(dipoles=(True, 'inv_rhos')) # dipoles = (bool, 'phi' OR 'inv_rhos'))
        inv_rs = [inv_r*B_multiplier for inv_r in inv_rs]
        
        theta, evolution = evolve_orbit(ls=ls, inv_rhos=inv_rs, plot=plot) # evolution[0,:] = xs is the projected length

        hypotenuse = np.sqrt(evolution[0,-1]**2 + evolution[1,-1]**2)

        return evolution[0,-1] - self.length_stage, hypotenuse 

    def get_dipole_lengths(self):
        """
        Sets the lengths of the dipoles such that the delay is achieved
        """
        L = (self.length_stage + SI.c*self.delay_per_stage - self.l_gap/2 - self.l_kick/2 - self.l_diag)/4

        return L
    def get_quad_counts(self):
        N = 0
        for element in self.lattice:
            if element.name=='quad':
                N+=1
        return N

    def get_dipole_fields(self, optimize_quads=False):
        from abel.utilities.beam_physics import evolve_dispersion, evolve_R56, evolve_beta_function

        p = self.E_nom/SI.c # eV/c, also: beam rigidity 
        self.l_dipole = self.get_dipole_lengths()

        if optimize_quads:
            beta0_x = 0.5
            beta0_y = 0.5
        

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
        print(opt_dipole)

        # Get the lengths of the lattice elements (does not vary)
        ls_list, = self.get_elements_arrays()

        # Initial guess in optimizing for Dispersion-prime and R56, manually set nr 2 and 3 negative
        B0 = [opt_dipole.root, -opt_dipole.root/self.x0, -opt_dipole.root] 

        if optimize_quads:
            f0_1 = np.sum(ls_list[:5])/2  # drift between focusing quads in triplet
            f0_2 = f0_1 # distance from mid quad to next mid quad (ish)

            k0 = [1/f0_1/self.l_quads, -1/f0_2/self.l_quads, 1/f0_1/self.l_quads]
            k0.extend(B0)
            B0 = k0

        if not optimize_quads and not self.ks:
            self.get_quad_strengths()

        def optimize_dipole_fields(Bs, plot=False):
            """
            if optimize_quads:
                Bs = [k1, k2, B1, B2, B3]
            else:
                Bs = [B1, B2, B3]
            """

            if optimize_quads:
                _, inv_rs_list, ks_list = self.get_elements_arrays(Bs=Bs[-3:], ks=Bs[:3]) # Get expanded numpy arrays
            else:
                _, inv_rs_list, ks_list = self.get_elements_arrays(Bs=Bs, ks=self.ks) # Get expanded numpy arrays

            _, Dpx, _ = evolve_dispersion(ls=ls_list, ks=ks_list, inv_rhos=inv_rs_list)

            R56, _ = evolve_R56(ls=ls_list, ks=ks_list, inv_rhos=inv_rs_list)

            #L_const = self.constraint(Bs[-3:])

            if optimize_quads:
                beta_x, alpha_x, _ = evolve_beta_function(ls_list, ks_list, beta0=beta0_x, alpha0=0, plot=plot)
                beta_y, alpha_y, _ = evolve_beta_function(ls_list, -ks_list, beta0=beta0_y, alpha0=0, plot=plot)

                return (Dpx*1e1)**2 + alpha_x**2 + alpha_y**2 + (R56*1e1)**2 + max(beta_x-beta0_x*5,0)**2 + max(beta_y-beta0_y*5,0)**2
            else:
                return Dpx**2 + (R56*1e1)**2
        
        B_bounds = [(-2,2)]*3 # Let B-values be negative
        #B_bounds[0] = (1,1.4)

        constraint = NonlinearConstraint(self.constraint, lb=0, ub=0)
        
        if optimize_quads:
            k_bounds = [(0,15), (-15,0), (0,15)]
            #k_bounds[1] = (-50,0)
            k_bounds.extend(B_bounds)
            opt = minimize(optimize_dipole_fields, x0=B0, bounds=k_bounds, options={'maxiter': 2000}, constraints=constraint)
        else:
            opt = minimize(optimize_dipole_fields, x0=B0, bounds=B_bounds, options={'maxiter': 2000}, constraints=constraint)
        print(opt)
            
        return opt.x
    
    def get_quad_strengths(self, match_full_length=False, ks_matched=False, match_dipole=False):
        ### Optimize for alpha_x/y to get quad-strengths ###
        # ls_B = [l_dipole1, l_dipole2, l_dipole3, l_diag, B1, B2]
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_R56
        from scipy.optimize import minimize

        p = self.E_nom/SI.c

        n_quads = self.get_quad_counts()
        print(n_quads)

        beta0_x = self.beta0_x
        beta0_y = self.beta0_y
        if not ks_matched:
            f0 = (self.length_stage+self.delay_per_stage*SI.c)/2
            if n_quads%2==0:
                k0 = [1/f0/self.l_quads, -1/f0/self.l_quads]*(n_quads//2)
                k_bounds = [(0,15), (-15,0)]*(n_quads//2)
                print(len(k_bounds))
            else:
                k0 = [1/f0/self.l_quads, -1/f0/self.l_quads]*(n_quads//2)
                k0.append(1/f0/self.l_quads)
                k_bounds = [(0,15), (-15,0)]*(n_quads//2)
                k_bounds.append((0,15))
        else:
            k0 = [element.k for element in self.lattice if element.name=='quad']
            if n_quads%2==0:
                k_bounds = [(0,15), (-15,0)]*(n_quads//2)
            else:
                k_bounds = [(0,15), (-15,0)]*(n_quads//2)
                k_bounds.append((0,15))
        if match_dipole:
            B_min = -2
            B_max = 2
            inv_r_max = B_max/p
            inv_r_min = B_min/p
            inv_r0 = -self.B_dipole/p
            k0.append(inv_r0)
            k_bounds.append((inv_r_min,inv_r_max))

        ls, inv_rs_list, ks_list = self.get_elements_arrays_from_lattice(dipoles=(True, 'inv_rhos'), quads=True) # returns numpy arrays

        # Get the indices of the quads in the lattice
        indices_quads = [i for i, element in enumerate(self.lattice) if element.name=='quad']
        indices_dipoles = [i for i, element in enumerate(self.lattice) if element.name=='dipole']
        
        def optimizer(params):
            # params = [k1, k2, k3, k4, ..., inv_r1]
            ks = params[:n_quads]
            inv_rs = params[n_quads:]
            # Vary the ks in the lattice without chinging the lattice values
            # This requires that all quads are named 'quad #' (#=number in which they appear in the lattice)
            for i, j in enumerate(indices_quads):
                ks_list[j] = ks[i]

            for i, j in enumerate(indices_dipoles):
                if i < len(inv_rs):
                    inv_rs_list[j] = inv_rs[i]
          
            beta_x, alpha_x, _ = evolve_beta_function(ls=ls, ks=ks_list, beta0=beta0_x)
            beta_y, alpha_y, _ = evolve_beta_function(ls=ls, ks=-ks_list, beta0=beta0_y)

            Dx, Dpx, _ = evolve_dispersion(ls=ls, ks=ks_list, inv_rhos=inv_rs_list)
            R56, _ = evolve_R56(ls=ls, ks=ks_list, inv_rhos=inv_rs_list)
            if match_full_length:
                return (beta_x-beta0_x)**2 + (beta_y-beta0_y)**2 + max(beta_x-10*beta0_x, 0)**2 + \
                 (Dx*1e2)**2 + (R56*1e2)**2 
            else:
                return alpha_x**2 + alpha_y**2 + Dpx**2 + R56**2 + \
                    max(beta_x-10*beta0_x, 0)**2 + max(beta_y-10*beta0_y, 0)**2
        
        opt = minimize(optimizer, x0=k0, bounds=k_bounds)
        print(opt)
        if len(opt.x)==n_quads:
            return opt.x
        else:
            return opt.x[:n_quads], opt.x[n_quads:]
    
    def set_ks(self, ks):
        j=0
        for element in self.lattice:
            if element.name == 'quad':
                print(ks[j])
                element.k = ks[j]
                j+=1

    
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
            B4 = -(Bs[0] + Bs[1] + Bs[2]) # implicitly using the signs of the field-values

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

    def get_elements_arrays_from_lattice(self, lengths: bool=True, dipoles: tuple=(False, 'inv_rhos'), quads: bool=False):
        # self.lattice = {'dipole/quad': (length, magnetic variable), 'drift': length, ...}
        # Make the lists that are required for the evolution functions in beam_physics.py
        # self.lattice consists of impactx elements
        if lengths:
            ls = [element.ds for element in self.lattice]
            yield np.array(ls)

        if dipoles[0]:
            if dipoles[1]=='inv_rhos':
                inv_rhos = [element.phi/element.ds if hasattr(element, 'phi') else 0 for element in self.lattice]
                yield np.array(inv_rhos)
            elif dipoles[1]=='phis':
                phis = [element.phi if hasattr(element, 'phi') else 0 for element in self.lattice]
                yield np.array(phis)
            else:
                raise ValueError('must use ´inv_rhos´ or ´phi´')

        if quads:
            ks = [element.k if hasattr(element, 'k') and element.name=='quad' else 0 for element in self.lattice]
            yield np.array(ks)





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
    
    def extend_lattice(self, full=False):
        deep_copy = self.copy_lattice()
        deep_copy.reverse()
        if full:
            for element in deep_copy:
                if element.name == 'dipole':
                    element.phi = -element.phi
        self.lattice.extend(deep_copy)
    
    def copy_lattice(self):
        """
        Manually creates a deep copy of the lattice.
        
        This is necessary because standard copy.deepcopy() fails on the
        custom impactx pybind objects.
        
        Returns:
            list: A new list with new, independent copies of each element.
        """
        import impactx
        import copy

        copied_lattice = []
        for element in self.lattice:
            # Re-create the element based on its type to ensure it's a new object
            if isinstance(element, impactx.elements.ExactSbend):
                new_element = impactx.elements.ExactSbend(
                    ds=element.ds,
                    phi=np.rad2deg(element.phi), # impactx constructor takes degrees, but attribute is in rads
                    nslice=element.nslice
                )
            elif isinstance(element, impactx.elements.ExactQuad):
                new_element = impactx.elements.ExactQuad(
                    ds=element.ds,
                    k=element.k,
                    nslice=element.nslice
                )
            elif isinstance(element, impactx.elements.ExactDrift):
                new_element = impactx.elements.ExactDrift(
                    ds=element.ds,
                    nslice=element.nslice
                )
            else:
                # For other non-impactx types (like BeamMonitor), a shallow copy is usually safe
                try:
                    new_element = copy.copy(element)
                except TypeError:
                    print(f"Warning: Could not copy element of type {type(element)}")
                    continue
            
            # Ensure other attributes like the name are also copied
            if hasattr(element, 'name'):
                new_element.name = element.name

            copied_lattice.append(new_element)
            
        return copied_lattice
    
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
    
    def plot_simplified(self):
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_R56
        ls, inv_rs, ks = self.get_elements_arrays_from_lattice(dipoles=(True, 'inv_rhos'), quads=True)

        _, _, _ = evolve_beta_function(ls=ls, ks=ks, beta0=self.beta0_x, plot=True)
        _, _, _ = evolve_beta_function(ls=ls, ks=-ks, beta0=self.beta0_y, plot=True)
        _, _, _ = evolve_dispersion(ls=ls, ks=ks, inv_rhos=inv_rs, plot=True)
        _, _ = evolve_R56(ls=ls, ks=ks, inv_rhos=inv_rs, plot=True)