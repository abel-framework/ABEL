import numpy as np
from abel.classes.bds.bds import BeamDeliverySystem
from abel.classes.source.impl.source_basic import SourceBasic
import scipy.constants as SI
from scipy.optimize import minimize, root_scalar, NonlinearConstraint


class DriverDelaySystem_M(BeamDeliverySystem):

    def __init__(self, E_nom=2, delay_per_stage=1, length_stage=12, num_stages=2, ks=[], B_dipole=1, \
                 keep_data=False, enable_space_charge=False, enable_csr=False, enable_isr=False, layoutnr = 1,\
                 l_diag = 2, use_monitors=False, x0=10, lattice=[], ns=100, beta0_x=1, beta0_y=1,\
                    alpha0_x=0, alpha0_y=0, Dx0=0, Dpx0=0, R560=0, k_bound=10):
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
        self.alpha0_x = alpha0_x
        self.alpha0_y = alpha0_y
        self.Dx0 = Dx0
        self.Dpx0 = Dpx0
        self.R560 = R560
        self.k_bound = k_bound

        # Lattice-elements lengths
        self.l_quads = 0.2
        self.l_kick = 1.5
        self.l_gap = 1
        self.l_diag = l_diag
        self.l_dipole = self.get_dipole_lengths()
        self.l_sext = 0.1
        # 2*cell_length = L_tot = 2*L_stage+2*c*delay_per_stage

        self.B_dipole = B_dipole
        self.full_extend=0

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

    def update_lattice(self):
        # Get the indices of the quads in the lattice

        ## Find doble quads and replace them with 1
        i = 1 # Start at 1 so we can look back at i-1
        while i < len(self.lattice):
            current = self.lattice[i]
            prev = self.lattice[i-1]
            
            # Check if both are 'quad'
            if current.name == 'quad' and prev.name == 'quad':
                
                # 1. Merge logic: Update the previous element
                prev.ds *= 2 
                
                # 2. Delete the current element
                self.lattice.pop(i)
                
                # 3. CRITICAL: Do NOT increment i.
                # The next element has now shifted into slot 'i'.
                # We need to check it against 'prev' in the next iteration.
                
            else:
                # Only move to the next index if we didn't remove anything
                i += 1

        self.indices_quads = [i for i, element in enumerate(self.lattice) if element.name=='quad']
        self.indices_dipoles = [i for i, element in enumerate(self.lattice) if element.name=='dipole']
        self.indices_sextupoles = [i for i, element in enumerate(self.lattice) if element.name=='sextupole']



    def scale_lattice(self, scale=1):
        for element in self.lattice:
            element.ds *=scale

    ### Create constraints (delay and length of lattice). The function returns 2 values that should be kept around 0 while optimizing ###
    # If I find the root of this function (I.e where it is 0, as it should be) by 
    def constraint(self, B_multiplier=1, plot=False, rotate=False, save=None, fig_size=None):
        from abel.utilities.beam_physics import evolve_orbit
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np # Ensure numpy is imported
        from matplotlib.lines import Line2D

        # Retrieve lengths, inverse radii, and sextupole strengths
        ls, inv_rs = self.get_elements_arrays_from_lattice(dipoles=(True, 'inv_rhos'))
        inv_rs = [inv_r*B_multiplier for inv_r in inv_rs]
        ms= 0*ls
        for index in self.indices_sextupoles:
            ms[index]=1
        
        theta, evolution = evolve_orbit(ls=ls, inv_rhos=inv_rs, plot=plot)

        hypotenuse = np.sqrt(evolution[0,-1]**2 + evolution[1,-1]**2)
        if rotate:
            phi = np.arctan(evolution[1,-1]/evolution[0,-1])
        else:
            phi = 0
        print("phi0 = ", phi, 'phi_end = ', theta)
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

        elements_location = self.get_z_elements(theta0=-phi)
        dipoles = elements_location['dipoles']
        quads = elements_location['quads']
        sexts = elements_location['sextupoles']
        print(sexts)

        rotated_data = np.dot(np.array([evolution[0,:], evolution[1,:]]).T, R)
        rotated_x = rotated_data[:, 0]
        rotated_y = rotated_data[:, 1]
        
        if fig_size is not None:
            fig, ax = plt.subplots(2,1, figsize=fig_size, gridspec_kw={'height_ratios': [4, 1]})
        else:
            fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [4, 1]})
            fig.tight_layout()
        
        ax[0].plot(rotated_x, rotated_y)
        ax[0].grid()
        ax[0].set_xlabel("z (m)")
        ax[0].set_ylabel("x (m)")
        ax[0].set_title("Delay chicane orbit")
        ax[1].tick_params(axis='y', left=False, labelleft=False)
        ax[1].tick_params(axis='x', bottom=False, labelbottom=False)
        print(rotated_y[-1],rotated_x[-1]/2)
        
        # Define magnet dimensions
        dipole_width = 0.5
        dipole_height = 0.2 
        quad_width = 0.1
        quad_height = 0.4 

        # Plot dipoles
        for z in dipoles:
            z_ = z[0]
            pos = z[1]
            corner_x = z_ - dipole_width / 2
            if pos==1:
                corner_y = 0
            elif pos==-1:
                corner_y = -dipole_height
            ax[1].add_patch(patches.Rectangle((corner_x, corner_y), dipole_width, dipole_height, facecolor='black', label='Dipole'))

        # Plot quadrupoles
        for z in quads:
            z_ = z[0]
            pos = z[1]
            corner_x = z_ - quad_width / 2
            
            if pos == 1:
                corner_y = 0
            elif pos == -1:
                corner_y = -quad_height
                
            ax[1].add_patch(patches.Rectangle((corner_x, corner_y), quad_width, quad_height, facecolor='black', label='Quadrupole'))
        
        # --- NEW SECTION: Plot Sextupole Diamonds ---
        # markersize=50, marker='d' (thin diamond), color='tab:red' (visible against black)
        if len(sexts) > 0:
            ax[1].scatter(sexts, [0]*len(sexts), 
                            marker='d', color='tab:red', s=50, zorder=10, label='Sextupole')
        # --------------------------------------------

        # Auto-adjust the plot limits
        ax[1].set_ylim(-(quad_height+0.5*quad_height), quad_height+0.5*quad_height)
        ax[1].set_xlim(ax[0].get_xlim()) 
        ax[1].axhline(0, color='black', linewidth=1.2, zorder=0)
          
        ax[1].set_aspect('auto', adjustable='box')

        # Get all handles and labels
        ax[1].set_ylim(-(quad_height+0.5*quad_height), quad_height+0.5*quad_height)
        ax[1].set_xlim(ax[0].get_xlim()) 
        ax[1].axhline(0, color='black', linewidth=1.2, zorder=0)
          
        ax[1].set_aspect('auto', adjustable='box')

        # --- UPDATED LEGEND SECTION ---
        # 1. Get current handles and labels
        handles, labels = ax[1].get_legend_handles_labels()
        
        # 2. Deduplicate using a dictionary
        by_label = dict(zip(labels, handles))

        # 3. Create a custom proxy for the Quadrupole
        # We use a Line2D with marker='|' (a vertical pipe) to simulate a thin, vertical rectangle.
        # markersize controls height, markeredgewidth controls thickness.
        if 'Quadrupole' in by_label:
            thin_quad_proxy = Line2D([], [], color='black', marker='|', 
                                     linestyle='None', markersize=15, markeredgewidth=3, 
                                     label='Quadrupole')
            # Replace the original rectangle handle with the new thin proxy
            by_label['Quadrupole'] = thin_quad_proxy

        # 4. Create the legend with the modified handles
        # Use 'edgecolor' and 'facecolor' to ensure the legend background doesn't hide things
        leg = ax[0].legend(by_label.values(), by_label.keys(), loc='upper right', 
                           frameon=True, facecolor='white', edgecolor='lightgray')

        if save is not None:
            print('saving plot')
            fig.savefig(save, bbox_inches='tight')
        plt.show()

        print('delay: ', (np.sum(ls)-evolution[0,-1])/SI.c*1e9)

        return evolution[0,-1] - self.length_stage, hypotenuse

    def get_delay(self):
        from abel.utilities.beam_physics import evolve_orbit
        # Retrieve lengths, inverse radii, and sextupole strengths
        ls, inv_rs = self.get_elements_arrays_from_lattice(dipoles=(True, 'inv_rhos'))
        inv_rs = [inv_r for inv_r in inv_rs]
        ms= 0*ls
        for index in self.indices_sextupoles:
            ms[index]=1
        
        theta, evolution = evolve_orbit(ls=ls, inv_rhos=inv_rs, plot=False)

        delay = (np.sum(ls)-evolution[0,-1])/SI.c*1e9

        return delay
    
    def get_z_elements(self, theta0=0):
        ls, phis, ks = self.get_elements_arrays_from_lattice(dipoles=(True, 'phis'), quads=True)

        elements_location={}
        dipoles = []
        quads = []
        sexts=[]
        L_z0 = 0

        for i, params in enumerate(zip(ls, phis, ks)):
            l, phi, k = params[0], params[1], params[2]
            if i in self.indices_sextupoles:
                L_z = l*np.cos(theta0)
                sexts.append(L_z0 + L_z/2)

            if phi != 0:
                r = l/phi
                L_z = 2*abs(r)*np.sin(abs(phi)/2)*np.cos(theta0 + phi/2)
                
                if phi<0:
                    dipoles.append((L_z0 + L_z/2, -1))
                else:
                    dipoles.append((L_z0 + L_z/2, 1))

                theta0 += phi
                L_z0 += L_z
            elif k!=0:
                L_z = l*np.cos(theta0)
                if k<0:
                    quads.append((L_z0 + L_z/2, -1))
                else:
                    quads.append((L_z0 + L_z/2, 1))
                L_z0 += L_z
            else:
                L_z0 += l*np.cos(theta0)
                
        elements_location['dipoles'] = dipoles
        elements_location['quads'] = quads
        elements_location['sextupoles'] = sexts

        return elements_location


    def get_dipole_lengths(self):
        """
        Sets the lengths of the dipoles such that the delay is achieved
        """
        L = (self.length_stage + SI.c*self.delay_per_stage - self.l_gap/2 - self.l_kick/2 - self.l_diag)/4

        return L
    def get_quad_counts(self):
        N = sum(1 for element in self.lattice if element.name=='quad') - self.get_n_dobble_quads()
        return N
    
    def get_n_dobble_quads(self):
        """
        Check how many quads consists of two halfs that are to be identical
        """
        S=0
        for i in self.indices_quads:
            if i+1 in self.indices_quads:
                S+=1
        return S
    
    def set_B_field(self, B):
        p = self.E_nom/SI.c
        for element in self.lattice:
            if element.name=='dipole':
                element.phi = element.ds*B/p * np.sign(element.phi)
                


    def match_lattice(self, matching_functions, ks_matched=False, ks_to_match='all', method=None,\
                      inv_rs_to_match=[], bounds=None, match_betas=False, extend_symmetrically=0):
        ### Optimize for alpha_x/y to get quad-strengths ###
        # ls_B = [l_dipole1, l_dipole2, l_dipole3, l_diag, B1, B2]
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_R56, evolve_second_order_dispersion
        from scipy.optimize import minimize
        """
        Returns matched values for the quads/dipoles marked in the input

        input: 
            ks_matched - if true, the initial guess will be the current quad values of the lattice

            ks_to_match - a list of integers, indicating which quads will be matched 
                            (only counts quads, not other lattice elemetns)

            inv_rs_to_match - a list of integers indicating which dipoles will be matched

            bounds - a list of tuples to match the k/inv_r values in the matching

            matching_functions - a dictionary giving the function (D, Dp, D2, D2p beta_x, beta_y, 
                                    alpha_x, alpha_y, R56) as key(s), and the desired values as values.

            symmetric - if symmetric, assume the second half of the lattice is mirror symmetric.

        output:
            The optimized values

        NOTE: must implement ks_to_match list in self.set_ks() as well.
                ALSO: Not implemented for amtching dipoles yet

        """
        p = self.E_nom/SI.c

        n_quads = self.get_quad_counts()
        print(n_quads)

        ls, inv_rs, ks = self.get_elements_arrays_from_lattice(dipoles=(True, 'inv_rhos'), quads=True)
        minimize_counter=0

        def optimizer(params):
            nonlocal minimize_counter
            # Params has same dimension as len(ks_to_match)+len(inv_rs_to_match)
            # if match_betas: params = [k0,k1,...,beta_x/y]

            # Check if all quads shall be matched
            if ks_to_match=='all' and not match_betas:
                for i, k in enumerate(params):
                    ks[self.indices_quads[i]] = k
            elif ks_to_match=='all' and match_betas:
                pars = params[:-2]
                for i, k in enumerate(pars):
                    ks[self.indices_quads[i]] = k

            # If only select quads are to be matched, set the param values 
            # that have same index as integers in ks_to_match
            # If only select quads are to be matched, set the param values 
            # that have same index as integers in ks_to_match
            else:
                k=0
                for i, j in enumerate(self.indices_quads):
                    if i in ks_to_match:
                        ks[j] = params[k]
                        k += 1

            # If we want to match betas as well            
            if match_betas:
                beta_x0 = params[-2]
                beta_y0 = params[-1]
            else:
                beta_x0 = self.beta0_x
                beta_y0 = self.beta0_y
            
            ls_ = ls
            ks_ = ks
            inv_rs_ = inv_rs
            

            for i in range(extend_symmetrically):
                if i%2==0:
                    ls_ = np.append(ls_, ls[::-1])
                    ks_ = np.append(ks_, ks[::-1])
                    inv_rs_ = np.append(inv_rs_, inv_rs[::-1])

                else:
                    ls_ = np.append(ls_, ls)
                    ks_ = np.append(ks_, ks)
                    inv_rs_ = np.append(inv_rs_, inv_rs)

            betax, alphax, evo_x = evolve_beta_function(ls=ls_, ks=ks_, inv_rhos=inv_rs_, beta0=beta_x0)
            betay, alphay, evo_y = evolve_beta_function(ls=ls_, ks=-ks_, beta0=beta_y0)
            D, Dp, _ = evolve_dispersion(ls=ls_, ks=ks_, inv_rhos=inv_rs_, high_res=True)
            R56, _ = evolve_R56(ls=ls_, ks=ks_, inv_rhos=inv_rs_, high_res=True)

            DD, DDp, _ = evolve_second_order_dispersion(ls=ls_, ks=ks_, inv_rhos=inv_rs_)
            
            max_betax = max(evo_x[1,:])
            max_betay = max(evo_y[1,:])

            min_betax = min(evo_x[1,:])
            min_betay = min(evo_y[1,:])

            S=0
            for key, value in matching_functions.items():
                match key:
                    case 'D':
                        S += ((D-value))**2
                    case 'Dp':
                        S += ((Dp-value)*1e1)**2
                    case 'DDp':
                        S += ((DDp-value)*1e1)**2 
                    case 'DD':
                        S += ((DD-value)*1e1)**2    
                    case 'R56':
                        S += ((R56-value)*1e2)**2
                    case 'beta_x':
                        S += ((betax-value))**2
                    case 'beta_y':
                        S += ((betay-value))**2
                    case 'alpha_x':
                        S += ((alphax-value))**2
                    case 'alpha_y':
                        S += ((alphay-value))**2
                    case 'max_betas':
                        # if beta is "value" times larger than initially, punish the optimizer
                        S += max(max_betax-beta_x0*value,0)**2 + max(max_betay-beta_y0*value,0)**2
                    case 'min_betas':
                        S += min(value-min_betax, 0)**2 + min(value-min_betay, 0)**2

            if minimize_counter%5==0:
                print(f'func: {S:.4e},', f'iter: {minimize_counter},', f'Dp: {Dp}', '             ', end='\r', flush=True)
            minimize_counter+=1

            return S
        
        # Set number of params/bounds to include
        loop_range = len(self.indices_quads)

        ## Prepare the initial guess        
        if ks_matched: # set the already optimized value as initial guess (for the quads to match)
            if ks_to_match=='all':
                x0 = [self.lattice[i].k for i in self.indices_quads]
            else:
                x0 = [self.lattice[j].k for i,j in enumerate(self.indices_quads) if i in ks_to_match]

        else:
            f0 = (self.get_length() + SI.c*self.delay_per_stage)/3
            k0 = 1/f0/self.l_quads
            print(f'k0: {k0}')
            x0=[]
            if ks_to_match=='all':
                if not bounds:
                    for i in range(loop_range):
                        if i%2==0:
                            x0.append(-k0)
                        else:
                            x0.append(k0)
                else:
                    bounds_list = bounds
                    if match_betas:
                        bounds_list = bounds_list[:-2]
                    for tup in bounds_list:
                        if tup[0]==0: # k0 positive
                            x0.append(k0)
                        else: # k0 is negative
                            x0.append(-k0)
            else:
                if not bounds:
                    for i in ks_to_match:
                        if i%2==0:
                            x0.append(-k0)
                        else:
                            x0.append(k0)
                else:
                    bounds_list = bounds                    
                    if match_betas:
                        bounds_list = bounds_list[:-2]

                    assert len(bounds_list)==len(ks_to_match)

                    for tup in bounds_list:
                        if tup[0]==0: # k0 positive
                            x0.append(k0)
                        else: # k0 is negative
                            x0.append(-k0)

        if match_betas:
            x0.extend([self.beta0_x, self.beta0_y])
        ## Run the optimization                
        opt = minimize(fun=optimizer, x0=x0, bounds=bounds, method=method, options={'gtol':1e-4})
        print() # <--- Move to a fresh line after optimization is done
        print("Optimization finished!")
        print(opt)

        if ks_to_match=='all' and not match_betas:
            ks = opt.x
        elif ks_to_match=='all' and match_betas:
            ks = opt.x[:-2]
        else:
            ks = opt.x[:len(ks_to_match)]
        if match_betas:
            betas = opt.x[-2:]
            return ks, betas
        else:
            return ks
        
    def match_second_order_dispersion(self, ms_init=None, ms_to_match='all', method=None, multiply_x0=1,
                                      extend_symmetrically=0):
        from abel.utilities.beam_physics import evolve_second_order_dispersion
        ls, inv_rs, ks = self.get_elements_arrays_from_lattice(dipoles=(True, 'inv_rhos'), quads=True)
        ms = ls*0
        taus = ls*0

        minimize_counter=0
        def optimizer(params):
            nonlocal minimize_counter

            if ms_to_match=='all':
                for i, index in enumerate(self.indices_sextupoles):
                    ms[index] = params[i]
            else:
                j=0
                for i, index in enumerate(self.indices_sextupoles):
                    if i in ms_to_match:
                        ms[index] = params[j]
                        j+=1
            ms_ = ms
            ks_ = ks
            inv_rs_ = inv_rs
            taus_ = taus
            ls_ = ls

            for i in range(extend_symmetrically):
                if i%2==0:
                    ms_=np.append(ms_, ms[::-1])
                    ls_=np.append(ls_, ls[::-1])
                    ks_=np.append(ks_, ks[::-1])
                    inv_rs_=np.append(inv_rs_, inv_rs[::-1])
                    taus_=np.append(taus_, taus[::-1])

                else:
                    ms_=np.append(ms_, ms)
                    ls_=np.append(ls_, ls)
                    ks_=np.append(ks_, ks)
                    inv_rs_=np.append(inv_rs_, inv_rs)
                    taus_=np.append(taus_, taus)

            DD, DDp, _ = evolve_second_order_dispersion(ls=ls_, ks=ks_, inv_rhos=inv_rs_, ms=ms_, taus=taus_)

            S = DD**2 + DDp**2

            if minimize_counter%5==0:
                print(f'func: {S:.4e}', f'iter: {minimize_counter}             ', end='\r', flush=True)
            minimize_counter+=1

            return S

        x0 = []
        if ms_to_match=='all':
            L = len(self.indices_sextupoles)
        else:
            L=len(ms_to_match)
            
        for i in range(L):
            if i%2==0:
                sign=-1*multiply_x0
            else:
                sign=1*multiply_x0
            x0.append(-100*sign)

        if ms_init is not None:
            x0 = ms_init

        opt = minimize(fun=optimizer, x0=x0, method=method, options={'gtol':1e-4})
        print()
        print(opt)

        return opt.x
        

    def reset_lattice(self):
        try:
            self.lattice = self.lattice_backup
            self.update_lattice()
            self.set_ks(self.ks)
            self.lattice[self.indices_dipoles[0]].phi = self.lattice[self.indices_dipoles[0]].ds*self.inv_r
        except:
            print('No ks, inv_r or lattice_backed up yet')
        
            
    
    def set_ks(self, ks, lattice=None, ks_to_set='all'):
        if not lattice:
            if ks_to_set =='all':
                for i, j in enumerate(self.indices_quads):
                    self.lattice[j].k = ks[i]
            else:
                k=0
                for i, j in enumerate(self.indices_quads):
                    if i in ks_to_set:
                        self.lattice[j].k = ks[k]
                        k+=1
        else:
            indices_quads = [i for i, element in enumerate(lattice) if element.name=='quad']
            if ks_to_set =='all':
                for i, j in enumerate(indices_quads):
                    lattice[j].k = ks[i]
            else:
                k=0
                for i, j in enumerate(indices_quads):
                    if i in ks_to_set:
                        lattice[j].k = ks[k]
                        k+=1

    def set_ms(self, ms, ms_to_set='all'):
        import impactx
        sextupole = impactx.elements.ExactMultipole

        if ms_to_set == 'all':
            for i, j in enumerate(self.indices_sextupoles):
                self.lattice[j] = sextupole(name='sextupole', ds=.2, k_normal=[0,0,ms[i]], k_skew=[0]*3)
        else:
            j=0
            for i, k in enumerate(self.indices_sextupoles):
                if i in ms_to_set:
                    self.lattice[k] = sextupole(name='sextupole', ds=.2, k_normal=[0,0,ms[j]], k_skew=[0]*3)
                    j+=1

    def get_elements_arrays_from_lattice(self, lengths: bool=True, dipoles: tuple=(False, 'inv_rhos'), quads: bool=False, sext: bool=False):
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

        if sext:
            ms = [element.k_normal[2] if hasattr(element, 'k_normal') and element.name=='sextupole' else 0 for element in self.lattice]
            yield ms

    
    def track(self, beam0, savedepth=0, runnable=None, verbose=False, plot=False):

        from abel.wrappers.impactx.impactx_wrapper import run_impactx
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_R56, evolve_second_order_dispersion

        # Get lattice
        #if not list(self.ks):
        #    print("matching quads")
        #    self.match_quads()
        
        #self.lattice = self.get_lattice(self.ks)
        #print(self.lattice)
        """
        Evolve beta function here
        """
        
        ls, inverse_rs, ks = self.get_elements_arrays_from_lattice(dipoles=(True, "inv_rhos"), quads=True) # phis given in rads
        
        ms = [0]*len(ls)
        taus = [0]*len(ls)
        
        _, _, evox = evolve_beta_function(ls, ks, beta0=beam0.beta_x(), inv_rhos=inverse_rs, alpha0=beam0.alpha_x(), plot=plot)
        _, _, evoy = evolve_beta_function(ls, -np.array(ks), beta0=beam0.beta_y(), alpha0=beam0.alpha_y(), plot=plot)
        _, _, evoD = evolve_dispersion(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, high_res=True, plot=plot)
        _, _, evoDD = evolve_second_order_dispersion(np.array(ls), inv_rhos=np.array(inverse_rs), ks=np.array(ks),\
                                                        ms=ms, taus=taus, plot=plot)
        _, _ = evolve_R56(np.array(ls), np.array(inverse_rs), np.array(ks), Dx0=0, Dpx0=0, high_res=True, plot=plot)

        # run ImpactX
        beam, evol = run_impactx(self.lattice, beam0, verbose=False, runnable=runnable, keep_data=self.use_monitors,\
                                  space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)
        
        # save evolution
        self.evolution = evol
        
        return super().track(beam, savedepth, runnable, verbose)
    
    def get_length(self):
        return sum(element.ds for element in self.lattice)
    
    def get_nom_energy(self):
        return self.E_nom
    
    
    def extend_lattice(self, full=False, quad_last=False):
        deep_copy = self.copy_lattice()
        if quad_last:
            deep_copy = deep_copy[:-1]

        if full and self.full_extend<1:
            for element in deep_copy:
                if element.name == 'dipole':
                    element.phi = -element.phi
            self.full_extend+=1
        else:
            deep_copy.reverse()
        

        self.lattice.extend(deep_copy)
        self.update_lattice()
    
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
                    name='dipole',
                    ds=element.ds,
                    phi=np.rad2deg(element.phi), # impactx constructor takes degrees, but attribute is in rads
                    nslice=element.nslice
                )
            elif isinstance(element, impactx.elements.ExactQuad):
                new_element = impactx.elements.ExactQuad(
                    name='quad',
                    ds=element.ds,
                    k=element.k,
                    nslice=element.nslice
                )
            elif isinstance(element, impactx.elements.ExactDrift):
                new_element = impactx.elements.ExactDrift(
                    name='drift',
                    ds=element.ds,
                    nslice=element.nslice
                )
            elif isinstance(element, impactx.elements.ExactMultipole):
                new_element = impactx.elements.ExactMultipole(
                    name='sextupole',
                    ds=element.ds,
                    nslice=element.nslice,
                    k_normal=[0,0,-100],
                    k_skew=[0]*3
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
    
    def plot_simplified(self, return_end_values=False, ms=None, extend=False, plot=True, return_figs=False):
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_R56,\
        evolve_second_order_dispersion
        ls, inv_rs, ks = self.get_elements_arrays_from_lattice(dipoles=(True, 'inv_rhos'), quads=True)
        taus = np.zeros_like(ls)
        ms_list = ls*0
        if ms is not None:
            print('changing ms')
            i=0
            for index in self.indices_sextupoles:
                ms_list[index]=ms[i]
                i+=1
        ls_, inv_rs_, ks_ = ls, inv_rs, ks
        ms_list_ = ms_list
        taus_ = taus

        for i in range(extend):
            if i%2==0:
                ls_ = np.append(ls_, ls[::-1])
                inv_rs_ = np.append(inv_rs_, inv_rs[::-1])
                ks_ = np.append(ks_, ks[::-1])
                ms_list_ = np.append(ms_list_,ms_list[::-1])
                taus_ = np.append(taus_, taus_[::-1])
            else:
                ls_ = np.append(ls_, ls)
                inv_rs_ = np.append(inv_rs_, inv_rs)
                ks_ = np.append(ks_, ks)
                ms_list_ = np.append(ms_list_,ms_list)
                taus_ = np.append(taus_, taus)

        beta_x, alpha_x, _, fig1 = evolve_beta_function(ls=ls_, ks=ks_, inv_rhos=inv_rs_, beta0=self.beta0_x, alpha0=self.alpha0_x, plot=plot, return_fig=True)
        beta_y, alpha_y, _, fig2 = evolve_beta_function(ls=ls_, ks=-ks_, beta0=self.beta0_y, alpha0=self.alpha0_y, plot=plot, plane='y', return_fig=True)
        D, Dp, _, fig3 = evolve_dispersion(ls=ls_, ks=ks_, inv_rhos=inv_rs_, Dx0=self.Dx0, Dpx0=self.Dpx0, plot=plot, high_res=True, return_fig=True)
        R56, _, fig4 = evolve_R56(ls=ls_, ks=ks_, inv_rhos=inv_rs_, plot=plot, R560=self.R560, high_res=True, return_fig=True)
        DD, DDp, _ = evolve_second_order_dispersion(ls=ls_, ks=ks_, inv_rhos=inv_rs_, \
                                                    plot=plot, ms=ms_list_, taus=taus_)
        if return_end_values:
            return beta_x, alpha_x, beta_y, alpha_y, D, Dp, R56, DD, DDp
        if return_figs:
            return fig1, fig2, fig3, fig4