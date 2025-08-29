import numpy as np
from abel.classes.bds.bds import BeamDeliverySystem
import scipy.constants as SI
from scipy.optimize import minimize, root_scalar, NonlinearConstraint

class DriverDelaySystem_M(BeamDeliverySystem):

    def __init__(self, E_nom=2, delay_per_stage=1, length_stage=12, num_stages=2, ks=[], B_dipole=1, \
                 keep_data=False, enable_space_charge=False, enable_csr=False, enable_isr=False, layoutnr = 1,\
                 l_diag = 2, use_monitors=False, x0=10, l_dipole=None, lattice={}):
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
        self.lattice=lattice # lattice = {'element': (length, focusing/bending value), etc...}

        # Lattice-elements lengths
        self.l_quads = 0.1
        self.l_kick = 2
        self.l_gap = 0.25
        self.l_diag = l_diag
        self.l_dipole = l_dipole
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
    def horizontal_length(self, theta_inc=0):
        p = self.E_nom/SI.c
        L = 0
        for key, value in self.lattice.items():
            if key=='drift' or key=='quad':
                L += value[0]*np.cos(theta_inc)
                
            if key=='dipole':
                inv_r = value[1]/p # B/p, can be ±
                theta = value[0]*inv_r # L*inv_r
                L += 2/abs(inv_r)*np.sin(abs(theta)/2)*np.cos(theta_inc + theta/2)
                theta_inc += theta

        return L

    
    def get_elements_arrays(self, lengths=True, dipoles=False, quads=False):
        # self.lattice = {'element': (length, magnetic variable), ...}
        # Make the lists that are required for the evolution functions in beam_physics.py
        if lengths:
            ls = [value[0] for value in self.lattice.values()]
            yield ls

        if dipoles:
            Bs = [value[1] if key=='dipole' else 0 for key, value in self.lattice.items()]
            yield Bs

        if quads:
            ks = [value[1] if key=='quad' else 0 for key, value in self.lattice.items()]
            yield ks
    
    
    def list2lattice(self):
        import impactx
        lattice=[]
        p = self.E_nom/SI.c
        # Get the lists that are turned into impactx objects
        ls, Bs, ks = self.get_elements_arrays(lengths=True, dipoles=False, quads=True)
        phis = [ls[i]*Bs[i]/p for i in range(len(ls))] # phi = L/r = L*B/p

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