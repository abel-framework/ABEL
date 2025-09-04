from abel.classes.spectrometer import Spectrometer
import scipy.constants as SI
import numpy as np

class SpectrometerQuadImaging(Spectrometer):
    
    def __init__(self, angle_dipole=None, length_dipole=None, imaging_energy_x=None, imaging_energy_y=None, object_plane_x=None, object_plane_y=None, magnification_x=None, magnification_y=None):

        super().__init__()
        
        self.imaging_energy_x = imaging_energy_x
        self.imaging_energy_y = imaging_energy_y
        self.object_plane_x = object_plane_x
        self.object_plane_y = object_plane_y
        self.magnification_x = magnification_x
        self.magnification_y = magnification_y

        self.angle_dipole = angle_dipole
        self.length_dipole = length_dipole

        self.strengths_quads = None
        self.length_quad = None

        self.length_drifts = None

    
    # lattice length
    def get_length(self):
        ls, *_ = self.matrix_lattice()
        return np.sum(ls)

        
    @property
    def field_dipole(self) -> float:
        return self.angle_dipole*self.nom_energy/(self.length_dipole*SI.c)
        
        
        
    # full lattice 
    def matrix_lattice(self, ks_quads=None):
        
        # construct length list
        ls = [self.length_drifts[0]]
        for i in range(len(self.length_drifts)-2):
            ls.append(self.length_quad)
            ls.append(self.length_drifts[i+1])
        ls.append(self.length_dipole)
        ls.append(self.length_drifts[-1])
        ls = np.array(ls)
        
        # bending strength array
        inv_rhos = np.zeros_like(ls)
        inv_rhos[-2] = self.angle_dipole/self.length_dipole
        
        # focusing strength array
        if ks_quads is None:
            ks_quads = self.strengths_quads
        ks = np.zeros_like(ls)
        for i in range(len(ks_quads)):
            ks[2*i+1] = ks_quads[i]
        
        # sextupole strength array
        ms = np.zeros_like(ls)

        # plasma-lens transverse taper array
        taus = np.zeros_like(ls)
        
        return ls, inv_rhos, ks, ms, taus

    
    def set_imaging(self):


        # minimizer function for point-to-point imaging
        from abel.utilities.beam_physics import evolve_transfer_matrix
        
        def minfun_point2point_imaging(params):
            
            ls, _, ks, _, _ = self.matrix_lattice(ks_quads=params)
            
            ks_x = ks*self.nom_energy/self.imaging_energy_x
            Rx = evolve_transfer_matrix(ls, ks_x) 
            ks_y = ks*self.nom_energy/self.imaging_energy_x
            Ry = evolve_transfer_matrix(ls, ks_y) 
            
            R12 = Rx[0,1]
            R34 = Ry[2,3]
            magx = Rx[0,0]
            
            return R12**2 + R34**2 + (magx-self.magnification_x)**2
    
        # perform minization (find k-values)
        ks_guess = [1,-1,1]
        from scipy.optimize import minimize
        result = minimize(minfun_point2point_imaging, ks_guess, tol=1e-5, options={'maxiter': 1000})
        
        # set solution to quads
        if result.fun < 1e-5:
            
            # scale quadrupole strengths to the nominal energy
            self.strengths_quads = result.x
                
        else:
            raise Exception('No imaging solution found.')


    ## PLOTTING OPTICS

    def plot_optics(self, show_beta_function=True, show_dispersion=False):

        from matplotlib import pyplot as plt
        from matplotlib import patches
        from copy import deepcopy
        from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion
        
        # calculate evolution
        ls, inv_rhos, ks, _, _ = self.matrix_lattice()
        ssl = np.append([0.0], np.cumsum(ls))

        beta0 = 0.01

        if show_beta_function:
            _, _, evol_beta_x = evolve_beta_function(ls, ks, beta0, fast=False)
            _, _, evol_beta_y = evolve_beta_function(ls, -ks, beta0, fast=False)
            ss_beta = evol_beta_x[0]
            beta_xs = evol_beta_x[1]
            beta_ys = evol_beta_y[1]

        # prepare plots
        num_plots = 1 + int(show_beta_function) + int(show_dispersion)
        height_ratios = np.ones((num_plots,1))
        height_ratios[0] = 0.1
        fig, axs = plt.subplots(num_plots,1, gridspec_kw={'height_ratios': height_ratios})
        fig.set_figwidth(7)
        fig.set_figheight(11/3.1*np.sum(height_ratios))
        col0 = "tab:gray"
        colx1 = "tab:blue"
        coly = "tab:orange"
        colx2 = "#d7e9f5" # lighter version of tab:blue
        colz = "tab:green"
        coloff = "#e69596" # lighter version of tab:red
        long_label = 'Location (m)'
        long_limits = [min(ssl), max(ssl)]

        # layout
        n = 0
        axs[n].plot(ssl, np.zeros_like(ssl), '-', linewidth=0.5, color='k')
        axs[n].axis('off')
        for i in range(len(ls)):
            if abs(inv_rhos[i]) > 0: # add dipoles
                axs[n].add_patch(patches.Rectangle((ssl[i],-0.75), ls[i], 1.5, fc='#d9d9d9'))
            if abs(ks[i]) > 0: # add quad or plasma lenses
                axs[n].add_patch(patches.Rectangle((ssl[i],0), ls[i], np.sign(ks[i]), fc='#fcb577'))
        axs[n].set_xlim(long_limits)
        axs[n].set_ylim([-1, 1])

        # shift the layout box down
        box = axs[0].get_position()
        vshift = 0.025
        box.y0 = box.y0 - vshift
        box.y1 = box.y1 - vshift
        axs[0].set_position(box)
        
        # plot beta function
        if show_beta_function:
            n += 1
            axs[n].plot(ss_beta, np.sqrt(beta0*np.ones_like(ss_beta)), ':', color=col0)
            axs[n].plot(ss_beta, np.sqrt(beta_ys), color=coly, label=r'$y$')
            axs[n].plot(ss_beta, np.sqrt(beta_xs), color=colx1, label=r'$x$')
            axs[n].legend(loc='best', reverse=True, fontsize='small')
            axs[n].set_ylabel(r'$\sqrt{\mathrm{Beta\hspace{0.3}function}}$ ($\sqrt{\mathrm{m}})$')
            axs[n].set_xlim(long_limits)
        
        # plot dispersion
        if show_dispersion:
            pass
        
        # add horizontal axis label
        axs[n].set_xlabel(long_label)
        
           
        
        