from abel.classes.interstage import Interstage
import numpy as np
import os
from types import SimpleNamespace
import scipy.constants as SI

class InterstageImpactX(Interstage):
    
    def __init__(self, nom_energy=None, beta0=None, length_dipole=None, field_dipole=None, 
                num_slices=50, keep_data=False):
        
        super().__init__(nom_energy=nom_energy, beta0=beta0, length_dipole=length_dipole, field_dipole=field_dipole)

        # simulation options
        self.num_slices = num_slices
        self.keep_data = keep_data

    
    def get_lattice(self):

        import impactx
        
        # initialize lattice
        lattice = []

        # calculate the momentum
        p0 = np.sqrt((self.nom_energy*SI.e)**2-(SI.m_e*SI.c**2)**2)/SI.c
        
        # add beam diagnostics
        #monitor = impactx.elements.BeamMonitor("monitor", backend="h5")

        # gap drift (with monitors)
        gap = []
        if self.use_gaps:
            gap.append(impactx.elements.ExactDrift(ds=self.length_gap, nslice=1))
            #gap.append(monitor)
        
        # define dipole
        B_dip = self.field_dipole
        phi_dip = self.length_dipole*B_dip*SI.e/p0
        dipole = impactx.elements.ExactSbend(ds=self.length_dipole, phi=np.rad2deg(phi_dip), B=B_dip, nslice=self.num_slices)

        # define plasma lens
        kl_lens = self.strength_plasma_lens
        tau_lens = self.nonlinearity_plasma_lens
        plasma_lens = []
        if self.use_thick_lenses:
            drift_slice_pl = impactx.elements.ExactDrift(ds=self.length_plasma_lens/(self.num_slices+1), nslice=1)
            pl_slice = impactx.elements.TaperedPL(k=kl_lens/self.num_slices, taper=tau_lens)
            plasma_lens.extend([drift_slice_pl, pl_slice]*self.num_slices)
            plasma_lens.append(drift_slice_pl)
        else:
            plasma_lens.append(impactx.elements.TaperedPL(k=kl_lens, taper=tau_lens))

        # define first chicane dipole
        B_chic1 = self.field_chicane_dipole1
        phi_chic1 = self.length_chicane_dipole*B_chic1*(SI.e/p0)
        if abs(B_chic1) > 0:
            chicane_dipole1 = impactx.elements.ExactSbend(ds=self.length_chicane_dipole, phi=np.rad2deg(phi_chic1), B=B_chic1, nslice=self.num_slices)
        else:
            chicane_dipole1 = impactx.elements.ExactDrift(ds=self.length_chicane_dipole, nslice=self.num_slices)

        # define second chicane dipole
        B_chic2 = self.field_chicane_dipole2
        phi_chic2 = self.length_chicane_dipole*B_chic2*(SI.e/p0)
        if abs(B_chic2) > 0:
            chicane_dipole2 = impactx.elements.ExactSbend(ds=self.length_chicane_dipole, phi=np.rad2deg(phi_chic2), B=B_chic2, nslice=self.num_slices)
        else:
            chicane_dipole2 = impactx.elements.ExactDrift(ds=self.length_chicane_dipole, nslice=self.num_slices)

        # define sextupole
        ml_sext = self.strength_sextupole
        half_sextupole = []
        if self.use_thick_lenses:
            drift_slice_sext = impactx.elements.ExactDrift(ds=self.length_sextupole/(self.num_slices+1)/2, nslice=1)
            sext_slice = impactx.elements.Multipole(multipole=3, K_normal=ml_sext/self.num_slices, K_skew=0)
            half_sextupole.extend([drift_slice_sext, sext_slice]*self.num_slices)
            half_sextupole.append(drift_slice_sext)
        else:
            half_sextupole.append(impactx.elements.Multipole(multipole=3, K_normal=ml_sext, K_skew=0))
        
        # specify the lattice sequence
        lattice.extend(gap)
        lattice.append(dipole)
        lattice.extend(gap)
        lattice.extend(plasma_lens)
        lattice.extend(gap)
        lattice.append(chicane_dipole1)
        lattice.extend(gap)
        lattice.append(chicane_dipole2)
        lattice.extend(gap)
        lattice.extend(half_sextupole)
        lattice.extend(half_sextupole)
        lattice.extend(gap)
        lattice.append(chicane_dipole2)
        lattice.extend(gap)
        lattice.append(chicane_dipole1)
        lattice.extend(gap)
        lattice.extend(plasma_lens)
        lattice.extend(gap)
        lattice.append(dipole)
        lattice.extend(gap)
        
        return lattice

        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # re-perform the matching
        self.match_all()
        
        # get lattice
        lattice = self.get_lattice()
        
        # run ImpactX
        from abel.apis.impactx.impactx_api import run_impactx
        beam, self.evolution = run_impactx(lattice, beam0, verbose=False, runnable=runnable, keep_data=self.keep_data, 
                                 space_charge=self.enable_space_charge, csr=self.enable_csr, isr=self.enable_isr)
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def plot_layout(self):

        from matplotlib import pyplot as plt
        
        L1 = float(self.length_dipole)
        B1 = float(self.field_dipole)
        L2 = self.length_chicane_dipole*2
        B2 = float(int(self.use_chicane)*self.field_ratio_chicane*B1)
        B3 = -B2 - B1*(L1/L2)*(1-L1/L2)
        
        nparts = 25;
        
        rho1 = self.nom_energy/(SI.c*B1)
        
        x0 = 0
        y0 = 0
        theta0 = 0*L1/rho1

        
        ss1 = np.linspace(0,L1,nparts);
        thetas1 = ss1/abs(rho1)
        xs_bend1 = rho1*np.sin(thetas1)
        ys_bend1 = rho1*(1-np.cos(thetas1))
        xs1 = x0 + xs_bend1*np.cos(theta0) + ys_bend1*np.sin(theta0)
        ys1 = y0 - xs_bend1*np.sin(theta0) + ys_bend1*np.cos(theta0)
        xs = xs1
        ys = ys1
        theta1 = theta0-thetas1[-1]
        
        rho2 = self.nom_energy/(SI.c*B2)
        ss2 = np.linspace(0,L2/2,nparts);
        thetas2 = ss2/abs(rho2)
        xs_bend2 = rho2*np.sin(thetas2)
        ys_bend2 = rho2*(1-np.cos(thetas2))
        xs2 = xs1[-1] + xs_bend2*np.cos(theta1) + ys_bend2*np.sin(theta1)
        ys2 = ys1[-1] - xs_bend2*np.sin(theta1) + ys_bend2*np.cos(theta1)
        xs = np.append(xs,xs2)
        ys = np.append(ys,ys2)
        theta2 = theta1-thetas2[-1]

        rho3 = self.nom_energy/(SI.c*B3)
        ss3 = np.linspace(0,L2/2,nparts);
        thetas3 = ss3/abs(rho3)
        xs_bend3 = abs(rho3)*np.sin(thetas3)
        ys_bend3 = (rho3)*(1-np.cos(thetas3))
        xs3 = xs2[-1] + xs_bend3*np.cos(theta2) + ys_bend3*np.sin(theta2)
        ys3 = ys2[-1] - xs_bend3*np.sin(theta2) + ys_bend3*np.cos(theta2)
        xs = np.append(xs,xs3)
        ys = np.append(ys,ys3)
        theta3 = theta0+theta2-thetas3[-1]
        
        xs_rest = -(np.flip(xs) - xs[-1])
        ys_rest = (np.flip(ys) - ys[-1])
        xs4 = xs3[-1] + xs_rest*np.cos(theta3) + ys_rest*np.sin(theta3)
        ys4 = ys3[-1] - xs_rest*np.sin(theta3) + ys_rest*np.cos(theta3)
        xs = np.append(xs,xs4)
        ys = np.append(ys,ys4)
        
        # prepare plot
        fig, ax = plt.subplots(1)
        fig.set_figwidth(14)
        fig.set_figheight(6)

        # plot nominal orbit
        ax.plot(xs, ys, 'g')
        #ax.plot([x0, xs1[-1], xs2[-1], xs3[-1]], [y0, ys1[-1], ys2[-1], ys3[-1]], 'ko')
        ax.axis('equal')
        
    
        