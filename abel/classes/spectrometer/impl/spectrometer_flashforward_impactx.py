from abel.classes.spectrometer.spectrometer import Spectrometer
from types import SimpleNamespace
import scipy.constants as SI
import numpy as np

class SpectrometerFLASHForwardImpactX(Spectrometer):
    
    def __init__(self, imaging_energy_x=None, imaging_energy_y=None, object_plane_x=0.0, object_plane_y=None, magnification_x=-8.0):

        super().__init__()

        self.imaging_energy_x = imaging_energy_x
        self.imaging_energy_y = imaging_energy_y
        self.object_plane_x = object_plane_x
        self.object_plane_y = object_plane_y
        self.magnification_x = magnification_x
        
        self.current_dipole = -200 # [A]
        self.current_quad1 = 60 # [A]
        self.current_quad2 = -60 # [A]
        self.current_quad3 = 50 # [A]
        # TODO: implement imaging (using regular matrix tracking, to be added as a utility)

        # pre-defined lengths
        self.length_dipole = 1.07 # [m]
        self.length_quad = 0.1137 # [m]

        # beam line positions (quad and dipole centers)
        self.s_CELLCENTRE = 209.932261 # [m]
        self.s_Q11FLFDIAG = 210.590261 # [m]
        self.s_Q12FLFDIAG = 210.856261 # [m]
        self.s_Q21FLFDIAG = 211.122261 # [m]
        self.s_Q22FLFDIAG = 211.388261 # [m]
        self.s_Q23FLFDIAG = 211.768178 # [m]
        self.s_dipoleLEMS = 215.686200 # [m]
        self.s_LEMS       = 217.045605 # [m] image plane
        
        self.evolution = SimpleNamespace()
        
    # lattice length
    def get_length(self):
        return self.s_LEMS - self.s_CELLCENTRE

    def current2strength_quad(self, I, p0):
        g = I * 0.9 # TODO: improve conversion
        k = g*SI.e/p0
        return k

    def current2field_dipole(self, I):
        return I * 0.0014 # TODO: improve conversion

    def get_dispersion(self, energy=None):
        if energy is None:
            energy = self.imaging_energy_x
        gamma0x = energy*SI.e/(SI.m_e*SI.c**2);
        p0x = SI.m_e*gamma0x*SI.c*np.sqrt(1-1/gamma0x**2);
        Bdip = self.current2field_dipole(self.current_dipole)
        phi = Bdip*self.length_dipole*SI.e/p0x
        return (self.s_LEMS-self.s_dipoleLEMS)*phi
        
        
    # get the Ocelot lattice
    def get_lattice(self, object_plane_x=None, object_plane_y=None, imaging_energy_x=None, imaging_energy_y=None):

        # do imports
        import impactx
        
        # set the object planes
        if object_plane_x is None:
            object_plane_x = self.object_plane_x
        if object_plane_y is None:
            if self.object_plane_y is not None:
                object_plane_y = self.object_plane_y
            else:
                object_plane_y = object_plane_x

        # set the imaging energies
        if imaging_energy_x is None:
            imaging_energy_x = self.imaging_energy_x
        if imaging_energy_y is None:
            if self.imaging_energy_y is not None:
                imaging_energy_y = self.imaging_energy_y
            else:
                imaging_energy_y = imaging_energy_x
        
        # initialize lattice
        lattice = []
        
        # number of slices per ds in the element
        ns = 25  

        gamma0x = imaging_energy_x*SI.e/(SI.m_e*SI.c**2);
        p0x = SI.m_e*gamma0x*SI.c*np.sqrt(1-1/gamma0x**2);
        gamma0y = imaging_energy_y*SI.e/(SI.m_e*SI.c**2);
        p0y = SI.m_e*gamma0x*SI.c*np.sqrt(1-1/gamma0x**2);
        
        # define dipole
        Bdip = self.current2field_dipole(self.current_dipole)
        phi = Bdip*self.length_dipole*SI.e/p0x
        dipole = impactx.elements.ExactSbend(name="dipole", ds=self.length_dipole, phi=np.rad2deg(phi), B=Bdip, nslice=ns, rotation=90)
        
        # define quads
        quad1 = impactx.elements.ExactQuad(name='quad1', ds=self.length_quad, k=self.current2strength_quad(self.current_quad1, p0x))
        quad2 = impactx.elements.ExactQuad(name="quad2", ds=self.length_quad, k=self.current2strength_quad(self.current_quad2, p0x))
        quad3 = impactx.elements.ExactQuad(name="quad3", ds=self.length_quad, k=self.current2strength_quad(self.current_quad3, p0x))
        
        # derived separations
        d1 = self.s_Q11FLFDIAG - self.s_CELLCENTRE - self.length_quad/2 - object_plane_x;
        d2 = self.s_Q12FLFDIAG - self.s_Q11FLFDIAG - self.length_quad;
        d3 = self.s_Q21FLFDIAG - self.s_Q12FLFDIAG - self.length_quad;
        d4 = self.s_Q22FLFDIAG - self.s_Q21FLFDIAG - self.length_quad;
        d5 = self.s_Q23FLFDIAG - self.s_Q22FLFDIAG - self.length_quad;
        d6 = self.s_dipoleLEMS - self.s_Q23FLFDIAG - self.length_quad/2 - self.length_dipole/2;
        d7 = self.s_LEMS       - self.s_dipoleLEMS - self.length_dipole/2;
        
        # drifts
        drift1 = impactx.elements.ExactDrift(name="drift1", ds=d1, nslice=ns)
        drift2 = impactx.elements.ExactDrift(name="drift2", ds=d2, nslice=ns)
        drift3 = impactx.elements.ExactDrift(name="drift3", ds=d3, nslice=ns)
        drift4 = impactx.elements.ExactDrift(name="drift4", ds=d4, nslice=ns)
        drift5 = impactx.elements.ExactDrift(name="drift5", ds=d5, nslice=ns)
        drift6 = impactx.elements.ExactDrift(name="drift6", ds=d6, nslice=ns)
        drift7 = impactx.elements.ExactDrift(name="drift7", ds=d7, nslice=ns)
        
        # specify the lattice sequence
        lattice.append(drift1)
        lattice.append(quad1)
        lattice.append(drift2)
        lattice.append(quad1)
        lattice.append(drift3)
        lattice.append(quad2)
        lattice.append(drift4)
        lattice.append(quad2)
        lattice.append(drift5)
        lattice.append(quad3)
        lattice.append(drift6)
        lattice.append(dipole)
        lattice.append(drift7)
        
        return lattice

    
    
    # tracking function
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        # set imaging energy automatically
        if self.imaging_energy_x is None:
            self.imaging_energy_x = beam0.energy()
        
        # get lattice
        lattice = self.get_lattice()
        
        # run ImpactX
        from abel.apis.impactx.impactx_api import run_impactx
        beam, evol = run_impactx(lattice, beam0, nom_energy=self.imaging_energy_x, verbose=False, runnable=runnable)
        
        # save evolution
        self.evolution = evol
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def plot_evolution(self):

        # do imports
        import matplotlib.pyplot as plt
        
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
        
        # plot transverse offset
        axs[2,0].plot(evol.location, evol.y*1e6, color=col2)  
        axs[2,0].plot(evol.location, evol.y*1e6, color=col1)
        axs[2,0].set_ylabel(r'Transverse offset [$\mathrm{\mu}$m]')
        axs[2,0].set_xlabel(long_label)
        axs[2,0].set_xlim(long_limits)
        
        # plot dispersion
        axs[2,1].plot(evol.location, evol.dispersion_y*1e3, color=col2)  
        axs[2,1].plot(evol.location, evol.dispersion_x*1e3, color=col1)
        axs[2,1].plot(evol.location, evol.second_order_dispersion_x*1e3, ':', color=col1)
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