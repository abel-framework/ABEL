from opal import CONFIG, Interstage
from opal.apis.elegant.elegant_api import elegant_run, elegant_apl_fieldmap2D
from opal.utilities.beamphysics import evolveBetaFunction, evolveDispersion, evolveSecondOrderDispersion
from opal.utilities import SI
from string import Template
import scipy
import uuid, os
import numpy as np
import matplotlib.pyplot as plt

class InterstageELEGANT(Interstage):
    
    def __init__(self, E0 = None, beta0 = None, Ldip = None, Bdip = 1, Bdip2 = None, enableISR = True, enableCSR = True):
        self.E0 = E0
        self.beta0 = beta0
        self.Ldip = Ldip
        self.Bdip = Bdip
        self.Bdip2 = Bdip2
        self.g_max = 1000 # [T/m] 
        self.enableISR = enableISR
        self.enableCSR = enableCSR
        self.disableNonlinearity = False
        
        self.default_Bdip2_Bdip_ratio = 0.8
    
    # evaluate beta function 
    def initialBetaFunction(self):
        if callable(self.beta0):
            return self.beta0(self.E0)
        else:
            return self.beta0
        
    # evaluate dipole field
    def dipoleField(self):
        if callable(self.Bdip):
            return self.Bdip(self.E0)
        else:
            return self.Bdip
    
    # evaluate dipole length  
    def dipoleLength(self):
        if callable(self.Ldip):
            return self.Ldip(self.E0)
        else:
            return self.Ldip
    
    # spacer length
    def spacerLength(self):
        return 0.05 * self.dipoleLength()
    
    # plasma-lens length
    def lensLength(self):
        f = (self.dipoleLength()+2*self.spacerLength())/2
        k = self.g_max*SI.c/self.E0
        return 1/(k*f)
    
    # chicane dipole length
    def chicaneDipoleLength(self):
        return 0.6 * self.dipoleLength()
    
    # evaluate dipole field (or use default)
    def chicaneDipoleField(self):
        if self.Bdip2 is None:
            return self.dipoleField() * self.default_Bdip2_Bdip_ratio
        elif callable(self.Bdip2):
            return self.Bdip2(self.E0)
        else:
            return self.Bdip2
    
    # sextupole length
    def sextupoleLength(self):
        return 0.4 * self.dipoleLength()
    
    # full lattice 
    def fullLattice(self, g_lens=0, Bdip3=0, m_sext=0, tau_lens=0):
        
        # element length array
        dL = self.spacerLength()
        L_dip = self.dipoleLength()
        L_lens = self.lensLength()
        L_chic = self.chicaneDipoleLength()
        L_sext = self.sextupoleLength()
        ls = np.array([dL, L_dip, dL, L_lens, dL, L_chic, dL, L_chic, dL, L_sext, dL, L_chic, dL, L_chic, dL, L_lens, dL, L_dip, dL])
        
        # bending strength array
        Bdip = self.dipoleField()
        Bdip2 = self.chicaneDipoleField()
        inv_rhos = np.array([0, Bdip, 0, 0, 0, Bdip2, 0, Bdip3, 0, 0, 0, Bdip3, 0, Bdip2, 0, 0, 0, Bdip, 0]) * SI.c / self.E0
        
        # focusing strength array
        ks = np.array([0, 0, 0, g_lens, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, g_lens, 0, 0, 0]) * SI.c / self.E0
        
        # sextupole strength array
        ms = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, m_sext, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # plasma-lens transverse taper array
        taus = np.array([0, 0, 0, tau_lens, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tau_lens, 0, 0, 0])
        
        return ls, inv_rhos, ks, ms, taus
    
    
    # first half of the lattice (up to middle of the sextupole)
    def halfLattice(self, g_lens=0, Bdip3=0, m_sext=0, tau_lens=0):
        ls, inv_rhos, ks, ms, taus = self.fullLattice(g_lens, Bdip3, m_sext, tau_lens)
        inds = range(int(np.ceil(len(ls)/2)))
        ls_half = ls[inds]
        ls_half[-1] = ls_half[-1]/2
        return ls_half, inv_rhos[inds], ks[inds], ms[inds], taus[inds]
    
    
    # first quarter of the lattice (up to middle of first lens)
    def quarterLattice(self, g_lens=0, Bdip3=0, m_sext=0, tau_lens=0):
        ls, inv_rhos, ks, ms, taus = self.fullLattice(g_lens, Bdip3, m_sext, tau_lens)
        inds = range(4)
        ls_quart = ls[inds]
        ls_quart[-1] = ls_quart[-1]/2
        return ls_quart, inv_rhos[inds], ks[inds], ms[inds], taus[inds]
    
    
    # total length of lattice
    def length(self):
        ls, _, _, _, _ = self.fullLattice()
        return np.sum(ls)
        
    
    # make run script file
    def makeRunScript(self):
        return CONFIG.opal_path + 'opal/apis/elegant/templates/runscript_interstage.ele'
    
    
    def makeLattice(self, beam, output_filename, latticefile, lensfile, dumpBeams=False):
        
        # perform matching to find exact element strengths
        g_lens, tau_lens, Bdip3, m_sext = self.match()
        
        if self.disableNonlinearity:
            tau_lens = 0
            m_sext = 0
        
        # make lens field
        elegant_apl_fieldmap2D(tau_lens, lensfile)
        
        # make lattice file from template
        lattice_template = CONFIG.opal_path + 'opal/apis/elegant/templates/lattice_interstage.lte'
        Ndumps = 1
        
        # inputs
        inputs = {'charge': abs(beam.charge()),
                  'dipole_length': self.dipoleLength() / Ndumps,
                  'dipole_angle': self.dipoleLength()*self.dipoleField()*SI.c/self.E0 / Ndumps,
                  'chicanedipole_length': self.chicaneDipoleLength() / Ndumps,
                  'chicanedipole_angle1': self.chicaneDipoleLength()*self.chicaneDipoleField() * SI.c/self.E0 / Ndumps,
                  'chicanedipole_angle2': self.chicaneDipoleLength()*Bdip3*SI.c/self.E0 / Ndumps,
                  'lens_length': self.lensLength() / Ndumps,
                  'lens_filename': lensfile,
                  'lens_strength': g_lens,
                  'spacer_length': self.spacerLength() / Ndumps,
                  'sextupole_length': self.sextupoleLength() / Ndumps,
                  'sextupole_strength': m_sext,
                  'output_filename': output_filename,
                  'enable_ISR': int(self.enableISR),
                  'enable_CSR': int(self.enableCSR)}
        
        with open(lattice_template, 'r') as fin, open(latticefile, 'w') as fout:
            results = Template(fin.read()).substitute(inputs)
            fout.write(results)
    
    
    # calculate the required size of the 2D lens field map
    def findFieldMapSize(self, beam):
        
        # find beta function and dispersion in the lens
        ls, inv_rhos, ks, _, _ = self.quarterLattice()
        beta, _, _ = evolveBetaFunction(ls, ks, self.initialBetaFunction(), fast=True)
        Dx, _, _ = evolveDispersion(ls, inv_rhos, ks, fast=True)
        
        # calculate lens dimensions
        nsig = 5
        relEnergySpread_max = 0.05
        offset_max = 50e-6
        lensdim_x = abs(Dx*relEnergySpread_max) + nsig*np.sqrt(beta*beam.geomEmittanceX()) + offset_max
        lensdim_y = nsig*np.sqrt(beta*beam.geomEmittanceY()) + offset_max
        
        return lensdim_x, lensdim_y
        
        
    # match the beta function, first- and second-order dispersions
    def match(self):
        
        # define half lattice
        ls_half, _, _, _, _ = self.halfLattice()
        Bdip = self.dipoleField()
        Bdip2 = self.chicaneDipoleField()
        inv_rhos_half = lambda B3: np.array([0, Bdip, 0, 0, 0, Bdip2, 0, B3, 0, 0]) * SI.c/self.E0
        ks_half = lambda g: np.array([0, 0, 0, g, 0, 0, 0, 0, 0, 0]) * SI.c/self.E0
        ms_half = lambda m: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, m])
        taus_half = lambda tau: np.array([0, 0, 0, tau, 0, 0, 0, 0, 0, 0])
        
        # minimizer function for beta matching (central alpha function is zero)
        def minfun_beta(params):
            _, alpha, _ = evolveBetaFunction(ls_half, ks_half(params[0]), self.initialBetaFunction(), fast=True)         
            return alpha**2
        
        # match the beta function
        result_beta = scipy.optimize.minimize(minfun_beta, self.g_max, tol=1e-20)
        g_lens = result_beta.x[0]
        
        # minimizer function for first-order dispersion (central dispersion prime is zero)
        def minfun_Dx(p):
            _, Dpx, _ = evolveDispersion(ls_half, inv_rhos_half(p[0]), ks_half(g_lens), fast=True)         
            return Dpx**2
        
        # match the first-order dispersion
        Bdip3_guess = -Bdip2;
        result_Dx = scipy.optimize.minimize(minfun_Dx, Bdip3_guess, tol=1e-20)
        Bdip3 = result_Dx.x[0]
        
        # calculate the required transverse-taper gradient
        ls_quart, _, _, _, _ = self.quarterLattice()
        inv_rhos_quart = np.array([0, Bdip, 0, 0]) * SI.c/self.E0
        ks_quart = np.array([0, 0, 0, g_lens]) * SI.c/self.E0
        Dx_lens, _, _ = evolveDispersion(ls_quart, inv_rhos_quart, ks_quart, fast=True)
        tau_lens = 1/Dx_lens
        
        # minimizer function for second-order dispersion (central second-order dispersion prime is zero)
        def minfun_DDx(p):
            _, DDpx, _ = evolveSecondOrderDispersion(ls_half, inv_rhos_half(Bdip3), ks_half(g_lens), ms_half(p[0]), taus_half(tau_lens), fast=True)
            return DDpx**2
        
        # match the second-order dispersion
        m_guess = 4*tau_lens/self.sextupoleLength()
        result_DDx = scipy.optimize.minimize(minfun_DDx, m_guess, method='Nelder-Mead', tol=1e-20, options={'maxiter': 50})
        m_sext = result_DDx.x[0]
        
        # plot results
        if False:
            ls, inv_rhos, ks, ms, taus = self.fullLattice(g_lens, Bdip3, m_sext, tau_lens)
            _, _, evolution_beta = evolveBetaFunction(ls, ks, self.initialBetaFunction())
            _, _, evolution_Dx = evolveDispersion(ls, inv_rhos, ks)
            _, _, evolution_DDx = evolveSecondOrderDispersion(ls, inv_rhos, ks, ms, taus)

            # prepare figure
            fig, axs = plt.subplots(1,3)
            fig.set_figwidth(20)
            fig.set_figheight(4)
            axs[0].plot(evolution_beta[0,:], evolution_beta[1,:])
            axs[1].plot(evolution_Dx[0,:], evolution_Dx[1,:])
            axs[1].plot(evolution_DDx[0,:], evolution_DDx[1,:])
            axs[2].plot(evolution_DDx[0,:], evolution_DDx[2,:])
        
        return g_lens, tau_lens, Bdip3, m_sext
        

    # track a beam through the lattice using ELEGANT
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):
        
        # make temporary folder and files
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4())
        os.mkdir(tmpfolder)
        beamfile = tmpfolder + '/beam.bun'
        latticefile = tmpfolder + '/interstage.lte'
        lensfile = tmpfolder + '/map.csv'
        
        # make lattice file
        self.makeLattice(beam0, beamfile, latticefile, lensfile)
        
        # environment variables
        envars = {}
        envars['ENERGY'] = self.E0 / 1e6 # [MeV]
        envars['LATTICE'] = latticefile
        
        # run ELEGANT
        runfile = self.makeRunScript()
        beam = elegant_run(runfile, beam0, beamfile, envars, quiet=True)
        
        # remove temporary files
        os.remove(beamfile)
        os.remove(latticefile)
        os.remove(lensfile)
        beamfile_backup = beamfile + '~'
        if os.path.exists(beamfile_backup):
            os.remove(beamfile_backup)
        os.rmdir(tmpfolder)

        return super().track(beam, savedepth, runnable, verbose)
    