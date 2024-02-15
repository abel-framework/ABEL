import uuid, os, scipy, shutil
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as SI
from string import Template
from types import SimpleNamespace
from abel import CONFIG, Interstage
from abel.apis.elegant.elegant_api import elegant_run, elegant_apl_fieldmap2D, elegant_read_beam
from abel.utilities.beam_physics import evolve_beta_function, evolve_dispersion, evolve_second_order_dispersion

class InterstageElegant(Interstage):
    
    def __init__(self, nom_energy=None, beta0=None, dipole_length=None, dipole_field=1, Bdip2=None, enable_isr=True, enable_csr=True, save_evolution=False):
        self.nom_energy = nom_energy
        self.beta0 = beta0
        self.dipole_length = dipole_length
        self.dipole_field = dipole_field
        self.Bdip2 = Bdip2
        self.g_max = 1000 # [T/m] 
        self.disableNonlinearity = False
        
        self.default_Bdip2_Bdip_ratio = 0.8

        self.enable_isr = enable_isr
        self.enable_csr = enable_csr
        self.save_evolution = save_evolution
        
        self.evolution = SimpleNamespace()
    
    # evaluate beta function 
    def __eval_initial_beta_function(self):
        if callable(self.beta0):
            return self.beta0(self.nom_energy)
        else:
            return self.beta0
        
    # evaluate dipole field
    def __eval_dipole_field(self):
        if callable(self.dipole_field):
            return self.dipole_field(self.nom_energy)
        else:
            return self.dipole_field
    
    # evaluate dipole length  
    def __eval_dipole_length(self):
        if callable(self.dipole_length):
            return self.dipole_length(self.nom_energy)
        else:
            return self.dipole_length
    
    # spacer length
    def __eval_spacer_length(self):
        return 0.05 * self.__eval_dipole_length()
    
    # plasma-lens length
    def __eval_lens_length(self):
        f = (self.__eval_dipole_length()+2*self.__eval_spacer_length())/2
        k = self.g_max*SI.c/self.nom_energy
        return 1/(k*f)
    
    # chicane dipole length
    def __eval_chicane_dipole_length(self):
        return 0.6 * self.__eval_dipole_length()
    
    # evaluate dipole field (or use default)
    def __eval_chicane_field(self):
        if self.Bdip2 is None:
            return self.__eval_dipole_field() * self.default_Bdip2_Bdip_ratio
        elif callable(self.Bdip2):
            return self.Bdip2(self.nom_energy)
        else:
            return self.Bdip2
    
    # sextupole length
    def __eval_sextupole_length(self):
        return 0.4 * self.__eval_dipole_length()
    
    # full lattice 
    def __full_lattice(self, g_lens=0, Bdip3=0, m_sext=0, tau_lens=0):
        
        # element length array
        dL = self.__eval_spacer_length()
        L_dip = self.__eval_dipole_length()
        L_lens = self.__eval_lens_length()
        L_chic = self.__eval_chicane_dipole_length()
        L_sext = self.__eval_sextupole_length()
        ls = np.array([dL, L_dip, dL, L_lens, dL, L_chic, dL, L_chic, dL, L_sext, dL, L_chic, dL, L_chic, dL, L_lens, dL, L_dip, dL])
        
        # bending strength array
        Bdip = self.__eval_dipole_field()
        Bdip2 = self.__eval_chicane_field()
        inv_rhos = np.array([0, Bdip, 0, 0, 0, Bdip2, 0, Bdip3, 0, 0, 0, Bdip3, 0, Bdip2, 0, 0, 0, Bdip, 0]) * SI.c / self.nom_energy
        
        # focusing strength array
        ks = np.array([0, 0, 0, g_lens, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, g_lens, 0, 0, 0]) * SI.c / self.nom_energy
        
        # sextupole strength array
        ms = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, m_sext, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # plasma-lens transverse taper array
        taus = np.array([0, 0, 0, tau_lens, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tau_lens, 0, 0, 0])
        
        return ls, inv_rhos, ks, ms, taus
    
    
    # first half of the lattice (up to middle of the sextupole)
    def __half_lattice(self, g_lens=0, Bdip3=0, m_sext=0, tau_lens=0):
        ls, inv_rhos, ks, ms, taus = self.__full_lattice(g_lens, Bdip3, m_sext, tau_lens)
        inds = range(int(np.ceil(len(ls)/2)))
        ls_half = ls[inds]
        ls_half[-1] = ls_half[-1]/2
        return ls_half, inv_rhos[inds], ks[inds], ms[inds], taus[inds]
    
    
    # first quarter of the lattice (up to middle of first lens)
    def __quarter_lattice(self, g_lens=0, Bdip3=0, m_sext=0, tau_lens=0):
        ls, inv_rhos, ks, ms, taus = self.__full_lattice(g_lens, Bdip3, m_sext, tau_lens)
        inds = range(4)
        ls_quart = ls[inds]
        ls_quart[-1] = ls_quart[-1]/2
        return ls_quart, inv_rhos[inds], ks[inds], ms[inds], taus[inds]
    
    
    # total length of lattice
    def get_length(self):
        ls, _, _, _, _ = self.__full_lattice()
        return np.sum(ls)
        
    
    # make run script file
    def __make_run_script(self, latticefile, inputbeamfile, tmpfolder=None):

        # create temporary CSV file and folder
        make_new_tmpfolder = tmpfolder is None
        if make_new_tmpfolder:
            tmpfolder = CONFIG.temp_path + str(uuid.uuid4())
            os.mkdir(tmpfolder)
        tmpfile = tmpfolder + '/runfile.ele'
        
        # inputs
        inputs = {'p_central_mev': self.nom_energy/1e6,
                  'latticefile': latticefile,
                  'inputbeamfile': inputbeamfile}

        runfile_template = CONFIG.abel_path + 'abel/apis/elegant/templates/runscript_interstage.ele'
        with open(runfile_template, 'r') as fin, open(tmpfile, 'w') as fout:
            results = Template(fin.read()).substitute(inputs)
            fout.write(results)
        
        return tmpfile
    
    
    def __make_lattice(self, beam, outputbeamfile, latticefile, lensfile, evolutionfolder, save_evolution=False, tmpfolder=None):
        
        # perform matching to find exact element strengths
        g_lens, tau_lens, Bdip3, m_sext = self.match()
        
        if self.disableNonlinearity:
            tau_lens = 0
            m_sext = 0
        
        # make lens field
        elegant_apl_fieldmap2D(tau_lens, lensfile, tmpfolder=tmpfolder)
        
        # make lattice file from template
        lattice_template = CONFIG.abel_path + 'abel/apis/elegant/templates/lattice_interstage.lte'
        if save_evolution:
            num_watches = 5
            watch_disabled = False
        else:
            num_watches = 1
            watch_disabled = True
        
        # inputs
        inputs = {'charge': abs(beam.charge()),
                  'num_kicks': int(100/num_watches),
                  'dipole_length': self.__eval_dipole_length()/num_watches,
                  'dipole_angle': self.__eval_dipole_length()*self.__eval_dipole_field()*SI.c/self.nom_energy/num_watches,
                  'chicanedipole_length': self.__eval_chicane_dipole_length()/num_watches,
                  'chicanedipole_angle1': self.__eval_chicane_dipole_length()*self.__eval_chicane_field() * SI.c/self.nom_energy/num_watches,
                  'chicanedipole_angle2': self.__eval_chicane_dipole_length()*Bdip3*SI.c/self.nom_energy/num_watches,
                  'lens_length': self.__eval_lens_length()/num_watches,
                  'lens_filename': lensfile,
                  'lens_strength': g_lens,
                  'spacer_length': self.__eval_spacer_length()/num_watches,
                  'sextupole_length': self.__eval_sextupole_length()/num_watches,
                  'sextupole_strength': m_sext,
                  'output_filename': outputbeamfile,
                  'enable_ISR': int(self.enable_isr),
                  'enable_CSR': int(self.enable_csr),
                  'num_out': int(num_watches),
                  'watch_filename': evolutionfolder + 'output_%03ld.bun',
                  'watch_disabled': int(watch_disabled)}
        
        with open(lattice_template, 'r') as fin, open(latticefile, 'w') as fout:
            results = Template(fin.read()).substitute(inputs)
            fout.write(results)
    
        
    # match the beta function, first- and second-order dispersions
    def match(self):
        
        # define half lattice
        ls_half, _, _, _, _ = self.__half_lattice()
        Bdip = self.__eval_dipole_field()
        Bdip2 = self.__eval_chicane_field()
        inv_rhos_half = lambda B3: np.array([0, Bdip, 0, 0, 0, Bdip2, 0, B3, 0, 0]) * SI.c/self.nom_energy
        ks_half = lambda g: np.array([0, 0, 0, g, 0, 0, 0, 0, 0, 0]) * SI.c/self.nom_energy
        ms_half = lambda m: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, m])
        taus_half = lambda tau: np.array([0, 0, 0, tau, 0, 0, 0, 0, 0, 0])
        
        # minimizer function for beta matching (central alpha function is zero)
        def minfun_beta(params):
            _, alpha, _ = evolve_beta_function(ls_half, ks_half(params[0]), self.__eval_initial_beta_function(), fast=True)         
            return alpha**2
        
        # match the beta function
        result_beta = scipy.optimize.minimize(minfun_beta, self.g_max, tol=1e-20)
        g_lens = result_beta.x[0]
        
        # minimizer function for first-order dispersion (central dispersion prime is zero)
        def minfun_Dx(p):
            _, Dpx, _ = evolve_dispersion(ls_half, inv_rhos_half(p[0]), ks_half(g_lens), fast=True)         
            return Dpx**2
        
        # match the first-order dispersion
        Bdip3_guess = -Bdip2;
        result_Dx = scipy.optimize.minimize(minfun_Dx, Bdip3_guess, tol=1e-20)
        Bdip3 = result_Dx.x[0]
        
        # calculate the required transverse-taper gradient
        ls_quart, _, _, _, _ = self.__quarter_lattice()
        inv_rhos_quart = np.array([0, Bdip, 0, 0]) * SI.c/self.nom_energy
        ks_quart = np.array([0, 0, 0, g_lens]) * SI.c/self.nom_energy
        Dx_lens, _, _ = evolve_dispersion(ls_quart, inv_rhos_quart, ks_quart, fast=True)
        tau_lens = 1/Dx_lens
        
        # minimizer function for second-order dispersion (central second-order dispersion prime is zero)
        def minfun_DDx(p):
            _, DDpx, _ = evolve_second_order_dispersion(ls_half, inv_rhos_half(Bdip3), ks_half(g_lens), ms_half(p[0]), taus_half(tau_lens), fast=True)
            return DDpx**2
        
        # match the second-order dispersion
        m_guess = 4*tau_lens/self.__eval_sextupole_length()
        result_DDx = scipy.optimize.minimize(minfun_DDx, m_guess, method='Nelder-Mead', tol=1e-20, options={'maxiter': 50})
        m_sext = result_DDx.x[0]
        
        # plot results
        if False:
            ls, inv_rhos, ks, ms, taus = self.__full_lattice(g_lens, Bdip3, m_sext, tau_lens)
            _, _, evolution_beta = evolve_beta_function(ls, ks, self.__eval_initial_beta_function())
            _, _, evolution_Dx = evolve_dispersion(ls, inv_rhos, ks)
            _, _, evolution_DDx = evolve_second_order_dispersion(ls, inv_rhos, ks, ms, taus)

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
        parent_dir = CONFIG.temp_path
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        
        # create the temporary folder
        tmpfolder = os.path.join(parent_dir, str(uuid.uuid4())) + '/'
        os.mkdir(tmpfolder)
        inputbeamfile = tmpfolder + 'input_beam.bun'
        outputbeamfile = tmpfolder + 'output_beam.bun'
        latticefile = tmpfolder + 'interstage.lte'
        lensfile = tmpfolder + 'map.csv'
        evolution_folder = tmpfolder + 'evolution/'
        os.mkdir(evolution_folder)
        
        # make lattice file
        self.__make_lattice(beam0, outputbeamfile, latticefile, lensfile, evolution_folder, self.save_evolution, tmpfolder=tmpfolder)
        
        # run ELEGANT
        runfile = self.__make_run_script(latticefile, inputbeamfile, tmpfolder)
        beam = elegant_run(runfile, beam0, inputbeamfile, outputbeamfile, quiet=True, tmpfolder=tmpfolder)

        # extract evolution
        self.evolution.location = np.empty([0])
        self.evolution.beam_size_x = np.empty([0])
        self.evolution.beam_size_y = np.empty([0])
        self.evolution.norm_emittance_x = np.empty([0])
        self.evolution.norm_emittance_y = np.empty([0])
        for file in sorted(os.listdir(evolution_folder)):
            output_beam = elegant_read_beam(evolution_folder + os.fsdecode(file), tmpfolder=tmpfolder)
            self.evolution.location = np.append(self.evolution.location, output_beam.location)
            self.evolution.beam_size_x = np.append(self.evolution.beam_size_x, output_beam.beam_size_x())
            self.evolution.beam_size_y = np.append(self.evolution.beam_size_y, output_beam.beam_size_y())
            self.evolution.norm_emittance_x = np.append(self.evolution.norm_emittance_x, output_beam.norm_emittance_x())
            self.evolution.norm_emittance_y = np.append(self.evolution.norm_emittance_y, output_beam.norm_emittance_y())
        
        # clean extreme outliers
        beam.remove_halo_particles()
        
        # remove temporary files
        shutil.rmtree(tmpfolder)

        return super().track(beam, savedepth, runnable, verbose)
    