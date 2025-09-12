import uuid, os, subprocess, csv, shutil
import numpy as np
from abel.CONFIG import CONFIG
from abel.classes.beam import Beam
import scipy.constants as SI


def elegant_run(filename, beam0, inputbeamfile, outputbeamfile, verbose=False, tmpfolder=None, runnable=None, save_beams=True):

    # convert incoming beam object to temporary SDDS file
    elegant_write_beam(beam0, inputbeamfile, tmpfolder=tmpfolder)

    # make evolution folder
    evolution_folder = tmpfolder + 'evolution' + os.sep
    os.mkdir(evolution_folder)

    # run system command
    cmd = CONFIG.elegant_exec + 'elegant ' + filename + CONFIG.elegant_rpnflag
    if verbose:
        stdout = subprocess.DEVNULL
    else:
        stdout = None
    subprocess.call(cmd, shell=True, stdout=stdout)
    
    # convert SDDS output to beam object
    beam = elegant_read_beam(outputbeamfile, tmpfolder=tmpfolder, model_beam=beam0)
    beam.location = beam0.location
    
    # save evolution
    evolution = extract_beams_and_evolution(tmpfolder, evolution_folder, runnable, save_beams=save_beams, model_beam=beam0)
    
    return beam, evolution


def elegant_read_beam(filename, tmpfolder=None, model_beam=None):

    from abel.utilities.relativity import gamma2energy
    
    make_new_tmpfolder = tmpfolder is None
    if make_new_tmpfolder:
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4())
        os.mkdir(tmpfolder)
    tmpfile =  tmpfolder + '/stream_' + str(uuid.uuid4()) + '.tmp'
    
    # convert SDDS file to CSV file
    subprocess.run(CONFIG.elegant_exec + 'sdds2stream ' + filename + ' -columns=x,xp,y,yp,t,p,dt > ' + tmpfile, shell=True)
    
    # extract charge
    Qabs = float(subprocess.check_output([CONFIG.elegant_exec + 'sdds2stream ' + filename + ' -parameter=Charge'], shell=True))  ######## Charge sign not set correctly?
    Q = abs(Qabs)*model_beam.charge_sign()
    
    # load phasespace from CSV file
    phasespace = np.loadtxt(open(tmpfile, "rb"), delimiter=' ')
    
    # delete CSV file and temporary folder
    os.remove(tmpfile)
    if make_new_tmpfolder:
        shutil.rmtree(tmpfolder)
    
    # make beam (note: z is flipped)
    beam = Beam()
    beam.set_phase_space(xs=phasespace[:,0], 
                       ys=phasespace[:,2], 
                       zs=-1*phasespace[:,6]*SI.c, 
                       xps=phasespace[:,1], 
                       yps=phasespace[:,3], 
                       Es=gamma2energy(phasespace[:,5]),
                       Q=Q) # TODO: deal with non-uniform weight particles
    beam.location = np.mean(phasespace[:,4])*SI.c
    
    # copy previous beam metadata
    if model_beam is not None:
        beam.location += model_beam.location
        beam.trackable_number = model_beam.trackable_number
        beam.stage_number = model_beam.stage_number
        beam.copy_particle_charge(model_beam)
    
    return beam


def elegant_write_beam(beam, filename, tmpfolder=None):

    from abel.utilities.relativity import energy2gamma
    
    # create temporary CSV file and folder
    make_new_tmpfolder = tmpfolder is None
    if make_new_tmpfolder:
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
        os.mkdir(tmpfolder)
    tmpfile = tmpfolder + 'beam_' + str(uuid.uuid4()) + '.csv'
    
    # write beam phasespace to CSV (note: z/t is flipped)
    M = np.matrix([beam.xs(),beam.xps(),beam.ys(),beam.yps(),-1*beam.ts(),beam.gammas(),beam.ts()])
    with open(tmpfile, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for i in range(int(len(beam))):
            csvwriter.writerow([M[0,i], M[1,i], M[2,i], M[3,i], M[4,i], M[5,i], M[6,i], int(i+1)])
    
    # convert CSV to SDDS (ascii for now)
    subprocess.call(CONFIG.elegant_exec + 'csv2sdds ' + tmpfile + ' ' + filename + 
                        ' -asciiOutput -columnData=name=x,type=double,units=m' +
                        ' -columnData=name=xp,type=double' + 
                        ' -columnData=name=y,type=double,units=m' + 
                        ' -columnData=name=yp,type=double' + 
                        ' -columnData=name=t,type=double,units=s' + 
                        ' -columnData=name=p,type=double,units=\"m\$be\$nc\"' +
                        ' -columnData=name=dt,type=double,units=s' +
                        ' -columnData=name=particleID,type=ulong64', shell=True)
    
    # add metadata
    with open(filename, 'r') as f:
        lines = f.readlines()

        line1 = 2
        lines.insert(line1, '&parameter name=SVNVersion, description="SVN version number", type=string, &end\n')
        lines.insert(line1, '&parameter name=IDSlotsPerBunch, description="Number of particle ID slots reserved to a bunch", type=long, &end\n')
        lines.insert(line1, '&parameter name=Particles, description="Number of particles before sampling", type=long, &end\n')
        lines.insert(line1, '&parameter name=Charge, units=C, description="Bunch charge before sampling", type=double, &end\n')
        lines.insert(line1, '&parameter name=pCentral, symbol="p$bcen$n", units="m$be$nc", description="Reference beta*gamma", type=double, &end\n')
        lines.insert(line1, '&parameter name=Step, description="Simulation step", type=long, &end\n')
        
        line2 = 18
        lines.insert(line2, '28584M'+'\n')
        lines.insert(line2, str(int(len(beam)))+'\n')
        lines.insert(line2, str(int(len(beam)))+'\n')
        lines.insert(line2, ' '+str(beam.charge())+'\n')
        lines.insert(line2, ' '+str(energy2gamma(beam.energy()))+'\n')
        lines.insert(line2, str(1)+'\n')
    
        with open(filename, 'w') as f:
            f.write("".join(lines))
        
    # convert SDDS to binary
    subprocess.run(CONFIG.elegant_exec + 'sddsconvert -binary ' + filename + ' -noWarnings', shell=True)
    
    # delete CSV file and temporary folder
    os.remove(tmpfile)
    if make_new_tmpfolder:
        shutil.rmtree(tmpfolder)
    
    return filename
    
    
# Extract the beams and evolution of various beam parameters as a function of s.
def extract_beams_and_evolution(tmpfolder, evolution_folder, runnable, save_beams=True, model_beam=None):
    
    insitu_path = tmpfolder + 'diags/insitu/'

    # prepare data structure
    from types import SimpleNamespace
    evol = SimpleNamespace()
    
    # run system command for converting .cen file to a .csv file
    cmd = CONFIG.elegant_exec + '/sdds2stream ' + tmpfolder + '/centroid_vs_s.cen -columns=s,Cx,Cy,Cxp,Cyp,Cs,Cdelta,Particles,pCentral,Charge,ElementName,ElementType,ElementOccurence >' + tmpfolder + '/centroids.csv'
    subprocess.call(cmd, shell=True)

    # load centroid data from .csv file. All quantities in SI units unless otherwise specified.
    file = open(tmpfolder + '/centroids.csv', "rb")
    data = np.loadtxt(file, delimiter=' ', usecols=range(10))  # Avoids extracting the columns containing strings such as ElementName.
    file.close()

    # make mask to only extract at monitor locations
    file = open(tmpfolder + '/centroids.csv', "rb")
    names = np.loadtxt(file, usecols=10, dtype='str')  # Avoids extracting the columns containing strings such as ElementName.
    file.close()
    mask = (names=='MONITOR')
    evol.location = data[mask,0]
    evol.x = data[mask,1]
    evol.y = data[mask,2]
    evol.xp = data[mask,3]
    evol.yp = data[mask,4]
    evol.z = data[mask,5]
    evol.energy = data[mask,8]*(1+data[mask,6])*SI.m_e*SI.c**2/SI.e  # [eV]
    evol.charge = data[mask,9]
    
    # extract from beam
    evol.beam_size_x = np.empty_like(evol.location)
    evol.beam_size_y = np.empty_like(evol.location)
    evol.beta_x = np.empty_like(evol.location)
    evol.beta_y = np.empty_like(evol.location)
    evol.dispersion_x = np.empty_like(evol.location)
    evol.dispersion_y = np.empty_like(evol.location)
    evol.bunch_length = np.empty_like(evol.location)
    evol.emit_nx = np.empty_like(evol.location)
    evol.emit_ny = np.empty_like(evol.location)
    evol.rel_energy_spread = np.empty_like(evol.location)
    
    for i, file in enumerate(sorted(os.listdir(evolution_folder))):

        # extract the beam
        beam_step = elegant_read_beam(evolution_folder + os.fsdecode(file), tmpfolder=tmpfolder, model_beam=model_beam)

        # save beam parameters
        evol.beam_size_x[i] = beam_step.beam_size_x()
        evol.beam_size_y[i] = beam_step.beam_size_y()
        evol.beta_x[i] = beam_step.beta_x()
        evol.beta_y[i] = beam_step.beta_y()
        evol.dispersion_x[i] = beam_step.dispersion_x()
        evol.dispersion_y[i] = beam_step.dispersion_y()
        evol.bunch_length[i] = beam_step.bunch_length()
        evol.emit_nx[i] = beam_step.norm_emittance_x()
        evol.emit_ny[i] = beam_step.norm_emittance_y()
        evol.rel_energy_spread[i] = beam_step.rel_energy_spread()
        
        # save beams if requested
        if runnable is not None and save_beams:
            beam_step.save(runnable=runnable)
            
    return evol
    

def elegant_apl_fieldmap2D(tau_lens, lensdim_x=2e-3, lensdim_y=2e-3, dx=0.0, dy=0.0, tmpfolder=None):
    
    # transverse dimensions
    xs = np.linspace(-lensdim_x, lensdim_x, 501)
    ys = np.linspace(-lensdim_y, lensdim_y, 501)
    
    # create map
    Bmap = np.zeros((len(xs)*len(ys), 4));
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            Bx = (y+dy) + tau_lens*(x+dx)*(y+dy)
            By = -((x+dx) + tau_lens*(((x+dx)**2 + (y+dy)**2)/2))
            Bmap[i + j*len(xs),:] = [x, y, Bx, By]

    # make temporary CSV file
    make_new_tmpfolder = tmpfolder is None
    if make_new_tmpfolder:
        tmpfolder = os.path.join(CONFIG.temp_path, str(uuid.uuid4()))
        os.mkdir(tmpfolder)
    tmpfile = os.path.join(tmpfolder, 'Bmap.csv')
       
    # save map to temp file
    np.savetxt(tmpfile, Bmap, delimiter=',')
    
    # convert SDDS to binary
    filename = os.path.join(tmpfolder, 'Bmap.sdds')
    subprocess.call(CONFIG.elegant_exec + 'csv2sdds ' + tmpfile + ' ' + filename + 
                        ' -columnData=name=x,type=double,unit=m' +
                        ' -columnData=name=y,type=double,unit=m' +
                        ' -columnData=name=Bx,type=double,unit=T' +
                        ' -columnData=name=By,type=double,unit=T', shell=True)
    
    # delete temporary CSV file and folder
    if make_new_tmpfolder:
        shutil.rmtree(tmpfolder)

    return filename


    