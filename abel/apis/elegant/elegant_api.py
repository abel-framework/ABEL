import uuid, os, subprocess, csv, shutil
import numpy as np
from abel import CONFIG, Beam
import scipy.constants as SI
from abel.utilities.relativity import gamma2energy, energy2gamma

def elegant_read_beam(filename):
    
    # create temporary stream file and folder
    tmpfolder = CONFIG.temp_path + str(uuid.uuid4())
    os.mkdir(tmpfolder)
    tmpfile =  tmpfolder + '/stream.tmp'
    
    # convert SDDS file to CSV file
    subprocess.call(CONFIG.elegant_exec + 'sdds2stream ' + filename + ' -columns=x,xp,y,yp,t,p,dt > ' + tmpfile, shell=True)
    
    # extract charge
    Q = float(subprocess.check_output([CONFIG.elegant_exec + 'sdds2stream ' + filename + ' -parameter=Charge'], shell=True))
    
    # load phasespace from CSV file
    phasespace = np.loadtxt(open(tmpfile, "rb"), delimiter=' ')
    
    # delete CSV file and temporary folder
    shutil.rmtree(tmpfolder)
    
    # make beam (note: z is flipped)
    beam = Beam()
    beam.set_phase_space(xs=phasespace[:,0], 
                       ys=phasespace[:,2], 
                       zs=-1*phasespace[:,6]*SI.c, 
                       xps=phasespace[:,1], 
                       yps=phasespace[:,3], 
                       Es=gamma2energy(phasespace[:,5]),
                       Q=Q)
    beam.location = np.mean(phasespace[:,4])*SI.c
    
    return beam


def elegant_write_beam(beam, filename):
    
    # create temporary CSV file and folder
    tmpfolder = CONFIG.temp_path + str(uuid.uuid4()) + '/'
    os.mkdir(tmpfolder)
    tmpfile = tmpfolder + 'beam.csv'
    
    # write beam phasespace to CSV (note: z/t is flipped)
    M = np.matrix([beam.xs(),beam.xps(),beam.ys(),beam.yps(),-1*beam.ts(),beam.gammas(),beam.ts()])
    with open(tmpfile, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for i in range(int(len(beam))):
            csvwriter.writerow([M[0,i], M[1,i], M[2,i], M[3,i], M[4,i], M[5,i], M[6,i], int(i+1)])
    
    # convert CSV to SDDS (ascii for now)
    subprocess.call(CONFIG.elegant_exec + 'csv2sdds ' + tmpfile + ' ' + filename + ' -asciiOutput -columnData=name=x,type=double,units=m' +
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
    subprocess.call(CONFIG.elegant_exec + 'sddsconvert -binary ' + filename + ' -noWarnings', shell=True)
    
    # delete CSV file and temporary folder
    shutil.rmtree(tmpfolder)
    
    return filename
    
    
def elegant_run(filename, beam0, beamfile, envars={}, quiet=False, run_from_container=False):
    
    # convert incoping beam object to temporary SDDS file
    tmpfolder = CONFIG.temp_path + str(uuid.uuid4())
    os.mkdir(tmpfolder)
    tmpfile = tmpfolder + '/beam.bun'
    elegant_write_beam(beam0, tmpfile)
    envars['BEAM'] = tmpfile
    
    # set environment variables
    for key in envars:
        os.environ[key] = str(envars[key])

    # run system command
    cmd = CONFIG.elegant_exec + 'elegant ' + filename + CONFIG.elegant_rpnflag
        
    if quiet:
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
    else:
        subprocess.call(cmd, shell=True, stdout=None)
        
    # convert SDDS output to beam object
    beam = elegant_read_beam(beamfile)
        
    # copy previous beam metadata
    beam.location = beam0.location
    beam.trackable_number = beam0.trackable_number
    beam.stage_number = beam0.stage_number
    
    # reset previous macroparticle charge
    beam.copy_particle_charge(beam0)
    
    # delete temporary bunch files and temporary folder
    shutil.rmtree(tmpfolder)
    
    return beam


def elegant_apl_fieldmap2D(tau_lens, filename, lensdim_x=5e-3, lensdim_y=1e-3):
    
    # transverse dimensions
    xs = np.linspace(-lensdim_x, lensdim_x, 1001)
    ys = np.linspace(-lensdim_y, lensdim_y, 201)
    
    # create temporary CSV file and folder
    tmpfolder = CONFIG.temp_path + str(uuid.uuid4())
    os.mkdir(tmpfolder)
    tmpfile = tmpfolder + '/map.csv'
    
    # create map
    Bmap = np.zeros((len(xs)*len(ys), 4));
    for i in range(len(xs)):
        x = xs[i]
        for j in range(len(ys)):
            y = ys[j]
            
            Bx = (y + x*y*tau_lens)
            By = -(x + ((x**2 + y**2)/2)*tau_lens)
            
            index = i + j*len(xs)
            Bmap[index,:] = [x, y, Bx, By]
    
    # filename
    np.savetxt(tmpfile, Bmap, delimiter=',')
    
    # convert SDDS to binary
    subprocess.call(CONFIG.elegant_exec + 'csv2sdds ' + tmpfile + ' ' + filename + ' -columnData=name=x,type=double,unit=m' +
                                                                        ' -columnData=name=y,type=double,unit=m' +
                                                                        ' -columnData=name=Bx,type=double,unit=T' +
                                                                        ' -columnData=name=By,type=double,unit=T', shell=True)
    
    # delete temporary CSV file and folder
    shutil.rmtree(tmpfolder)
    
    return filename
