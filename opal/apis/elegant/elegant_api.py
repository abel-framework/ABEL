import tempfile, os, subprocess, csv
from os.path import exists
import numpy as np
from opal import CONFIG, Beam
from opal.utilities import SI
from opal.utilities.relativity import gamma2energy, energy2gamma

def elegant_read_beam(filename):

    # create temporary file
    tmpfile = tempfile.gettempdir() + '/stream.tmp'

    # convert SDDS file to CSV file
    subprocess.call(CONFIG.elegant_path + 'sdds2stream ' + filename + ' -columns=x,xp,y,yp,t,p,dt > ' + tmpfile, shell=True)
    
    # extract charge
    Q = float(subprocess.check_output([CONFIG.elegant_path + 'sdds2stream ' + filename + ' -parameter=Charge'], shell=True))
    
    # load phasespace from CSV file
    phasespace = np.loadtxt(open(tmpfile, "rb"), delimiter=' ')
    
    # delete CSV file
    os.remove(tmpfile)
    
    # make beam
    beam = Beam()
    beam.setPhaseSpace(xs=phasespace[:,0], 
                       ys=phasespace[:,2], 
                       zs=phasespace[:,6]*SI.c, 
                       xps=phasespace[:,1], 
                       yps=phasespace[:,3], 
                       Es=gamma2energy(phasespace[:,5]),
                       Q=Q)
    beam.location = np.mean(phasespace[:,4])*SI.c
    
    return beam


def elegant_write_beam(beam, filename=None):
    
    # create temporary CSV file
    tmpfile = tempfile.gettempdir() + '/beam.csv'
    
    # write beam phasespace to CSV
    M = np.matrix([beam.xs(),beam.xps(),beam.ys(),beam.yps(),beam.ts(),beam.gammas(),beam.ts()])
    with open(tmpfile, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for i in range(int(beam.Npart())):
            csvwriter.writerow([M[0,i], M[1,i], M[2,i], M[3,i], M[4,i], M[5,i], M[6,i], int(i+1)])
    
    # create temporary SDDS file
    if filename is None:
        filename = tempfile.gettempdir() + '/beam.bun'
    
    # convert CSV to SDDS (ascii for now)
    subprocess.call(CONFIG.elegant_path + 'csv2sdds ' + tmpfile + ' ' + filename + ' -asciiOutput -columnData=name=x,type=double,units=m' +
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
    lines.insert(line2, str(int(beam.Npart()))+'\n')
    lines.insert(line2, str(int(beam.Npart()))+'\n')
    lines.insert(line2, ' '+str(beam.charge())+'\n')
    lines.insert(line2, ' '+str(energy2gamma(beam.energy()))+'\n')
    lines.insert(line2, str(1)+'\n')
                 
    with open(filename, 'w') as f:
        f.write("".join(lines))
        
    # convert SDDS to binary
    subprocess.call(CONFIG.elegant_path + 'sddsconvert -binary ' + filename + ' -noWarnings', shell=True)
    
    # delete CSV file
    os.remove(tmpfile)
    
    return filename
    
    
def elegant_run(filename, beam0, beamfile, envars={}, quiet=False):

    # convert beam object to SDDS
    beamfile0 = elegant_write_beam(beam0)
    envars['BEAM'] = beamfile0
    
    # set environment variables
    for key in envars:
        os.environ[key] = str(envars[key])

    # run system command
    cmd = CONFIG.elegant_path + 'elegant ' + filename + ' -rpnDefns=' + CONFIG.elegant_path + 'defns.rpn'
    if quiet:
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
    else:
        subprocess.call(cmd, shell=True, stdout=None)
        
    # convert SDDS output to beam object
    beam = elegant_read_beam(beamfile)
        
    # add previous beam metadata
    beam.location = beam0.location
    beam.trackableNumber = beam0.trackableNumber
    beam.stageNumber = beam0.stageNumber
    
    # reset previous macroparticle charge
    beam.copyParticleCharge(beam0)
    
    return beam


def elegant_apl_fieldmap2D(tau_lens, filename = None):
    
    # transverse dimensions
    xs = np.linspace(-6e-3, 6e-3, 1201);
    ys = np.linspace(-2e-3, 2e-3, 401);
    
    # create temporary CSV file
    tmpfile = tempfile.gettempdir() + '/map.csv'
    
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
    
    # create temporary SDDS file
    if filename is None:
        filename = tempfile.gettempdir() + '/map.sdds'
    
    # convert SDDS to binary
    subprocess.call(CONFIG.elegant_path + 'csv2sdds ' + tmpfile + ' ' + filename + ' -columnData=name=x,type=double,unit=m' +
                                                                        ' -columnData=name=y,type=double,unit=m' +
                                                                        ' -columnData=name=Bx,type=double,unit=T' +
                                                                        ' -columnData=name=By,type=double,unit=T', shell=True)
    
    # delete CSV file
    os.remove(tmpfile)
    
    return filename
