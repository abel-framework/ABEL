import uuid, os, subprocess, csv, shutil
import numpy as np
from abel.CONFIG import CONFIG
from abel.classes.beam import Beam
import scipy.constants as SI

def elegant_read_beam(filename, tmpfolder=None):
    """
    Convert an ELEGANT beam file (SDDS format) into an ABEL ``Beam`` object.

    This function uses the ELEGANT ``sdds2stream`` utility to extract particle 
    phase space data and beam charge from an SDDS file. The data is then mapped 
    into an ABEL ``Beam`` object, with proper coordinate transformations 
    (including a flipped z-coordinate) and energy conversion from Lorentz factor 
    (gamma).

    Parameters
    ----------
    filename : str
        Path to the input ELEGANT SDDS beam file.

    tmpfolder : str, optional
        Temporary folder used to store intermediate files. If not provided, a 
        new temporary folder will be created and deleted automatically.

    Returns
    -------
    Beam or None
        An ABEL ``Beam`` object containing the phase space and beam charge.
        Returns ``None`` if the charge parameter cannot be extracted.
    """

    from abel.utilities.relativity import gamma2energy
    
    make_new_tmpfolder = tmpfolder is None
    if make_new_tmpfolder:
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4())
        os.mkdir(tmpfolder)
    tmpfile =  tmpfolder + '/stream_' + str(uuid.uuid4()) + '.tmp'
    
    # convert SDDS file to CSV file
    subprocess.run(CONFIG.elegant_exec + 'sdds2stream ' + filename + ' -columns=x,xp,y,yp,t,p,dt > ' + tmpfile, shell=True)
    
    # extract charge
    try:
        Q = float(subprocess.check_output([CONFIG.elegant_exec + 'sdds2stream ' + filename + ' -parameter=Charge'], shell=True))  ######## Charge sign not set correctly?
    except:
        return None
    
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
    
    return beam


def elegant_write_beam(beam, filename, tmpfolder=None):
    """
    Convert an ABEL ``Beam`` object into an ELEGANT SDDS beam file.

    This function takes the phase space data stored in an ABEL ``Beam`` and 
    exports it into ELEGANT-compatible SDDS format. The data is first written to 
    a temporary CSV file, converted to SDDS (ASCII), metadata headers are 
    inserted, and finally converted into a binary SDDS file. Temporary files and 
    directories are removed unless explicitly provided.

    Parameters
    ----------
    beam : Beam
        An ABEL ``Beam`` object containing particle phase space data.

    filename : str
        Path to the output ELEGANT SDDS file.

    tmpfolder : str, optional
        Path to a temporary folder for intermediate files. If not provided, a 
        new temporary folder will be created and deleted automatically.

    Returns
    -------
    str
        The path to the generated ELEGANT SDDS beam file.
    """

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
    subprocess.run(CONFIG.elegant_exec + 'sddsconvert -binary ' + filename + ' -noWarnings', shell=True)
    
    # delete CSV file and temporary folder
    os.remove(tmpfile)
    if make_new_tmpfolder:
        shutil.rmtree(tmpfolder)
    
    return filename
    
    
def elegant_run(filename, beam0, inputbeamfile, outputbeamfile, envars={}, quiet=False, run_from_container=False, tmpfolder=None):
    """
    Run an ELEGANT simulation with an ABEL ``Beam`` object as input and return 
    the output beam.

    This function converts an ABEL ``Beam`` into an ELEGANT-compatible SDDS 
    input file, executes an ELEGANT simulation, and then converts the resulting
    SDDS output back into an ABEL ``Beam``. Metadata from the input beam is
    preserved and propagated to the output beam.

    Parameters
    ----------
    filename : str
        Path to the ELEGANT input file (``runfile.ele``).

    beam0 : ABEL ``Beam``
        Input beam.

    inputbeamfile : str
        Path to the temporary SDDS file that will store the input beam for 
        ELEGANT.

    outputbeamfile : str
        Path to the SDDS file generated by ELEGANT containing the output beam.

    envars : dict, optional
        Environment variables to set for the ELEGANT execution.

    quiet : bool, default=``False``
        If ``True``, suppress ELEGANT console output.

    run_from_container : bool, default=``False``
        If ``True``, run ELEGANT from a containerized environment (currently 
        unused).

    tmpfolder : str, optional
        Temporary folder for intermediate files. If not provided, a new folder
        is created and cleaned up automatically.

    Returns
    -------
    beam : ABEL ``Beam`` object
        Contains the output particle distribution after the ELEGANT run. 
        Metadata such as location, trackable number, stage number, and 
        per-particle charge are transferred from the input beam.
    """

    # convert incoming beam object to temporary SDDS file
    elegant_write_beam(beam0, inputbeamfile, tmpfolder=tmpfolder)

    # run system command
    cmd = CONFIG.elegant_exec + 'elegant ' + filename + CONFIG.elegant_rpnflag
        
    if quiet:
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
    else:
        subprocess.call(cmd, shell=True, stdout=None)
        
    # convert SDDS output to beam object
    beam = elegant_read_beam(outputbeamfile, tmpfolder=tmpfolder)
        
    # copy previous beam metadata
    beam.location = beam0.location
    beam.trackable_number = beam0.trackable_number
    beam.stage_number = beam0.stage_number
    
    # reset previous macroparticle charge
    beam.copy_particle_charge(beam0)
    
    return beam


def elegant_apl_fieldmap2D(tau_lens, filename, lensdim_x=5e-3, lensdim_y=1e-3, lens_x_offset=0.0, lens_y_offset=0.0, tmpfolder=None):
    """
    Generates a 2D magnetic field map for an APL (Active Plasma Lens) and export 
    it in ELEGANT-compatible SDDS format.

    This function computes a 2D transverse magnetic field map (Bx, By) across a 
    rectangular region defined by the lens dimensions. The map is written to a 
    temporary CSV file, converted into an SDDS dataset, and returned as a binary 
    SDDS file for use in ELEGANT.

    Parameters
    ----------
    tau_lens : float
        Plasma lens parameter that determines the strength and nonlinearities of 
        the transverse field.

    filename : str
        Path to the output SDDS file for the field map.

    lensdim_x : [m] float, default=5e-3
        Half-width of the transverse field map in the x-direction.

    lensdim_y : [m] float, default=1e-3
        Half-width of the transverse field map in the y-direction.
        
    lens_x_offset : [m] float, default=0.0
        Lens transverse offset in x.

    lens_y_offset : [m] float, default=0.0
        Lens transverse offset in y.

    tmpfolder : str, optional
        Temporary folder for intermediate CSV storage. If not provided, a new 
        folder will be created automatically.

    Returns
    -------
    filename : str
        The path to the generated ELEGANT SDDS field map file.
    """
    
    # transverse dimensions
    #xs = np.linspace(-lensdim_x, lensdim_x, 1001)
    xs = np.linspace(-lensdim_x, lensdim_x, 2001)
    ys = np.linspace(-lensdim_y, lensdim_y, 201)
    
    make_new_tmpfolder = tmpfolder is None
    if make_new_tmpfolder:
        tmpfolder = CONFIG.temp_path + str(uuid.uuid4())
        os.mkdir(tmpfolder)
    #tmpfile = tmpfolder + '/map_' + str(uuid.uuid4()) + '.csv'
    tmpfile = tmpfolder + '/Bmap.csv'
    
    # create map
    Bmap = np.zeros((len(xs)*len(ys), 4));
    
    for i in range(len(xs)):
        x = xs[i]   
        for j in range(len(ys)):
            y = ys[j]
            
            Bx = ((y+lens_y_offset) + (x+lens_x_offset) * (y+lens_y_offset) * tau_lens)
            By = -((x+lens_x_offset) + (( (x+lens_x_offset)**2 + (y+lens_y_offset)**2 )/2)*tau_lens)
            
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
    #os.remove(tmpfile)
    #if make_new_tmpfolder:
    #    shutil.rmtree(tmpfolder)

    return filename


