import tempfile, shlex, subprocess, csv
from os.path import exists
from os import mkdir, remove
import numpy as np
from opal import CONFIG, Event
import time

def guineapig_run(inputfile, beam1, beam2):
    
    # make beam files
    beamfile1 = guineapig_write_beam(beam1, 'inputbeam1')
    beamfile2 = guineapig_write_beam(beam2, 'inputbeam2')

    # make temp folder
    tmpfolder = tempfile.gettempdir()
    if not exists(tmpfolder):
        mkdir(tmpfolder)
    
    # temporary output file
    outputfile = tmpfolder + "/output.ref"
    
    # run GUINEA-PIG
    cmd = CONFIG.guineapig_path + 'guinea default default ' + outputfile + ' --el_file=' + beamfile1 + ' --pos_file=' + beamfile2 + ' --acc_file=' + inputfile
    subprocess.run(cmd, shell=True, check=True, capture_output=True)
    
    # parse outputs
    lumi_ee_full, lumi_ee_peak, lumi_ee_geom = guineapig_read_output(outputfile)
    
    # remove files
    remove(outputfile)
    remove(beamfile1)
    remove(beamfile2)
    
    # make event object
    event = Event(beam1, beam2)
    event.luminosity_geom = lumi_ee_geom
    event.luminosity_full = lumi_ee_full
    event.luminosity_peak = lumi_ee_peak
    
    #return event
    return event
    

def guineapig_write_beam(beam, name):
    
    # create temporary CSV file
    tmpfile = tempfile.gettempdir() + '/' + name + '.ini'
    
     # write beam phasespace to CSV
    M = np.matrix([beam.Es()/1e9, beam.xps(), beam.yps(), beam.xs()*1e9, beam.ys()*1e9, (beam.offsetZ()-beam.zs())*1e6])
    with open(tmpfile, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for i in range(int(beam.Npart())):
            csvwriter.writerow([M[0,i], M[1,i], M[2,i], M[3,i], M[4,i], M[5,i]])
    
    return tmpfile


def guineapig_read_output(outputfile):
    
    # string to search in file
    with open(outputfile, 'r') as fp:
        lines = fp.readlines()
        for row in lines:
            header = ' ------------- general results ------------ '
            if row.find(header) != -1:
                ind_header = lines.index(row)
                
                # calculate
                lumi_ee_geom = float((lines[ind_header+2]).split(" ")[2]) # [m^-2 per crossing]
                lumi_ee_full = float((lines[ind_header+3]).split(" ")[2]) # [m^-2 per crossing]
                lumi_ee_peak = float((lines[ind_header+4]).split(" ")[2]) # [m^-2 per crossing]
                
                return lumi_ee_full, lumi_ee_peak, lumi_ee_geom