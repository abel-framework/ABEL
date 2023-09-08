from abel import Beam
import scipy.constants as SI
import numpy as np
import wake_t

# convert from WakeT particle bunch to ABEL beam
def wake_t_bunch2beam(bunch):
    
    # extract phase space (with charge)
    phasespace = bunch.get_6D_matrix_with_charge()
    
    # initialize beam
    beam = Beam()
    
    # set the phase space of the ABEL beam
    beam.set_phase_space(Q=sum(phasespace[6]),
                         xs=phasespace[0],
                         ys=phasespace[2],
                         zs=phasespace[4], 
                         pxs=phasespace[1]*SI.c*SI.m_e,
                         pys=phasespace[3]*SI.c*SI.m_e,
                         pzs=phasespace[5]*SI.c*SI.m_e)
    
    return beam
      
    
# convert from ABEL beam to WakeT particle bunch
def beam2wake_t_bunch(beam, name='beam'):
    
    # convert the beam
    bunch = wake_t.ParticleBunch(beam.weightings(),
                                 beam.xs(),
                                 beam.ys(),
                                 beam.zs(), 
                                 beam.pxs()/(SI.c*SI.m_e),
                                 beam.pys()/(SI.c*SI.m_e),
                                 beam.pzs()/(SI.c*SI.m_e),
                                 name=name)
    
    return bunch
