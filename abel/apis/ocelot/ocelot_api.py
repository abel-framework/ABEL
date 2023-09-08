from ocelot import ParticleArray
from abel import Beam
import scipy.constants as SI
import numpy as np


# convert from OCELOT particle array to ABEL beam
def ocelot_particle_array2beam(particle_array):
    
    # initialize beam object
    beam = Beam()
    
    # reference momentum
    E0 = particle_array.E * 1e9 # [eV]
    p0 = np.sqrt((E0*SI.e)**2/SI.c**2 - SI.m_e**2*SI.c**2) # [kg m s^-1]
    
    # set phase space
    beam.set_phase_space(Q=-sum(particle_array.q_array), xs=particle_array.x(), ys=particle_array.y(),
                            zs=-particle_array.tau(), 
                            pxs=particle_array.px() * p0, pys=particle_array.py() * p0,
                            Es=(1 + particle_array.p())*E0)
    
    # add location
    beam.location = particle_array.s
    
    return beam
      
    
# convert from ABEL beam to OCELOT particle array
def beam2ocelot_particle_array(beam):
    
    # initalize particle array object
    p_array = ParticleArray(len(beam))
    
    # reference momentum
    E0 = beam.energy() # [eV]
    p0 = np.sqrt((E0*SI.e)**2/SI.c**2 - SI.m_e**2*SI.c**2) # [kg m s^-1]
    
    # set phase space array
    p_array.rparticles[0] = beam.xs()
    p_array.rparticles[1] = beam.pxs() / p0
    p_array.rparticles[2] = beam.ys()
    p_array.rparticles[3] = beam.pys() / p0
    p_array.rparticles[4] = -beam.zs()
    p_array.rparticles[5] = beam.deltas()
    
    # set mean beam energy
    p_array.E = beam.energy() / 1e9
    
    # set particle location
    p_array.s = beam.location
    
    # set particle weights
    p_array.q_array = beam.weightings()*SI.e
    
    return p_array
