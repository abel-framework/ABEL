from opal import Stage

class StageBasic(Stage):
    
    def __init__(self, nom_energy_gain=None, length=None, chirp=0, z_offset=0):
        self.nom_energy_gain = nom_energy_gain
        self.length = length
        self.chirp = chirp
        self.z_offset = z_offset
       
    def get_length(self):
        return self.length
    
    def get_nom_energy_gain(self):
        return self.nom_energy_gain
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # adiabatic damping
        beam.betatronDamping(self.deltaE)

        # flip transverse phase spaces
        beam.flipTransversePhaseSpaces()

        # accelerate beam
        beam.accelerate(energy_gain=self.nom_energy_gain, chirp=self.chirp, z_offset=self.z_offset)

        return super().track(beam, savedepth, runnable, verbose)
        
    