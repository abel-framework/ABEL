from abel import Stage
from abel.utilities.plasma_physics import beta_matched

class StageBasic(Stage):
    
    def __init__(self, nom_energy_gain=None, length=None, chirp=0, z_offset=0):
        self.nom_energy_gain = nom_energy_gain
        self.length = length
        self.chirp = chirp
        self.z_offset = z_offset

    def energy_usage(self):
        pass

    # matched beta function of the stage (for a given energy)
    def matched_beta_function(self, energy):
        return beta_matched(self.plasma_density, energy)
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        
        # adiabatic damping
        beam.apply_betatron_damping(self.nom_energy_gain)

        # flip transverse phase spaces
        beam.flip_transverse_phase_spaces()

        # accelerate beam
        beam.accelerate(energy_gain=self.nom_energy_gain, chirp=self.chirp, z_offset=self.z_offset)

        return super().track(beam, savedepth, runnable, verbose)
        
    