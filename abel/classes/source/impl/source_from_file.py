from abel import Source, Beam

class SourceFromFile(Source):
    
    def __init__(self, length=0, charge=None, energy=None, accel_gradient=None, wallplug_efficiency=1, file=None, x_offset=0, y_offset=0, x_angle=0, y_angle=0, waist_shift_x=0, waist_shift_y=0):
        
        self.file = file

        super().__init__(length, charge, energy, accel_gradient, wallplug_efficiency, x_offset, y_offset, x_angle, y_angle, waist_shift_x, waist_shift_y)
        
    
    def track(self, _=None, savedepth=0, runnable=None, verbose=False):
        
        # make empty beam
        beam = Beam.load(self.file)

        # scale the charge (if set)
        if self.charge is not None:
            beam.scale_charge(self.charge)
        else:
            self.charge = beam.charge()

        # scale the energy (if set)
        if self.energy is not None:
            beam.scale_energy(self.energy)
        else:
            self.energy = beam.energy()

        # add jitters and offsets in super function
        return super().track(beam, savedepth, runnable, verbose)

    
    def get_charge(self):
        if self.charge is None:
            beam = Beam.load(self.file)
            self.charge = beam.charge()
        return self.charge

    
    def get_energy(self):
        if self.energy is None:
            beam = Beam.load(self.file)
            self.energy = beam.energy()
        return self.energy
    