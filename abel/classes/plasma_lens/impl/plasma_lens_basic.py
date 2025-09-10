from abel.classes.plasma_lens.plasma_lens import PlasmaLens
import numpy as np
import scipy.constants as SI

class PlasmaLensBasic(PlasmaLens):
    
    def __init__(self, length=None, radius=None, current=None):
        super().__init__(length, radius, current)
        
    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        from abel.physics_models.hills_equation import evolve_hills_equation_analytic
        from abel.utilities.plasma_physics import mag_field_grad2wave_number

        kp = mag_field_grad2wave_number(g=-self.get_focusing_gradient())
        
        # calculate the evolution
        xs, uxs = evolve_hills_equation_analytic(beam.xs()-self.offset_x, beam.uxs(), self.length, beam.gammas(), 0.0, kp)
        ys, uys = evolve_hills_equation_analytic(beam.ys()-self.offset_y, beam.uys(), self.length, beam.gammas(), 0.0, kp)
        
        # set new beam positions and angles (shift back plasma-lens offsets)
        beam.set_xs(xs+self.offset_x)
        beam.set_uxs(uxs)
        beam.set_ys(ys+self.offset_y)
        beam.set_uys(uys)
        
        return super().track(beam, savedepth, runnable, verbose)   


    def get_focusing_gradient(self):
        return SI.mu_0 * self.current / (2*np.pi * self.radius**2)
    