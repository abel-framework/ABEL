from abel.classes.plasma_lens.plasma_lens import PlasmaLens
import numpy as np
import scipy.constants as SI
import matplotlib.pyplot as plt

class PlasmaLensNonlinearThin(PlasmaLens):

    def __init__(self, length=None, radius=None, current=None, rel_nonlinearity=0, nonlinearity_in_x=True):

        super().__init__(length, radius, current)

        # set nonlinearity (defined as R/Dx)
        self.rel_nonlinearity = rel_nonlinearity
        self.nonlinearity_in_x = nonlinearity_in_x

    def remove_charge_outside(self, beam, plot=False):

        mask = np.sqrt((beam.xs()-self.offset_x)**2 + (beam.ys()-self.offset_y)**2) > self.radius

        if plot:

            fig, ax = plt.subplots()
            ax.add_patch(patches.Circle((self.offset_x, self.offset_y), self.radius, fill=False, linestyle='--'))

            ax.scatter(beam.xs(), beam.ys(), s=1, alpha=0.1, zorder=0)
            ax.set_aspect('equal', 'box')
            plt.show()

            rel_loss = mask.sum() / mask.size
            print(f"Removing {rel_loss*100:.2f}% of the charge outside the lens")

        # remove charge outside the lens
        del beam[mask]

        return beam

    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # remove charge outside the lens (start)
        # del beam[np.sqrt((beam.xs()-self.offset_x)**2 + (beam.ys()-self.offset_y)**2) > self.radius]
        self.remove_charge_outside(beam, plot=False)

        # drift half the distance
        beam.transport(self.length/2)

        # remove charge outside the lens (middle)
        # del beam[np.sqrt((beam.xs()-self.offset_x)**2 + (beam.ys()-self.offset_y)**2) > self.radius]
        self.remove_charge_outside(beam)

        # get particles
        xs = beam.xs()-self.offset_x
        xps = beam.xps()
        ys = beam.ys()-self.offset_y
        yps = beam.yps()

        # nominal focusing gradient
        g0 = self.get_focusing_gradient()

        # calculate the nonlinearity
        if self.nonlinearity_in_x:
            inv_Dx = self.rel_nonlinearity/self.radius
            inv_Dy = 0.0
        else:
            inv_Dx = 0.0
            inv_Dy = self.rel_nonlinearity/self.radius

        # thin lens kick
        Bx = g0*(ys + xs*ys*inv_Dx  + (xs**2 + ys**2)/2*inv_Dy)
        By = -g0*(xs + (xs**2 + ys**2)/2*inv_Dx + xs*ys*inv_Dy)

        # calculate the angular kicks
        delta_xp = self.length*(By*beam.charge_sign()*SI.c/beam.Es())
        delta_yp = -self.length*(Bx*beam.charge_sign()*SI.c/beam.Es())

        # set new beam positions and angles (shift back plasma-lens offsets)
        beam.set_xps(xps + delta_xp)
        beam.set_yps(yps + delta_yp)

        # drift another half the distance
        beam.transport(self.length/2)

        # remove charge outside the lens (end)
        # del beam[np.sqrt((beam.xs()-self.offset_x)**2 + (beam.ys()-self.offset_y)**2) > self.radius]
        self.remove_charge_outside(beam)

        return super().track(beam, savedepth, runnable, verbose)


    def get_focusing_gradient(self):
        return SI.mu_0 * self.current / (2*np.pi * self.radius**2)
