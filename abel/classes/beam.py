import numpy as np
import openpmd_api as io
from datetime import datetime
from pytz import timezone
from abel.CONFIG import CONFIG
from types import SimpleNamespace
import scipy.constants as SI
from abel.utilities.relativity import energy2proper_velocity, proper_velocity2energy, momentum2proper_velocity, proper_velocity2momentum, proper_velocity2gamma, energy2gamma, gamma2proper_velocity
from abel.utilities.statistics import weighted_mean, weighted_std, weighted_cov
from abel.utilities.plasma_physics import k_p, wave_breaking_field, beta_matched
from abel.physics_models.hills_equation import evolve_hills_equation_analytic
from abel.physics_models.betatron_motion import evolve_betatron_motion

import scipy.sparse as sp
from scipy.spatial.transform import Rotation as Rot
import copy

from matplotlib import pyplot as plt

class Beam():
    
    def __init__(self, phasespace=None, num_particles=1000, num_bunches_in_train=1, bunch_separation=0.0):

        # the phase space variable is private
        if phasespace is not None:
            self.__phasespace = phasespace
        else:
            self.__phasespace = self.reset_phase_space(num_particles)

        # bunch pattern information
        self.num_bunches_in_train = num_bunches_in_train
        self.bunch_separation = bunch_separation # [s]
        
        self.trackable_number = -1 # will increase to 0 after first tracking element
        self.stage_number = 0
        self.location = 0        
    
    
    # reset phase space
    def reset_phase_space(self, num_particles):
        self.__phasespace = np.zeros((8, num_particles))
    
    # filter out macroparticles based on a mask (true means delete)
    def __delitem__(self, indices):
        if hasattr(indices, 'len'):
            if len(indices) == len(self):
                indices = np.where(indices)
        self.__phasespace = np.ascontiguousarray(np.delete(self.__phasespace, indices, 1))
    
    # filter out nans
    def remove_nans(self):
        del self[np.isnan(self).any(axis=1)]
        
    # set phase space
    def set_phase_space(self, Q, xs, ys, zs, uxs=None, uys=None, uzs=None, pxs=None, pys=None, pzs=None, xps=None, yps=None, Es=None, weightings=None, particle_mass=SI.m_e):
        
        # make empty phase space
        num_particles = len(xs)
        self.reset_phase_space(num_particles)
        
        # add positions
        self.set_xs(xs)
        self.set_ys(ys)
        self.set_zs(zs)
        
        # add momenta
        if uzs is None:
            if pzs is not None:
                uzs = momentum2proper_velocity(pzs)
            elif Es is not None:
                uzs = energy2proper_velocity(Es)
        self.__phasespace[5,:] = uzs
        
        if uxs is None:
            if pxs is not None:
                uxs = momentum2proper_velocity(pxs)
            elif xps is not None:
                uxs = xps * uzs
        self.__phasespace[3,:] = uxs
        
        if uys is None:
            if pys is not None:
                uys = momentum2proper_velocity(pys)
            elif yps is not None:
                uys = yps * uzs
        self.__phasespace[4,:] = uys
        
        # charge
        if weightings is None:
            self.__phasespace[6,:] = Q/num_particles
        else:
            self.__phasespace[6,:] = Q*weightings/np.sum(weightings)
        
        # ids
        self.__phasespace[7,:] = np.arange(num_particles)

        # single particle mass [kg]
        self.particle_mass = particle_mass
       
    
    # addition operator (add two beams using the + operator)
    def __add__(self, beam):
        return Beam(phasespace = np.append(self.__phasespace, beam.__phasespace, axis=1))
        
    # in-place addition operator (add one beam to another using the += operator)
    def __iadd__(self, beam):
        if beam is not None:    
            self.__phasespace = np.append(self.__phasespace, beam.__phasespace, axis=1)
        return self
    
    # indexing operator (get single particle out)
    def __getitem__(self, index):
        return self.__phasespace[:,index]
    
    # "length" operator (number of macroparticles)
    def __len__(self):
        return self.__phasespace.shape[1]
    
    # string operator (called when printing)
    def __str__(self):
        return f"Beam: {len(self)} macroparticles, {self.charge()*1e9:.2f} nC, {self.energy()/1e9:.2f} GeV"
        
        
    ## BUNCH PATTERN

    def bunch_frequency(self) -> float:
        if self.num_bunches_in_train == 1:
            return None
        elif self.bunch_separation == 0.0:
            return None
        else:
            return 1/self.bunch_separation

    def train_duration(self) -> float:
        if self.num_bunches_in_train == 1:
            return 0.0
        elif self.bunch_separation == 0.0:
            return None
        else:
            return self.bunch_separation * (self.num_bunches_in_train-1)
    
    def average_current_train(self) -> float:
        return self.charge()*self.bunch_frequency()

    
    ## BEAM ARRAYS

    # get phase space variables
    def xs(self):
        return self.__phasespace[0,:]
    def ys(self):
        return self.__phasespace[1,:]
    def zs(self):
        return self.__phasespace[2,:]
    def uxs(self):
        return self.__phasespace[3,:]
    def uys(self):
        return self.__phasespace[4,:]
    def uzs(self):
        return self.__phasespace[5,:]
    def qs(self):
        return self.__phasespace[6,:]
    def ids(self):
        return self.__phasespace[7,:]
    
    # set phase space variables
    def set_xs(self, xs):
        self.__phasespace[0,:] = xs
    def set_ys(self, ys):
        self.__phasespace[1,:] = ys
    def set_zs(self, zs):
        self.__phasespace[2,:] = zs
    def set_uxs(self, uxs):
        self.__phasespace[3,:] = uxs
    def set_uys(self, uys):
        self.__phasespace[4,:] = uys
    def set_uzs(self, uzs):
        self.__phasespace[5,:] = uzs
        
    def set_xps(self, xps):
        self.set_uxs(xps*self.uzs())
    def set_yps(self, yps):
        self.set_uys(yps*self.uzs())
    def set_Es(self, Es):
        self.set_uzs(energy2proper_velocity(Es))
        
    def set_qs(self, qs):
        self.__phasespace[6,:] = qs
    def set_ids(self, ids):
        self.__phasespace[7,:] = ids
        
    def weightings(self):
        return self.__phasespace[6,:]/(self.charge_sign()*SI.e)
    
    # copy another beam's macroparticle charge
    def copy_particle_charge(self, beam):
        self.set_qs(np.median(beam.qs()))

    def scale_charge(self, Q):
        self.set_qs((Q/self.charge())*self.qs())

    def scale_energy(self, E):
        self.set_Es((E/self.energy())*self.Es())
    
    
    def rs(self):
        return np.sqrt(self.xs()**2 + self.ys()**2)
    
    def pxs(self):
        return proper_velocity2momentum(self.uxs())
    def pys(self):
        return proper_velocity2momentum(self.uys())
    def pzs(self):
        return proper_velocity2momentum(self.uzs())
    
    def xps(self):
        return self.uxs()/self.uzs()
    def yps(self):
        return self.uys()/self.uzs()

    def gammas(self):
        return proper_velocity2gamma(self.uzs())
    def Es(self):
        return proper_velocity2energy(self.uzs())
    def deltas(self, pz0=None):
        if pz0 is None:
            pz0 = np.mean(self.pzs())
        return self.pzs()/pz0 -1
        
    
    def ts(self):
        return self.zs()/SI.c
    
    # vector of transverse positions and angles: (x, x', y, y')
    def transverse_vector(self):
        vector = np.zeros((4,len(self)))
        vector[0,:] = self.xs()
        vector[1,:] = self.xps()
        vector[2,:] = self.ys()
        vector[3,:] = self.yps()
        return vector
    
    # set phase space based on transverse vector: (x, x', y, y')
    def set_transverse_vector(self, vector):
        self.set_xs(vector[0,:])
        self.set_xps(vector[1,:])
        self.set_ys(vector[2,:])
        self.set_yps(vector[3,:]) 

    def norm_transverse_vector(self):
        vector = np.zeros((4,len(self)))
        vector[0,:] = self.xs()
        vector[1,:] = self.uxs()/SI.c
        vector[2,:] = self.ys()
        vector[3,:] = self.uys()/SI.c
        return vector


    ## Rotate the coordinate system of the beam
    # ==================================================
    def rotate_coord_sys_3D(self, axis1, angle1, axis2=np.array([0, 1, 0]), angle2=0.0, axis3=np.array([1, 0, 0]), angle3=0.0, invert=False):
        """
        Rotates the coordinate system (passive transformation) of the beam first with angle1 around axis1, then with angle2 around axis2 and lastly with angle3 around axis3.
        
        Parameters
        ----------
        axis1, axis2, axis3: 1x3 float nd arrays
            Unit vectors specifying the rotation axes.
        
        angle1, angle2, angle3: [rad] float
            Angles used for rotation of the beam's coordinate system around the respective axes.

        invert: bool
            Performs a standard passive transformation when False. If True, will perform an active transformation and can thus be used to invert the passive transformation.
            
        Returns
        ----------
        Modified beam xs, ys, zs, uxs, uys and uzs.
        """

        # Check the inputs
        if np.linalg.norm(axis1) != 1.0 or np.linalg.norm(axis2) != 1.0 or np.linalg.norm(axis3) != 1.0:
            raise ValueError('The rotation axes have to be unit vectors.')

        if angle1 < -np.pi or angle1 > np.pi or angle2 < -np.pi or angle2 > np.pi or angle3 < -np.pi or angle3 > np.pi:
            raise ValueError('The rotation angles have to be in the interval [-pi, pi].')
        
        # Combine into (N, 3) arrays
        zs = self.zs()
        xs = self.xs()
        ys = self.ys()
        uzs = self.uzs()
        uxs = self.uxs()
        uys = self.uys()
        coords = np.column_stack((zs, xs, ys))
        u_vecs = np.column_stack((uzs, uxs, uys))

        # Create rotation objects
        rotation1 = Rot.from_rotvec(angle1 * axis1)
        rotation2 = Rot.from_rotvec(angle2 * axis2)
        rotation3 = Rot.from_rotvec(angle3 * axis3)

        # Combine the rotations by applying them in sequence
        combined_rotation = rotation1 * rotation2 * rotation3  # Rotation order is right-to-left, but effectively opposite when inverse=True in combined_rotation.apply().

        # Apply rotation to the arrays
        rotated_coords = combined_rotation.apply(coords, inverse= not invert)  # Since combined_rotation.apply() performs active transformations by default, inverse must be set to True for passive transformation.
        rotated_u_vecs = combined_rotation.apply(u_vecs, inverse= not invert)

        # Extract the rotated arrays
        rotated_zs, rotated_xs, rotated_ys = rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2]
        rotated_uzs, rotated_uxs, rotated_uys = rotated_u_vecs[:, 0], rotated_u_vecs[:, 1], rotated_u_vecs[:, 2]

        self.set_zs(rotated_zs)
        self.set_xs(rotated_xs)
        self.set_ys(rotated_ys)
        self.set_uzs(rotated_uzs)
        self.set_uxs(rotated_uxs)
        self.set_uys(rotated_uys)
        

    # ==================================================
    def beam_alignment_angles(self):
        """
        Calculates the angles for rotation around the y- and x-axis to align the z-axis to the beam proper velocity.
        
        Parameters
        ----------
        ...
        
            
        Returns
        ----------
        x_angle: [rad] float
            Used for rotating the beam's frame around the y-axis.
        
        y_angle: [rad] float
            Used for rotating the beam's frame around the x-axis. Note that due to the right hand rule, a positive rotation angle in the zy-plane corresponds to rotation from z-axis towards negative y. I.e. the opposite sign convention of beam.yps().
        """

        # Get the mean proper velocity component offsets
        uz_offset = energy2proper_velocity(self.energy())
        ux_offset = self.ux_offset()
        uy_offset = self.uy_offset()

        point_vec = np.array([uz_offset, ux_offset, uy_offset])
        point_vec = point_vec/np.linalg.norm(point_vec)

        # Calculate the angles to be used for beam rotation
        z_axis = np.array([1, 0, 0])  # Axis as an unit vector
        
        zx_projection = point_vec * np.array([1, 1, 0])  # The projection of the pointing vector onto the zx-plane.

        # Separate treatments for small angles to avoid numerical instability
        if np.abs(self.x_angle()) < 1e-4:
            x_angle = ux_offset/uz_offset
        else:
            x_angle = np.sign(point_vec[1]) * np.arccos( np.dot(zx_projection, z_axis) / np.linalg.norm(zx_projection) )  # The angle between zx_projection and z_axis.

        
        if np.abs(self.y_angle()) < 1e-4:
            y_angle = point_vec[2]/np.linalg.norm(zx_projection)
        else:
            rotated_zy_projection = np.array([ np.linalg.norm(zx_projection), 0, point_vec[2] ])  # The new pointing vector after its zx-projection has been aligned to the rotated z-axis.
    
            y_angle = np.sign(point_vec[2]) * np.arccos( np.dot(rotated_zy_projection, z_axis) / np.linalg.norm(rotated_zy_projection) )  # Note that due to the right hand rule, a positive y_angle corresponds to rotation from z-axis towards negative y. I.e. opposite sign convention of yps.

        return x_angle, y_angle

    
    # ==================================================
    def xy_rotate_coord_sys(self, x_angle=None, y_angle=None, invert=False):
        """
        Rotates the coordinate system of the beam first with x_angle around the y-axis then with y_angle around the x-axis.
        
        Parameters
        ----------
        x_angle: [rad] float
            Angle to rotate the coordinate system with in the zx-plane.
        
        y_angle: [rad] float
            Angle to rotate the coordinate system with in the zy-plane. Note that due to the right hand rule, a positive rotation angle in the zy-plane corresponds to rotation from z-axis towards negative y. I.e. the opposite sign convention of beam.yps().

        invert: bool
            Performs a standard passive transformation when False. If True, will perform an active transformation and can thus be used to invert the passive transformation.
        
            
        Returns
        ----------
        Modified beam xs, ys, zs, uxs, uys and uzs.
        """
        
        x_axis = np.array([0, 1, 0])  # Axis as an unit vector. Axis permutaton is zxy.
        y_axis = np.array([0, 0, 1])

        if x_angle is None:
            x_angle, _ = self.beam_alignment_angles()
        if y_angle is None:
            _, y_angle = self.beam_alignment_angles()
            y_angle = -y_angle
        
        self.rotate_coord_sys_3D(y_axis, x_angle, x_axis, y_angle, invert=invert)

    
    # ==================================================
    def add_pointing_tilts(self, align_x_angle=None, align_y_angle=None):
        """
        Uses active transformation to tilt the beam in the zx- and zy-planes.
        
        Parameters
        ----------
         align_x_angle: [rad] float
            Beam coordinates with in the zx-plane are rotated with this angle.
        
        align_y_angle: [rad] float
            Beam coordinates with in the zy-plane are rotated with this angle. Note that due to the right hand rule, a positive rotation angle in the zy-plane corresponds to rotation from z-axis towards negative y. I.e. the opposite sign convention of beam.yps().
        
            
        Returns
        ----------
        Modified beam xs, ys and zs.
        """

        if align_x_angle is None:
            align_x_angle, _ = self.beam_alignment_angles()
        if align_y_angle is None:
            _, align_y_angle = self.beam_alignment_angles()
            align_y_angle = -align_y_angle

        y_axis = np.array([0, 0, 1])
        x_axis = np.array([0, 1, 0])
        
        zs = self.zs()
        xs = self.xs()
        ys = self.ys()

        # Combine into (N, 3) arrays
        coords = np.column_stack((zs, xs, ys))
        
        # Create the rotation object
        rotation_y = Rot.from_rotvec(align_x_angle * y_axis)
        rotation_x = Rot.from_rotvec(align_y_angle * x_axis)
        combined_rotation = rotation_y * rotation_x

        # Apply rotation to the coordinates only
        rotated_coords = combined_rotation.apply(coords, inverse=False)  # Active transformation

        # Extract the rotated coordinates
        rotated_zs, rotated_xs, rotated_ys = rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2]

        self.set_zs(rotated_zs)
        self.set_xs(rotated_xs)
        self.set_ys(rotated_ys)


    # ==================================================
    def slice_centroids(self, beam_quant, bin_number=None, cut_off=None, make_plot=False):
        """
        Returns the slice centroids of a beam quantity beam_quant.

        Parameters
        ----------
        beam_quant: 1D float array
            Beam quantity to be binned into bins/slices defined by z_centroids. The mean is calculated for the quantity for all particles in the z-bins. Includes e.g. beam.xs(), beam.Es() etc.

        bin_number: float
            Number of beam slices.

        cut_off: float
            Determines the longitudinal coordinates inside the region of interest

        make_plot: bool
            Flag for making plots.

            
        Returns
        ----------
        beam_quant_slices: 1D float array
            beam_quant binned into bins/slices defined by z_centroids. The mean is calculated for the quantity for all particles in the z-bins. Includes e.g. beam.xs(), beam.Es() etc.

        z_centroids: [m] 1D float array
            z-coordinates of the beam slices.
        """
        
        zs = self.zs()
        mean_z = self.z_offset()
        weights = self.weightings()

        if cut_off is None:
            cut_off = 1.5 * self.bunch_length()

        # Sort the arrays
        indices = np.argsort(zs)
        zs_sorted = zs[indices]  # Particle quantity.
        weights_sorted = weights[indices]  # Particle quantity.
        beam_quant_sorted = beam_quant[indices]  # Particle quantity.

        # Filter out elements outside the region of interest
        bool_indices = (zs_sorted <= mean_z + cut_off) & (zs_sorted >= mean_z - cut_off)
        zs_roi = zs_sorted[bool_indices]
        weights_roi = weights_sorted[bool_indices]
        beam_quant_roi = beam_quant_sorted[bool_indices]

        if bin_number is None:
            bin_number = int(np.sqrt(len(zs_roi)/2))

        # Beam slice zs
        _, edges = np.histogram(zs_roi, bins=bin_number)  # Get the edges of the histogram of z with bin_number bins.
        z_centroids = (edges[0:-1] + edges[1:])/2  # Centres of the beam slices (z)
        
        # Compute the mean of beam_quant of all particles inside a z-bin
        beam_quant_centroids = np.empty(len(z_centroids))
        for i in range(0,len(edges)-1):
            left = np.searchsorted(zs_roi, edges[i])  # zs_sorted[left:len(zs_sorted)] >= edges[i], left side of bin i.
            right = np.searchsorted(zs_roi, edges[i+1], side='right')  # zs_sorted[0:right] <= edges[i+1], right (larger) side of bin i.
            beam_quant_centroids[i] = weighted_mean(beam_quant_roi[left:right], weights_roi[left:right])

        if make_plot is True:
            plt.figure()
            plt.scatter(zs*1e6, beam_quant)
            plt.plot(z_centroids*1e6, beam_quant_centroids, 'rx-')
            plt.xlabel(r'$\xi$ [$\mathrm{\mu}$m]')
        
        return beam_quant_centroids, z_centroids
        
    
    # ==================================================
    def x_tilt_angle(self, z_cutoff=None):
        if z_cutoff is None:
            z_cutoff = 1.5 * self.bunch_length()
            
        x_centroids, z_centroids = self.slice_centroids(self.xs(), cut_off=z_cutoff, make_plot=False)
        
        # Perform linear regression
        slope, _ = np.polyfit(z_centroids, x_centroids, 1)
        
        return np.arctan(slope)

    
    # ==================================================
    def y_tilt_angle(self, z_cutoff=None):
        if z_cutoff is None:
            z_cutoff = 1.5 * self.bunch_length()
        y_centroids, z_centroids = self.slice_centroids(self.ys(), cut_off=z_cutoff, make_plot=False)
        
        # Perform linear regression
        slope, _ = np.polyfit(z_centroids, y_centroids, 1)
        
        return np.arctan(slope)


    
    ## BEAM STATISTICS

    def total_particles(self):
        return int(np.nansum(self.weightings()))
    
    def charge(self):
        return np.nansum(self.qs())
    
    def abs_charge(self):
        return abs(self.charge())
    
    def charge_sign(self):
        if self.charge() == 0:
            return 1.0
        else:
            return self.charge()/abs(self.charge())
    
    def energy(self, clean=False):
        return weighted_mean(self.Es(), self.weightings(), clean)
    
    def gamma(self, clean=False):
        return weighted_mean(self.gammas(), self.weightings(), clean)
    
    def total_energy(self):
        return SI.e * np.nansum(self.weightings()*self.Es())
    
    def energy_spread(self, clean=False):
        return weighted_std(self.Es(), self.weightings(), clean)
    
    def rel_energy_spread(self, clean=False):
        return self.energy_spread(clean)/self.energy(clean)
    
    def z_offset(self, clean=False):
        return weighted_mean(self.zs(), self.weightings(), clean)
    
    def bunch_length(self, clean=False):
        return weighted_std(self.zs(), self.weightings(), clean)
    
    def x_offset(self, clean=False):
        return weighted_mean(self.xs(), self.weightings(), clean)
    
    def beam_size_x(self, clean=False):
        return weighted_std(self.xs(), self.weightings(), clean)

    def y_offset(self, clean=False):
        return weighted_mean(self.ys(), self.weightings(), clean)

    def beam_size_y(self, clean=False):
        return weighted_std(self.ys(), self.weightings(), clean)
    
    def x_angle(self, clean=False):
        return weighted_mean(self.xps(), self.weightings(), clean)
    
    def divergence_x(self, clean=False):
        return weighted_std(self.xps(), self.weightings(), clean)

    def y_angle(self, clean=False):
        return weighted_mean(self.yps(), self.weightings(), clean)
    
    def divergence_y(self, clean=False):
        return weighted_std(self.yps(), self.weightings(), clean)
    
    def ux_offset(self, clean=False):
        return weighted_mean(self.uxs(), self.weightings(), clean)
    
    def uy_offset(self, clean=False):
        return weighted_mean(self.uys(), self.weightings(), clean)

    
    def geom_emittance_x(self, clean=False):
        return np.sqrt(np.linalg.det(weighted_cov(self.xs(), self.xps(), self.weightings(), clean)))
    
    def geom_emittance_y(self, clean=False):
        return np.sqrt(np.linalg.det(weighted_cov(self.ys(), self.yps(), self.weightings(), clean)))
    
    def norm_emittance_x(self, clean=False):
        return np.sqrt(np.linalg.det(weighted_cov(self.xs(), self.uxs()/SI.c, self.weightings(), clean)))
    
    def norm_emittance_y(self, clean=False):
        return np.sqrt(np.linalg.det(weighted_cov(self.ys(), self.uys()/SI.c, self.weightings(), clean)))
    
    def beta_x(self, clean=False):
        covx = weighted_cov(self.xs(), self.xps(), self.weightings(), clean)
        return covx[0,0]/np.sqrt(np.linalg.det(covx))
    
    def beta_y(self, clean=False):
        covy = weighted_cov(self.ys(), self.yps(), self.weightings(), clean)
        return covy[0,0]/np.sqrt(np.linalg.det(covy))
    
    def alpha_x(self, clean=False):
        covx = weighted_cov(self.xs(), self.xps(), self.weightings(), clean)
        return -covx[1,0]/np.sqrt(np.linalg.det(covx))
    
    def alpha_y(self, clean=False):
        covy = weighted_cov(self.ys(), self.yps(), self.weightings(), clean)
        return -covy[1,0]/np.sqrt(np.linalg.det(covy))
    
    def gamma_x(self, clean=False):
        covx = weighted_cov(self.xs(), self.xps(), self.weightings(), clean)
        return covx[1,1]/np.sqrt(np.linalg.det(covx))
    
    def gamma_y(self, clean=False):
        covy = weighted_cov(self.ys(), self.yps(), self.weightings(), clean)
        return covy[1,1]/np.sqrt(np.linalg.det(covy))

    def intrinsic_emittance(self):
        covxy = np.cov(self.norm_transverse_vector(), aweights=self.weightings())
        return np.sqrt(np.sqrt(np.linalg.det(covxy)))

    def angular_momentum(self):
        covxy = np.cov(self.norm_transverse_vector(), aweights=self.weightings())
        det_covxy_cross = np.linalg.det(covxy[2:4,0:2])
        return np.sign(covxy[3,0]-covxy[2,1])*np.sqrt(np.abs(det_covxy_cross))

    def eigen_emittance_max(self):
        return np.sqrt(self.norm_emittance_x()*self.norm_emittance_y()) + self.angular_momentum()

    def eigen_emittance_min(self):
        return np.sqrt(self.norm_emittance_x()*self.norm_emittance_y()) - self.angular_momentum()

    def norm_amplitude_x(self, plasma_density=None, clean=False):
        if plasma_density is not None:
            beta_x = beta_matched(plasma_density, self.energy())
            alpha_x = 0
        else:
            covx = weighted_cov(self.xs(), self.xps(), self.weightings(), clean)
            emgx = np.sqrt(np.linalg.det(covx))
            beta_x = covx[0,0]/emgx
            alpha_x = -covx[1,0]/emgx
        return np.sqrt(self.gamma()/beta_x)*np.sqrt(self.x_offset()**2 + (self.x_offset()*alpha_x + self.x_angle()*beta_x)**2)
        
    def norm_amplitude_y(self, plasma_density=None, clean=False):
        if plasma_density is not None:
            beta_y = beta_matched(plasma_density, self.energy())
            alpha_y = 0
        else:
            covy = weighted_cov(self.ys(), self.yps(), self.weightings(), clean)
            emgy = np.sqrt(np.linalg.det(covy))
            beta_y = covy[0,0]/emgy
            alpha_y = -covy[1,0]/emgy
        return np.sqrt(self.gamma()/beta_y)*np.sqrt(self.y_offset()**2 + (self.y_offset()*alpha_y + self.y_angle()*beta_y)**2)
        
    def peak_density(self):  # TODO: this is only valid for Gaussian beams.
        return (self.charge()/SI.e)/(np.sqrt(2*SI.pi)**3*self.beam_size_x()*self.beam_size_y()*self.bunch_length())
    
    def peak_current(self):
        Is, _ = self.current_profile()
        return max(abs(Is))
    

    ## BEAM HALO CLEANING (EXTREME OUTLIERS)
    def remove_halo_particles(self, nsigma=20):
        xfilter = np.abs(self.xs()-self.x_offset(clean=True)) > nsigma*self.beam_size_x(clean=True)
        xpfilter = np.abs(self.xps()-self.x_angle(clean=True)) > nsigma*self.divergence_x(clean=True)
        yfilter = np.abs(self.ys()-self.y_offset(clean=True)) > nsigma*self.beam_size_y(clean=True)
        ypfilter = np.abs(self.yps()-self.y_angle(clean=True)) > nsigma*self.divergence_y(clean=True)
        filter = np.logical_or(np.logical_or(xfilter, xpfilter), np.logical_or(yfilter, ypfilter))
        del self[filter]

    
    ## BEAM PROJECTIONS
    
    def projected_density(self, fcn, bins=None):
        if bins is None:
            Nbins = int(np.sqrt(len(self)/2))
            bins = np.linspace(min(fcn()), max(fcn()), Nbins)
        counts, edges = np.histogram(fcn(), weights=self.qs(), bins=bins)
        ctrs = (edges[0:-1] + edges[1:])/2
        proj = counts/np.diff(edges)
        return proj, ctrs
        
    def current_profile(self, bins=None):
        return self.projected_density(self.ts, bins=bins)
    
    def longitudinal_num_density(self, bins=None):
        dQdz, zs = self.projected_density(self.zs, bins=bins)
        dNdz = dQdz / SI.e / self.charge_sign()
        return dNdz, zs
    
    def energy_spectrum(self, bins=None):
        return self.projected_density(self.Es, bins=bins)
    
    def rel_energy_spectrum(self, nom_energy=None, bins=None):
        if nom_energy is None:
            nom_energy = self.energy()
        return self.projected_density(lambda: self.Es()/nom_energy-1, bins=bins)
    
    def transverse_profile_x(self, bins=None):
        return self.projected_density(self.xs, bins=bins)
    
    def transverse_profile_y(self, bins=None):
        return self.projected_density(self.ys, bins=bins)

    def transverse_profile_xp(self, bins=None):
        return self.projected_density(self.xps, bins=bins)
    
    def transverse_profile_yp(self, bins=None):
        return self.projected_density(self.yps, bins=bins)
    
    ## phase spaces
    
    def phase_space_density(self, hfcn, vfcn, hbins=None, vbins=None):
        self.remove_nans()
        if hbins is None:
            hbins = round(np.sqrt(len(self))/2)
        if vbins is None:
            vbins = round(np.sqrt(len(self))/2)
        counts, hedges, vedges = np.histogram2d(hfcn(), vfcn(), weights=self.qs(), bins=(hbins, vbins))
        hctrs = (hedges[0:-1] + hedges[1:])/2
        vctrs = (vedges[0:-1] + vedges[1:])/2
        density = (counts/np.diff(vedges)).T/np.diff(hedges)

        #dx = np.diff(hedges)
        #dy = np.diff(vedges)
        #bin_areas = dx[:, None] * dy[None, :]
        #density = counts/bin_areas
        #print(np.sum(density*np.diff(vedges)*np.diff(hedges))/self.charge())
        return density, hctrs, vctrs
    
    def density_lps(self, hbins=None, vbins=None):
        return self.phase_space_density(self.zs, self.Es, hbins=hbins, vbins=vbins)
    
    def density_transverse(self, hbins=None, vbins=None):
        return self.phase_space_density(self.xs, self.ys, hbins=hbins, vbins=vbins)

    
    # ==================================================
    # TODO: Currently does not reproduce the correct peak density for Gaussian beams unless the bin numbers are adjusted manually.
    def charge_density_3D(self, zbins=None, xbins=None, ybins=None):
        """
        Calculates the 3D charge density.
        
        Parameters
        ----------
        zbins, xbins, ybins: [m] float or 1D float ndarray
            The bins along z(x,y).
            
        Returns
        ----------
        dQ_dxdydz: [C/m^3] 3D float ndarray 
            Charge density of the beam.
        
        zctrs, xctrs, yctrs: [m] 1D float ndarray 
            The centre positions of the bins of dQ_dxdydz.
        """
        
        zs = self.zs()
        xs = self.xs()
        ys = self.ys()
        
        if zbins is None:
            zbins = round(np.sqrt(len(self))/2)
        if xbins is None:
            xbins = round(np.sqrt(len(self))/2)
        if ybins is None:
            ybins = round(np.sqrt(len(self))/2)
            
        # Create a 3D histogram
        counts, edges = np.histogramdd((zs, xs, ys), bins=(zbins, xbins, ybins), weights=self.qs())
        edges_z = edges[0]
        edges_x = edges[1]
        edges_y = edges[2]
        
        # Calculate volume of each bin
        dz = np.diff(edges_z)
        dx = np.diff(edges_x)
        dy = np.diff(edges_y)
        bin_volumes = dz[:, None, None] * dx[None, :, None] * dy[None, None, :]  # The None indexing is used to add new axes to the differences arrays, allowing them to be broadcasted properly for division with counts. This ensures that each element of counts is divided by the corresponding bin volume (element-wise division).
        
        # Calculate charge density per unit volume
        with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero
            dQ_dzdxdy = np.divide(counts, bin_volumes, out=np.zeros_like(counts), where=bin_volumes != 0)

        #dQ_dzdxdy, edges = np.histogramdd((zs, xs, ys), bins=(zbins, xbins, ybins), weights=self.qs(), density=True)       #######
        #dQ_dzdxdy = -dQ_dzdxdy

        zctrs = (edges_z[0:-1] + edges_z[1:])/2
        xctrs = (edges_x[0:-1] + edges_x[1:])/2
        yctrs = (edges_y[0:-1] + edges_y[1:])/2

        #print(np.sum(dQ_dzdxdy*bin_volumes)/self.charge())        
        
        return dQ_dzdxdy, zctrs, xctrs, yctrs, edges_z, edges_x, edges_y

    
    # ==================================================
    def Dirichlet_BC_system_matrix(self, main_diag, upper_inner_off_diag, lower_inner_off_diag, upper_outer_off_diag, lower_outer_off_diag, num_x_cells, num_unknowns, rhs, boundary_val):
        """
        Applies Dirichlet boundary conditions and assemble the system matrix and the right hand side (source term) of the Poisson equation.
        
        Parameters
        ----------
        main_diag: [m^-2] 1D float ndarray
            The main diagonal of the system matrix to be modified according to the boundary conditions.

        upper_inner_off_diag: [m^-2] 1D float ndarray
            The upper inne off-diagonal of the system matrix to be modified according to the boundary conditions.

        lower_inner_off_diag: [m^-2] 1D float ndarray
            The lower inne off-diagonal of the system matrix to be modified according to the boundary conditions.

        outer_inner_off_diag: [m^-2] 1D float ndarray
            The outer inne off-diagonal of the system matrix to be modified according to the boundary conditions.

        outer_inner_off_diag: [m^-2] 1D float ndarray
            The outer inne off-diagonal of the system matrix to be modified according to the boundary conditions.
            
        num_x_cells: float
            The number of cells in the x-direction. Determines the number of columns of the system matrix A.

        num_x_cells: float
            The number of unknowns in the system, which is determined by The number of cells in the x and y-direction.

        rhs: [V/m^3] 1D float ndarray
            The right hand side of the Poisson equation to be modified according to the boundary conditions.

        boundary_val: [V/m] float
            The value of the electric fields Ex and Ey at the simulation box boundary.

            
        Returns
        ----------
        A: [m^-2] 2D float sparse matrix
            System matrix.

        rhs: [V/m^3] 1D float ndarray
            The modified right hand side of the Poisson equation.
        """
        
        # Set the right side boundary conditions
        rhs[num_x_cells-1::num_x_cells] = boundary_val  # Set BC. Set every num_x_cells-th element starting from the num_x_cells-1 index to 1
        main_diag[num_x_cells-1::num_x_cells] = 1  # Set BC        
        upper_inner_off_diag[num_x_cells-1::num_x_cells] = 0  # Remove off-diagonal elements at boundaries
        lower_inner_off_diag[num_x_cells-2::num_x_cells] = 0  # Remove off-diagonal elements at boundaries
        upper_outer_off_diag[num_x_cells-1::num_x_cells] = 0  # Remove off-diagonal elements at boundaries
        lower_outer_off_diag[-1::-num_x_cells] = 0  # Remove off-diagonal elements at boundaries
        
        # Set the left side boundary conditions
        rhs[0::num_x_cells] = boundary_val
        main_diag[0::num_x_cells] = 1
        upper_inner_off_diag[0::num_x_cells] = 0
        lower_inner_off_diag[-num_x_cells::-num_x_cells] = 0
        upper_outer_off_diag[0::num_x_cells] = 0
        lower_outer_off_diag[-num_x_cells::-num_x_cells] = 0
        
        # Set the top boundary conditions
        rhs[1:num_x_cells-1] = boundary_val
        main_diag[1:num_x_cells-1] = 1
        upper_inner_off_diag[1:num_x_cells-1] = 0
        lower_inner_off_diag[0:num_x_cells-2] = 0
        upper_outer_off_diag[1:num_x_cells-1] = 0
        
        # Set the bottom boundary conditions
        rhs[-num_x_cells+1:-1] = boundary_val
        main_diag[-num_x_cells+1:-1] = 1
        upper_inner_off_diag[-num_x_cells+2:] = 0
        lower_inner_off_diag[-num_x_cells+1:-1] = 0
        lower_outer_off_diag[-num_x_cells+1:-1] = 0

        # Assemble the system matrix as a sparse diagonal dominant matrix
        diagonals = [main_diag, lower_inner_off_diag, upper_inner_off_diag, lower_outer_off_diag, upper_outer_off_diag]  # list
        offsets = [0, -1, 1, -num_x_cells, num_x_cells]  # Offsets of the diagonals. The outer diagonals outer_off_diag containing 1/dy^2 are num_x_cells away from the main diagonal.
        A = sp.diags(diagonals, offsets, shape=(num_unknowns, num_unknowns), format="csr")

        return A, rhs
        

    # ==================================================
    def Ex_Ey_2D(self, num_x_cells, num_y_cells, charge_density_xy_slice, dx, dy, boundary_val=0.0):
        """
        2D Poisson solver for the transverse electric fields Ex and Ey of a beam slice in the xy-plane. The equations solved are a combination of Gauss' law and Faraday's law assuming no time-varying z-component of magnetic field Bz. I.e.

        dEx/dx + dEy/dy = 1/epsilon_0 * dQ/dzdxdy
        dEy/dx - dEx/dy = 0.
        
        
        Parameters
        ----------
        num_x_cells, num_y_cells: float
            The number of cells in the x and y-direction.

        charge_density_xy_slice: [C/m^3] 2D ndarray
            A xy-slice of the beam charge density.

        dx, dy: [m] float
            Bin widths in x and y of the bins of dQ_dzdxdy.

        boundary_val: [V/m] 

            
        Returns
        ----------
        Ex: [V/m] 2D float array 
            x-conponent of electric field generated by the chosen beam slice.

        Ey: [V/m] 2D float array 
            y-conponent of electric field generated by the chosen beam slice.
        """
        
        num_rows, num_cols = charge_density_xy_slice.shape

        # Set up the system matrix
        num_unknowns = int(num_x_cells * num_y_cells)
        main_diag = np.ones(num_unknowns)* (-2/dx**2 - 2/dy**2)
        upper_inner_off_diag = np.ones(num_unknowns - 1)/dx**2
        lower_inner_off_diag = copy.deepcopy(upper_inner_off_diag)
        upper_outer_off_diag = np.ones(num_unknowns - num_x_cells)/dy**2
        lower_outer_off_diag = copy.deepcopy(upper_outer_off_diag)

        # Construct the right hand side of the Poisson equation for Ex
        rhs_2d = 1/SI.epsilon_0*np.gradient(charge_density_xy_slice, dx, axis=1)
        rhs = rhs_2d.flatten()

        # Apply Dirichlet boundary conditions
        A, rhs_BC = self.Dirichlet_BC_system_matrix(main_diag, upper_inner_off_diag, lower_inner_off_diag, upper_outer_off_diag, lower_outer_off_diag, num_x_cells, num_unknowns, rhs, boundary_val=boundary_val)

        # Solve the matrix equation for Ex
        #Ex = sp.linalg.spsolve(A, rhs_BC)
        Ex, has_converged_x = sp.linalg.cg(A, rhs_BC, x0=np.zeros(len(rhs_BC)), tol=1e-2)  # Works for positive definite A.

        # Construct the right hand side of the Poisson equation for Ey
        rhs_2d = 1/SI.epsilon_0*np.gradient(charge_density_xy_slice, dy, axis=0)
        rhs = rhs_2d.flatten()

        # Apply Dirichlet boundary conditions
        A, rhs_BC = self.Dirichlet_BC_system_matrix(main_diag, upper_inner_off_diag, lower_inner_off_diag, upper_outer_off_diag, lower_outer_off_diag, num_x_cells, num_unknowns, rhs, boundary_val=boundary_val)

        # Solve the matrix equation for Ey
        Ey, has_converged_y = sp.linalg.cg(A, rhs_BC, x0=np.zeros(len(rhs_BC)), tol=1e-2)  # Works for positive definite A.

        return Ex.reshape((num_rows, num_cols)), Ey.reshape((num_rows, num_cols))


    # ==================================================
    def Ex_Ey(self, x_box_min, x_box_max, y_box_min, y_box_max, dx, dy, num_z_cells=None, boundary_val=0.0, tolerance=5.0):
        """
        Calculate slice Ex and Ey for the entire beam by solving the Poisson equations for Ex and Ey slice by slice.

        Parameters
        ----------
        x_box_min, y_box_min: [m] float
            The lower x(y) boundary of the simulation domain. Should be much larger than the plasma bubble radius.

        x_box_max, y_box_max: [m] float
            The upper x(y) boundary of the simulation domain. Should be much larger than the plasma bubble radius.
        
        dx, dy: [m] float
            Bin widths in x and y of the bins of dQ_dzdxdy.

        num_z_cells: float
            The number of cells in the z-direction.

        boundary_val: [V/m]
            The values of the electric fields Ex and Ey at the simulation domain boundary.

            
        Returns
        ----------
        Ex: [V/m] 2D float array 
            x-conponent of electric field generated by the chosen beam slice.

        Ey: [V/m] 2D float array 
            y-conponent of electric field generated by the chosen beam slice.

        zctrs, xctrs, yctrs: [m] 1D float ndarray
            Coordinates in z, x and y for the centres of the bins of Ex and Ey.
        """

        # Check if the selected simulation boundaries are significantly larger than the beam extent
        xs = self.xs()
        ys = self.ys()
        
        if np.abs(x_box_min/xs.min()) < tolerance or np.abs(y_box_min/ys.min()) < tolerance or np.abs(x_box_max/xs.max()) < tolerance or np.abs(y_box_max/ys.max()) < tolerance:
            raise ValueError('Simulation box size is too small compared to beam size.')
        
        if num_z_cells is None:
            num_z_cells = round(np.sqrt(len(self))/2)

        z_box_max = np.max(self.zs())
        z_box_min = np.min(self.zs())
        zbins = np.linspace(z_box_min, z_box_max, num_z_cells+1)
        
        num_x_cells = int((x_box_max-x_box_min)/dx)
        num_y_cells = int((y_box_max-y_box_min)/dy)
        
        xbins = np.linspace(x_box_min, x_box_max, num_x_cells+1)
        ybins = np.linspace(y_box_min, y_box_max, num_y_cells+1)

        dQ_dzdxdy, zctrs, xctrs, yctrs, edges_z, edges_x, edges_y = self.charge_density_3D(zbins=zbins, xbins=xbins, ybins=ybins)
        
        Ex = np.zeros((num_z_cells, num_y_cells, num_x_cells))
        Ey = np.zeros((num_z_cells, num_y_cells, num_x_cells))

        for slice_idx in range(0, num_z_cells):
            # Extract a xy-slice from the charge density
            charge_density_xy_slice = dQ_dzdxdy[slice_idx, :, :].T

            # Calculate the fields for the charge density slice
            Ex_2d, Ey_2d = self.Ex_Ey_2D(num_x_cells, num_y_cells, charge_density_xy_slice, dx, dy, boundary_val=boundary_val)
            
            Ex[slice_idx,:,:] = Ex_2d
            Ey[slice_idx,:,:] = Ey_2d
            
        return Ex, Ey, zctrs, xctrs, yctrs
          
        
    
    ## PLOTTING
    def plot_current_profile(self):
        dQdt, ts = self.current_profile()

        fig, ax = plt.subplots()
        fig.set_figwidth(6)
        fig.set_figheight(4)        
        ax.plot(ts*SI.c*1e6, -dQdt/1e3)
        ax.set_xlabel('z (um)')
        ax.set_ylabel('Beam current (kA)')
    
    def plot_lps(self):
        dQdzdE, zs, Es = self.density_lps()

        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(5)  
            
        p = ax.pcolor(zs*1e6, Es/1e9, -dQdzdE*1e15, cmap=CONFIG.default_cmap, shading='auto')
        ax.set_xlabel('z (um)')
        ax.set_ylabel('E (GeV)')
        ax.set_title('Longitudinal phase space')
        cb = fig.colorbar(p)
        cb.ax.set_ylabel('Charge density (pC/um/GeV)')
        
    def plot_trace_space_x(self):
        dQdxdxp, xs, xps = self.phase_space_density(self.xs, self.xps)

        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(5)  
        p = ax.pcolor(xs*1e6, xps*1e3, -dQdxdxp*1e3, cmap=CONFIG.default_cmap, shading='auto')
        ax.set_xlabel('x (um)')
        ax.set_ylabel('x'' (mrad)')
        ax.set_title('Horizontal trace space')
        cb = fig.colorbar(p)
        cb.ax.set_ylabel('Charge density (pC/um/mrad)')
        
    def plot_trace_space_y(self):
        dQdydyp, ys, yps = self.phase_space_density(self.ys, self.yps)

        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(5)  
        p = ax.pcolor(ys*1e6, yps*1e3, -dQdydyp*1e3, cmap=CONFIG.default_cmap, shading='auto')
        ax.set_xlabel('y (um)')
        ax.set_ylabel('y'' (mrad)')
        ax.set_title('Vertical trace space')
        cb = fig.colorbar(p)
        cb.ax.set_ylabel('Charge density (pC/um/mrad)')

    def plot_transverse_profile(self):
        dQdxdy, xs, ys = self.phase_space_density(self.xs, self.ys)

        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(5)
        p = ax.pcolor(xs*1e6, ys*1e6, -dQdxdy, cmap=CONFIG.default_cmap, shading='auto')
        #p = ax.imshow(-dQdxdy, extent=[xs.min()*1e6, xs.max()*1e6, ys.min()*1e6, ys.max()*1e6], 
        #   origin='lower', cmap=CONFIG.default_cmap, aspect='auto')
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_title('Transverse profile')
        cb = fig.colorbar(p)
        cb.ax.set_ylabel('Charge density (pC/um^2)')

    
    def plot_bunch_pattern(self):
        
        fig, ax = plt.subplots()
        fig.set_figwidth(6)
        fig.set_figheight(4)        
        ax.plot(ts*SI.c*1e6, -dQdt/1e3)
        ax.set_xlabel('z (um)')
        ax.set_ylabel('Beam current (kA)')
        
        
    
    ## CHANGE BEAM
    
    def accelerate(self, energy_gain=0, chirp=0, z_offset=0):
        
        # add energy and chirp
        Es = self.Es() + energy_gain 
        Es = Es + np.sign(self.qs()) * (self.zs()-z_offset) * chirp
        self.set_uzs(energy2proper_velocity(Es))
        
        # remove particles with subzero energy
        del self[Es < 0]
        
    def compress(self, R_56, nom_energy):
        zs = self.zs() + (1-self.Es()/nom_energy) * R_56
        self.set_zs(zs)
        
    def scale_to_length(self, bunch_length):
        z_mean = self.z_offset()
        zs_scaled = z_mean + (self.zs()-z_mean)*bunch_length/self.bunch_length()
        self.set_zs(zs_scaled)

    def scale_norm_emittance_x(self, norm_emit_nx):
        scale_factor = norm_emit_nx/self.norm_emittance_x()
        self.set_xs(self.xs() * np.sqrt(scale_factor))
        self.set_uxs(self.uxs() * np.sqrt(scale_factor))

    def scale_norm_emittance_y(self, norm_emit_ny):
        scale_factor = norm_emit_ny/self.norm_emittance_y()
        self.set_ys(self.ys() * np.sqrt(scale_factor))
        self.set_uys(self.uys() * np.sqrt(scale_factor))
        
    # betatron damping (must be done before acceleration)
    def apply_betatron_damping(self, deltaE):
        gammasBoosted = energy2gamma(abs(self.Es()+deltaE))
        betamag = np.sqrt(self.gammas()/gammasBoosted)
        self.magnify_beta_function(betamag)
        
    
    # magnify beta function (increase beam size, decrease divergence)
    def magnify_beta_function(self, beta_mag, axis_defining_beam=None):
        
        # calculate beam (not beta) magnification
        mag = np.sqrt(beta_mag)

        if axis_defining_beam is None:
            x_offset = 0
            y_offset = 0
            ux_offset = 0
            uy_offset = 0
        else:
            x_offset = axis_defining_beam.x_offset()
            y_offset = axis_defining_beam.y_offset()
            ux_offset = axis_defining_beam.ux_offset()
            uy_offset = axis_defining_beam.uy_offset()
        
        self.set_xs((self.xs()-x_offset)*mag + x_offset)
        self.set_ys((self.ys()-y_offset)*mag + y_offset)
        self.set_uxs((self.uxs()-ux_offset)/mag + ux_offset)
        self.set_uys((self.uys()-uy_offset)/mag + uy_offset)

    
    # transport in a drift
    def transport(self, L):
        self.set_xs(self.xs() + L*self.xps())
        self.set_ys(self.ys() + L*self.yps())

    
    def flip_transverse_phase_spaces(self, flip_momenta=True, flip_positions=False):
        if flip_momenta:
            self.set_uxs(-self.uxs())
            self.set_uys(-self.uys())
        elif flip_positions:
            self.set_xs(-self.xs())
            self.set_ys(-self.ys())

        
    def apply_betatron_motion(self, L, n0, deltaEs, x0_driver=0, y0_driver=0, radiation_reaction=False, calc_evolution=False, evolution_samples=None):
        
        # remove particles with subzero energy
        neg_indices = self.Es() < 0
        del self[neg_indices]
        deltaEs = np.delete(deltaEs, neg_indices)
        nan_indices = np.isnan(self.Es())
        del self[nan_indices]
        deltaEs = np.delete(deltaEs, nan_indices)
        
        # determine initial and final Lorentz factor
        gamma0s = energy2gamma(self.Es())
        Es_final = self.Es()+deltaEs
        gammas = energy2gamma(Es_final)
        dgamma_ds = (gammas-gamma0s)/L
        
        if calc_evolution:
                
            # calculate evolution
            num_evol_steps = max(20, min(400, round(2*L/(beta_matched(n0, self.energy()+min(0,np.mean(deltaEs)))))))
            evol = SimpleNamespace()
            evol.location = np.linspace(0, L, num_evol_steps)
            evol.x, evol.ux, evol.ux, evol.y, evol.uy, evol.energy, evol.energy_spread, evol.rel_energy_spread, evol.beam_size_x, evol.beam_size_y, evol.emit_nx, evol.emit_ny, evol.beta_x, evol.beta_y = (np.empty(evol.location.shape) for _ in range(14))

            # select a given subset of the particles (for faster calculation)
            sample_fraction = round(2*np.sqrt(len(self))) #
            inds_sample = np.arange(0, len(self), sample_fraction, dtype=int) # not random, to be consistent across ramps and stages
            xs_sample, uxs_sample = self.xs()[inds_sample], self.uxs()[inds_sample]
            ys_sample, uys_sample = self.ys()[inds_sample], self.uys()[inds_sample]
            gamma0s_sample, dgamma_ds_sample = gamma0s[inds_sample], dgamma_ds[inds_sample]
            Es_sample, deltaEs_sample = self.Es()[inds_sample], deltaEs[inds_sample]

            # go through steps
            for i in range(num_evol_steps):
                
                # evolve the beam (no radiation reaction)
                xs_i, uxs_i = evolve_hills_equation_analytic(xs_sample-x0_driver, uxs_sample, evol.location[i], gamma0s_sample, dgamma_ds_sample, k_p(n0))
                ys_i, uys_i = evolve_hills_equation_analytic(ys_sample-y0_driver, uys_sample, evol.location[i], gamma0s_sample, dgamma_ds_sample, k_p(n0))
                Es_i = Es_sample+deltaEs_sample*i/(num_evol_steps-1)
                uzs_i = energy2proper_velocity(Es_i)
                
                # save the parameters
                evol.x[i], evol.ux[i] = np.mean(xs_i), np.mean(uxs_i)
                evol.y[i], evol.uy[i] = np.mean(ys_i), np.mean(uys_i)
                evol.energy[i], evol.energy_spread[i] = np.mean(Es_i), np.std(Es_i)
                evol.beam_size_x[i], evol.beam_size_y[i] = np.std(xs_i), np.std(ys_i)
                evol.emit_nx[i] = np.sqrt(np.linalg.det(np.cov(xs_i, uxs_i/SI.c))) # only works for equal weight particles
                evol.emit_ny[i] = np.sqrt(np.linalg.det(np.cov(ys_i, uys_i/SI.c))) # only works for equal weight particles
                covx, covy = np.cov(xs_i, uxs_i/uzs_i), np.cov(ys_i, uys_i/uzs_i)
                evol.beta_x[i] = covx[0,0]/np.sqrt(np.linalg.det(covx))
                evol.beta_y[i] = covy[0,0]/np.sqrt(np.linalg.det(covy))

            evol.rel_energy_spread = evol.energy_spread/evol.energy
            evol.charge = self.charge()*np.ones(evol.location.shape)
            evol.z = self.z_offset()*np.ones(evol.location.shape)
            evol.bunch_length = self.bunch_length()*np.ones(evol.location.shape)
            evol.plasma_density = n0*np.ones(evol.location.shape)
            
            
        # calculate final positions and angles after betatron motion
        if radiation_reaction:
            xs, uxs, ys, uys, Es_final = evolve_betatron_motion(self.xs()-x0_driver, self.uxs(), self.ys()-y0_driver, self.uys(), L, gamma0s, dgamma_ds, k_p(n0))
        else:
            xs, uxs = evolve_hills_equation_analytic(self.xs()-x0_driver, self.uxs(), L, gamma0s, dgamma_ds, k_p(n0))
            ys, uys = evolve_hills_equation_analytic(self.ys()-y0_driver, self.uys(), L, gamma0s, dgamma_ds, k_p(n0))
        
        # set new beam positions and angles (shift back driver offsets)
        self.set_xs(xs+x0_driver)
        self.set_uxs(uxs)
        self.set_ys(ys+y0_driver)
        self.set_uys(uys)

        if calc_evolution:
            return Es_final, evol
        else:
            return Es_final
        
  
    ## SAVE AND LOAD BEAM
    
    def filename(self, runnable, beam_name):
        return runnable.shot_path() + "/" + beam_name + "_" + str(self.trackable_number).zfill(3) + "_{:012.6F}".format(self.location) + ".h5"
    
    
    # save beam (to OpenPMD format)
    def save(self, runnable=None, filename=None, beam_name="beam", series=None):
        
        if len(self) == 0:
            return
        
        # make new file if not provided
        if series is None:

            # open a new file
            if runnable is not None:
                filename = self.filename(runnable, beam_name)
            
            # open a new file
            series = io.Series(filename, io.Access.create)
            
        
            # add metadata
            series.author = "ABEL (the Adaptable Beginning-to-End Linac simulation framework)"
            series.date = datetime.now(timezone('CET')).strftime('%Y-%m-%d %H:%M:%S %z')

        # make step (only one)
        index = 0
        iteration = series.iterations[index]
        
        # add attributes
        iteration.set_attribute("time", self.location/SI.c)
        for key, value in self.__dict__.items():
            if not "__phasespace" in key:
                iteration.set_attribute(key, value)
                
        # make beam record
        particles = iteration.particles[beam_name]
       
        # generate datasets
        dset_z = io.Dataset(self.zs().dtype, extent=self.zs().shape)
        dset_x = io.Dataset(self.xs().dtype, extent=self.xs().shape)
        dset_y = io.Dataset(self.ys().dtype, extent=self.ys().shape)
        dset_zoff = io.Dataset(np.dtype('float64'), extent=[1])
        dset_xoff = io.Dataset(np.dtype('float64'), extent=[1])
        dset_yoff = io.Dataset(np.dtype('float64'), extent=[1])
        dset_uz = io.Dataset(self.uzs().dtype, extent=self.uzs().shape)
        dset_ux = io.Dataset(self.uxs().dtype, extent=self.uxs().shape)
        dset_uy = io.Dataset(self.uys().dtype, extent=self.uys().shape)
        dset_w = io.Dataset(self.weightings().dtype, extent=self.weightings().shape)
        dset_id = io.Dataset(self.ids().dtype, extent=self.ids().shape)
        dset_q = io.Dataset(np.dtype('float64'), extent=[1])
        dset_m = io.Dataset(np.dtype('float64'), extent=[1])
        
        dset_n = io.Dataset(self.ids().dtype, extent=[1])
        dset_f = io.Dataset(np.dtype('float64'), extent=[1])
        
        # prepare for writing
        particles['position']['z'].reset_dataset(dset_z)
        particles['position']['x'].reset_dataset(dset_x)
        particles['position']['y'].reset_dataset(dset_y)
        particles['positionOffset']['z'].reset_dataset(dset_zoff)
        particles['positionOffset']['x'].reset_dataset(dset_xoff)
        particles['positionOffset']['y'].reset_dataset(dset_yoff)
        particles['momentum']['z'].reset_dataset(dset_uz)
        particles['momentum']['x'].reset_dataset(dset_ux)
        particles['momentum']['y'].reset_dataset(dset_uy)
        particles['weighting'][io.Record_Component.SCALAR].reset_dataset(dset_w)
        particles['id'][io.Record_Component.SCALAR].reset_dataset(dset_id)        
        particles['charge'][io.Record_Component.SCALAR].reset_dataset(dset_q)
        particles['mass'][io.Record_Component.SCALAR].reset_dataset(dset_m)
        
        # store data
        particles['position']['z'].store_chunk(self.zs())
        particles['position']['x'].store_chunk(self.xs())
        particles['position']['y'].store_chunk(self.ys())
        particles['positionOffset']['x'].make_constant(0.)
        particles['positionOffset']['y'].make_constant(0.)
        particles['positionOffset']['z'].make_constant(0.)
        particles['momentum']['z'].store_chunk(self.uzs())
        particles['momentum']['x'].store_chunk(self.uxs())
        particles['momentum']['y'].store_chunk(self.uys())
        particles['weighting'][io.Record_Component.SCALAR].store_chunk(self.weightings())
        particles['id'][io.Record_Component.SCALAR].store_chunk(self.ids())
        particles['charge'][io.Record_Component.SCALAR].make_constant(self.charge_sign()*SI.e)
        particles['mass'][io.Record_Component.SCALAR].make_constant(SI.m_e)
        
        # set SI units (scaling factor)
        particles['momentum']['z'].unit_SI = SI.m_e
        particles['momentum']['x'].unit_SI = SI.m_e
        particles['momentum']['y'].unit_SI = SI.m_e
        
        # set dimensional units
        particles['position'].unit_dimension = {io.Unit_Dimension.L: 1}
        particles['positionOffset'].unit_dimension = {io.Unit_Dimension.L: 1}
        particles['momentum'].unit_dimension = {io.Unit_Dimension.L: 1, io.Unit_Dimension.M: 1, io.Unit_Dimension.T: -1}
        particles['charge'].unit_dimension = {io.Unit_Dimension.T: 1, io.Unit_Dimension.I: 1}
        particles['mass'].unit_dimension = {io.Unit_Dimension.M: 1}
        
        # save data to file
        series.flush()
        
        return series
        
        
    # load beam (from OpenPMD format)
    @classmethod
    def load(_, filename, beam_name='beam'):
        
        # load file
        series = io.Series(filename, io.Access.read_only)
        
        # find index (use last one)
        *_, index = series.iterations
        
        # get particle data
        particles = series.iterations[index].particles[beam_name]
        
        # get attributes
        charge = particles["charge"][io.Record_Component.SCALAR].get_attribute("value")
        mass = particles["mass"][io.Record_Component.SCALAR].get_attribute("value")
        
        # extract phase space
        ids = particles["id"][io.Record_Component.SCALAR].load_chunk()
        weightings = particles["weighting"][io.Record_Component.SCALAR].load_chunk()
        xs = particles['position']['x'].load_chunk()
        ys = particles['position']['y'].load_chunk()
        zs = particles['position']['z'].load_chunk()
        pxs_unscaled = particles['momentum']['x'].load_chunk()
        pys_unscaled = particles['momentum']['y'].load_chunk()
        pzs_unscaled = particles['momentum']['z'].load_chunk()
        series.flush()
        
        # apply SI scaling
        pxs = pxs_unscaled * particles['momentum']['x'].unit_SI
        pys = pys_unscaled * particles['momentum']['y'].unit_SI
        pzs = pzs_unscaled * particles['momentum']['z'].unit_SI
        
        # make beam
        beam = Beam()
        beam.set_phase_space(Q=np.sum(weightings*charge), xs=xs, ys=ys, zs=zs, pxs=pxs, pys=pys, pzs=pzs, weightings=weightings)
        
        # add metadata to beam
        try:
            beam.trackable_number = series.iterations[index].get_attribute("trackable_number")
            beam.stage_number = series.iterations[index].get_attribute("stage_number")
            beam.location = series.iterations[index].get_attribute("location")
            beam.num_bunches_in_train = series.iterations[index].get_attribute("num_bunches_in_train")
            beam.bunch_separation = series.iterations[index].get_attribute("bunch_separation")
        except:
            beam.trackable_number = None
            beam.stage_number = None
            beam.location = None
            beam.num_bunches_in_train = None
            beam.bunch_separation = None
        
        return beam


    # ==================================================
    def imshow_plot(self, data, axes=None, extent=None, vmin=None, vmax=None, colmap='seismic', xlab=None, ylab=None, clab='', gridOn=False, origin='lower', interpolation=None, aspect='auto', log_cax=False, reduce_cax_pad=False):
        
        if axes is None:
            fig = plt.figure()  # an empty figure with an axes
            ax = fig.add_axes([.15, .15, .75, .75])
            cbar_ax = fig.add_axes([.85, .15, .03, .75])
        else:
            #ax = axes[0]  # TODO: adjust colourbar axes
            #cbar_ax = axes[1]
            
            ax = axes
            cbar_ax = None

        if reduce_cax_pad is True:
            # Create an axis on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cbar_ax = divider.append_axes("right", size="5%", pad=0.05)

        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()

        # Make a 2D plot
        if log_cax is True:
            p = ax.imshow(data, extent=extent, cmap=plt.get_cmap(colmap), origin=origin, aspect=aspect, interpolation=interpolation, norm=colors.LogNorm(vmin+1, vmax))
        else:
            p = ax.imshow(data, extent=extent, vmin=vmin, vmax=vmax, cmap=plt.get_cmap(colmap), origin=origin, aspect=aspect, interpolation=interpolation)

        # Add a grid
        if gridOn == True:
            ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)

        # Add a colourbar
        cbar = plt.colorbar(p, ax=ax, cax=cbar_ax)
        cbar.set_label(clab)

        # Set the tick formatter to use power notation
        #import matplotlib.ticker as ticker
        #cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        #cbar.ax.tick_params(axis='y', which='major', pad=10)

        #import matplotlib.ticker as ticker
        #fmt = ticker.ScalarFormatter(useMathText=True)
        #fmt.set_powerlimits((-3, 19))
        #cbar.ax.yaxis.set_major_formatter(fmt)

        # Customize the colorbar tick locator and formatter
        #from matplotlib.ticker import ScalarFormatter
        #cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))  # Set the number of tick intervals
        #cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # Use scientific notation

        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)

    
    # ==================================================
    def distribution_plot_2D(self, arr1, arr2, weights=None, hist_bins=None, hist_range=None, axes=None, extent=None, vmin=None, vmax=None, colmap=CONFIG.default_cmap, xlab='', ylab='', clab='', origin='lower', interpolation='nearest', reduce_cax_pad=False):

        if weights is None:
            weights = self.weightings()
        if hist_bins is None:
            nbins = int(np.sqrt(len(arr1)/2))
            hist_bins = [ nbins, nbins ]  # list of 2 ints. Number of bins along each direction, for the histograms
        if hist_range is None:
            hist_range = [[None, None], [None, None]]
            hist_range[0] = [ arr1.min(), arr1.max() ]  # List contains 2 lists of 2 floats. Extent of the histogram along each direction
            hist_range[1] = [ arr2.min(), arr2.max() ]
        if extent is None:
            extent = hist_range[0] + hist_range[1]
        
        binned_data, zedges, xedges = np.histogram2d(arr1, arr2, hist_bins, hist_range, weights=weights)
        beam_hist2d = binned_data.T/np.diff(zedges)/np.diff(xedges)
        self.imshow_plot(beam_hist2d, axes=axes, extent=extent, vmin=vmin, vmax=vmax, colmap=colmap, xlab=xlab, ylab=ylab, clab=clab, gridOn=False, origin=origin, interpolation=interpolation, reduce_cax_pad=reduce_cax_pad)

    
    # ==================================================
    def density_map_diags(self):
        
        #colors = ['white', 'aquamarine', 'lightgreen', 'green']
        #colors = ['white', 'forestgreen', 'limegreen', 'lawngreen', 'aquamarine', 'deepskyblue']
        #bounds = [0, 0.2, 0.4, 0.8, 1]
        #cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=256)
        
        cmap = CONFIG.default_cmap

        # Macroparticles data
        zs = self.zs()
        xs = self.xs()
        xps = self.xps()
        ys = self.ys()
        yps = self.yps()
        Es = self.Es()
        weights = self.weightings()

        # Labels for plots
        zlab = r'$z$ [$\mathrm{\mu}$m]'
        xilab = r'$\xi$ [$\mathrm{\mu}$m]'
        xlab = r'$x$ [$\mathrm{\mu}$m]'
        ylab = r'$y$ [$\mathrm{\mu}$m]'
        xps_lab = r"$x'$ [mrad]"
        yps_lab = r"$y'$ [mrad]"
        energ_lab = r'$\mathcal{E}$ [GeV]'
        
        # Set up a figure with axes
        fig, axs = plt.subplots(nrows=3, ncols=3, layout='constrained', figsize=(5*3, 4*3))
        fig.suptitle(r'$\Delta s=$' f'{format(self.location, ".2f")}' ' m')

        nbins = int(np.sqrt(len(weights)/2))
        hist_bins = [ nbins, nbins ]  # list of 2 ints. Number of bins along each direction, for the histograms

        # 2D z-x distribution
        hist_range = [[None, None], [None, None]]
        hist_range[0] = [ zs.min(), zs.max() ]  # [m], list contains 2 lists of 2 floats. Extent of the histogram along each direction
        hist_range[1] = [ xs.min(), xs.max() ]
        extent_zx = hist_range[0] + hist_range[1]
        extent_zx = [i*1e6 for i in extent_zx]  # [um]

        self.distribution_plot_2D(arr1=zs, arr2=xs, weights=weights, hist_bins=hist_bins, hist_range=hist_range, axes=axs[0][0], extent=extent_zx, vmin=None, vmax=None, colmap=cmap, xlab=xilab, ylab=xlab, clab=r'$\partial^2 N/\partial\xi \partial x$ [$\mathrm{m}^{-2}$]', origin='lower', interpolation='nearest')
        

        # 2D z-x' distribution
        hist_range_xps = [[None, None], [None, None]]
        hist_range_xps[0] = hist_range[0]
        hist_range_xps[1] = [ xps.min(), xps.max() ]  # [rad]
        extent_xps = hist_range_xps[0] + hist_range_xps[1]
        extent_xps[0] = extent_xps[0]*1e6  # [um]
        extent_xps[1] = extent_xps[1]*1e6  # [um]
        extent_xps[2] = extent_xps[2]*1e3  # [mrad]
        extent_xps[3] = extent_xps[3]*1e3  # [mrad]

        self.distribution_plot_2D(arr1=zs, arr2=xps, weights=weights, hist_bins=hist_bins, hist_range=hist_range_xps, axes=axs[0][1], extent=extent_xps, vmin=None, vmax=None, colmap=cmap, xlab=xilab, ylab=xps_lab, clab=r"$\partial^2 N/\partial z \partial x'$ [$\mathrm{m}^{-1}$ $\mathrm{rad}^{-1}$]", origin='lower', interpolation='nearest')
        
        
        # 2D x-x' distribution
        hist_range_xxp = [[None, None], [None, None]]
        hist_range_xxp[0] = hist_range[1]
        hist_range_xxp[1] = [ xps.min(), xps.max() ]  # [rad]
        extent_xxp = hist_range_xxp[0] + hist_range_xxp[1]
        extent_xxp[0] = extent_xxp[0]*1e6  # [um]
        extent_xxp[1] = extent_xxp[1]*1e6  # [um]
        extent_xxp[2] = extent_xxp[2]*1e3  # [mrad]
        extent_xxp[3] = extent_xxp[3]*1e3  # [mrad]

        self.distribution_plot_2D(arr1=xs, arr2=xps, weights=weights, hist_bins=hist_bins, hist_range=hist_range_xxp, axes=axs[0][2], extent=extent_xxp, vmin=None, vmax=None, colmap=cmap, xlab=xlab, ylab=xps_lab, clab=r"$\partial^2 N/\partial x\partial x'$ [$\mathrm{m}^{-1}$ $\mathrm{rad}^{-1}$]", origin='lower', interpolation='nearest')
        

        # 2D z-y distribution
        hist_range_zy = [[None, None], [None, None]]
        hist_range_zy[0] = hist_range[0]
        hist_range_zy[1] = [ ys.min(), ys.max() ]
        extent_zy = hist_range_zy[0] + hist_range_zy[1]
        extent_zy = [i*1e6 for i in extent_zy]  # [um]

        self.distribution_plot_2D(arr1=zs, arr2=ys, weights=weights, hist_bins=hist_bins, hist_range=hist_range_zy, axes=axs[1][0], extent=extent_zy, vmin=None, vmax=None, colmap=cmap, xlab=xilab, ylab=ylab, clab=r'$\partial^2 N/\partial\xi \partial y$ [$\mathrm{m}^{-2}$]', origin='lower', interpolation='nearest')
        

        # 2D z-y' distribution
        hist_range_yps = [[None, None], [None, None]]
        hist_range_yps[0] = hist_range[0]
        hist_range_yps[1] = [ yps.min(), yps.max() ]  # [rad]
        extent_yps = hist_range_yps[0] + hist_range_yps[1]
        extent_yps[0] = extent_yps[0]*1e6  # [um]
        extent_yps[1] = extent_yps[1]*1e6  # [um]
        extent_yps[2] = extent_yps[2]*1e3  # [mrad]
        extent_yps[3] = extent_yps[3]*1e3  # [mrad]
        
        self.distribution_plot_2D(arr1=zs, arr2=yps, weights=weights, hist_bins=hist_bins, hist_range=hist_range_yps, axes=axs[1][1], extent=extent_yps, vmin=None, vmax=None, colmap=cmap, xlab=xilab, ylab=yps_lab, clab=r"$\partial^2 N/\partial z \partial y'$ [$\mathrm{m}^{-1}$ $\mathrm{rad}^{-1}$]", origin='lower', interpolation='nearest')
        

        # 2D y-y' distribution
        hist_range_yyp = [[None, None], [None, None]]
        hist_range_yyp[0] = hist_range_zy[1]
        hist_range_yyp[1] = [ yps.min(), yps.max() ]  # [rad]
        extent_yyp = hist_range_yyp[0] + hist_range_yyp[1]
        extent_yyp[0] = extent_yyp[0]*1e6  # [um]
        extent_yyp[1] = extent_yyp[1]*1e6  # [um]
        extent_yyp[2] = extent_yyp[2]*1e3  # [mrad]
        extent_yyp[3] = extent_yyp[3]*1e3  # [mrad]
        
        self.distribution_plot_2D(arr1=ys, arr2=yps, weights=weights, hist_bins=hist_bins, hist_range=hist_range_yyp, axes=axs[1][2], extent=extent_yyp, vmin=None, vmax=None, colmap=cmap, xlab=ylab, ylab=yps_lab, clab=r"$\partial^2 N/\partial y\partial y'$ [$\mathrm{m}^{-1}$ $\mathrm{rad}^{-1}$]", origin='lower', interpolation='nearest')
       

        # 2D x-y distribution
        hist_range_xy = [[None, None], [None, None]]
        hist_range_xy[0] = hist_range[1]
        hist_range_xy[1] = hist_range_zy[1]
        extent_xy = hist_range_xy[0] + hist_range_xy[1]
        extent_xy = [i*1e6 for i in extent_xy]  # [um]

        self.distribution_plot_2D(arr1=xs, arr2=ys, weights=weights, hist_bins=hist_bins, hist_range=hist_range_xy, axes=axs[2][0], extent=extent_xy, vmin=None, vmax=None, colmap=cmap, xlab=xlab, ylab=ylab, clab=r'$\partial^2 N/\partial x \partial y$ [$\mathrm{m}^{-2}$]', origin='lower', interpolation='nearest')
        

        # Energy distribution
        ax = axs[2][1]
        dN_dE, rel_energ = self.rel_energy_spectrum()
        dN_dE = dN_dE/SI.e*self.charge_sign()
        ax.fill_between(rel_energ*100, y1=dN_dE, y2=0, color='b', alpha=0.3)
        ax.plot(rel_energ*100, dN_dE, color='b', alpha=0.3, label='Relative energy density')
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=1, alpha=.5)
        ax.set_xlabel(r'$\mathcal{E}/\langle\mathcal{E}\rangle-1$ [%]')
        ax.set_ylabel('Relative energy density')
        # Add text to the plot
        ax.text(0.05, 0.95, r'$\sigma_\mathcal{E}/\langle\mathcal{E}\rangle=$' f'{format(self.rel_energy_spread()*100, ".2f")}' '%', fontsize=12, color='black', ha='left', va='top', transform=ax.transAxes)

        # 2D z-energy distribution
        hist_range_energ = [[None, None], [None, None]]
        hist_range_energ[0] = hist_range[0]
        hist_range_energ[1] = [ Es.min(), Es.max() ]  # [eV]
        extent_energ = hist_range_energ[0] + hist_range_energ[1]
        extent_energ[0] = extent_energ[0]*1e6  # [um]
        extent_energ[1] = extent_energ[1]*1e6  # [um]
        extent_energ[2] = extent_energ[2]/1e9  # [GeV]
        extent_energ[3] = extent_energ[3]/1e9  # [GeV]
        self.distribution_plot_2D(arr1=zs, arr2=Es, weights=weights, hist_bins=hist_bins, hist_range=hist_range_energ, axes=axs[2][2], extent=extent_energ, vmin=None, vmax=None, colmap=cmap, xlab=xilab, ylab=energ_lab, clab=r'$\partial^2 N/\partial \xi \partial\mathcal{E}$ [$\mathrm{m}^{-1}$ $\mathrm{eV}^{-1}$]', origin='lower', interpolation='nearest')
