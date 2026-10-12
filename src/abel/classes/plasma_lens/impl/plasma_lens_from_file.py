
import re
import numpy as np
import scipy.constants as SI
from scipy.interpolate import LinearNDInterpolator
from abel import PlasmaLens

class PlasmaLensFromFile(PlasmaLens):
    # This class uses a transverse magnetic-field map from file instead of
    # an analytic plasma-lens field. The selected 2D map is used along the
    # full lens length, which is divided into num_slice tracking steps.

    # Interpolators are rebuilt when an object is unpickled because scipy's
    # interpolator objects are not always safe to serialize directly.
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('Bx_interpolator', None)
        state.pop('By_interpolator', None)
        return state

    def __setstate__(self, state):
        """Restore the field map and rebuild its interpolators after loading it."""
        self.__dict__.update(state)
        self.set_discharge_time(self.selected_time)

    # Configure the lens, load the maps, determine the usable radius, and
    # build interpolators for the requested discharge time.
    def __init__(self, bx_file, by_file, discharge_time=None, length=None, radius=None, num_slice=30, offset_x=0.0, offset_y=0.0, time_scale=1.0, uniform_bx=0.0, uniform_by=0.0, field_polarity=1):
        self.bx_file = bx_file
        self.by_file = by_file
        self.length = length
        self.requested_radius = radius
        self.discharge_time = discharge_time
        self.num_slice = int(num_slice)
        self.offset_x = float(offset_x)
        self.offset_y = float(offset_y)
        self.time_scale = float(time_scale)
        self.uniform_bx = float(uniform_bx)
        self.uniform_by = float(uniform_by)
        if field_polarity not in (-1, 1):
            raise ValueError('field_polarity must be +/-1')
        self.field_polarity = int(field_polarity)
        if length is None:
            raise ValueError('length must be specified.')
        if length <= 0:
            raise ValueError('length must be > 0.')
        if self.num_slice < 1:
            raise ValueError('num_slice must be >= 1.')
        # Read both field components and establish the map's available times.
        self._load_field_map()
        self._calculate_map_radius()
        # Never track beyond the imported map: default to its full radius,
        # or allow the user to select a smaller circular tracking aperture.
        if radius is None:
            tracking_radius = self.map_radius
        else:
            radius = float(radius)
            if radius <= 0:
                raise ValueError('radius must be > 0.')
            if radius > self.map_radius:
                raise ValueError(f'\nRequested tracking radius is larger than the imported field map.\n\nRequested radius : {radius:.6e} m\nMap radius       : {self.map_radius:.6e} m\n')
            tracking_radius = radius
        super().__init__(length=length, radius=tracking_radius, current=None, offset_x=offset_x, offset_y=offset_y)
        self.radius = tracking_radius
        if discharge_time is None:
            self.discharge_time = self.available_times[0]
        self.set_discharge_time(self.discharge_time)
        print()
        print('=' * 72)
        print('Imported plasma-lens field map')
        print('=' * 72)
        print(f'Bx file              : {self.bx_file}')
        print(f'By file              : {self.by_file}')
        print(f'Number of grid points: {len(self.x_all)}')
        print(f'Number of time steps : {len(self.available_times)}')
        print(f'Map radius           : {self.map_radius:.6e} m')
        print(f'Tracking radius      : {self.radius:.6e} m')
        print(f'Lens length          : {self.length:.6e} m')
        print(f'Number of slices     : {self.num_slice}')
        print(f'Selected time        : {self.selected_time:.6e}')
        print('=' * 72)
        print()

    # Estimate the local field gradient around the lens' centre. For a general
    # imported map, this is an effective value rather than an exact quadrupole
    # gradient.
    def get_focusing_gradient(self):
        """
        Return an effective focusing gradient [T/m].
        
        The imported field map is not necessarily a perfect linear
        quadrupole, so there is generally no unique focusing gradient.

        We estimate the local on-axis gradient numerically from:
            dBy/dx and dBx/dy
        and return average magnitudes.
        """
        
        if self.map_radius <= 0:
            return 0.0
        dx = min(1e-06, 0.001 * self.map_radius)
        if dx <= 0:
            return 0.0
        _, bx0, by0 = self.get_field(x=0.0, y=0.0)
        _, bxx, byx = self.get_field(x=dx, y=0.0)
        _, bxy, byy = self.get_field(x=0.0, y=dx)
        dBy_dx = (float(np.asarray(byx)) - float(np.asarray(by0))) / dx
        dBx_dy = (float(np.asarray(bxy)) - float(np.asarray(bx0))) / dx
        gradient = 0.5 * (abs(dBy_dx) + abs(dBx_dy))
        return float(gradient)

    # COMSOL stores discharge times in a header line; extract the numbers
    # following entries such as "@ t=...".
    @staticmethod
    def _read_time_values(filename):
        header_line = None
        with open(filename, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('% x'):
                    header_line = stripped
                    break
        if header_line is None:
            raise ValueError(f'Could not find COMSOL header in:\n{filename}')
        pattern = '@\\s*t\\s*=\\s*([-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?)'
        matches = re.findall(pattern, header_line)
        if len(matches) == 0:
            raise ValueError(f'No discharge times found in:\n{filename}')
        return np.asarray([float(value) for value in matches], dtype=float)

    # Read the numerical table, ignoring comment/header lines beginning with %.
    @staticmethod
    def _read_data(filename):
        try:
            data = np.loadtxt(filename, comments='%', dtype=float)
        except Exception as exc:
            raise ValueError(f'Could not read field map:\n{filename}\n\n{exc}') from exc
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return data

    # Load the coordinate columns and all field-time columns. Bx and By must
    # use the same grid and list the same discharge times for this to work.
    def _load_field_map(self):
        print()
        print('Loading imported field map...')
        self.bx_data = self._read_data(self.bx_file)
        self.by_data = self._read_data(self.by_file)
        if self.bx_data.shape != self.by_data.shape:
            raise ValueError(f'\nBx and By files have different shapes:\nBx: {self.bx_data.shape}\nBy: {self.by_data.shape}')
        bx_times_raw = self._read_time_values(self.bx_file)
        by_times_raw = self._read_time_values(self.by_file)
        if len(bx_times_raw) != len(by_times_raw):
            raise ValueError('Bx and By contain different numbers of discharge times.')
        if not np.allclose(bx_times_raw, by_times_raw, rtol=0.0, atol=1e-12):
            raise ValueError('Bx and By contain different discharge times.')
        self.available_times_raw = bx_times_raw.copy()
        self.available_times = bx_times_raw * self.time_scale
        self.x_all = self.bx_data[:, 0]
        self.y_all = self.bx_data[:, 1]
        self.z_all = self.bx_data[:, 2]
        if not np.allclose(self.x_all, self.by_data[:, 0], rtol=0.0, atol=1e-15):
            raise ValueError('Bx and By x coordinates do not match.')
        if not np.allclose(self.y_all, self.by_data[:, 1], rtol=0.0, atol=1e-15):
            raise ValueError('Bx and By y coordinates do not match.')
        if not np.allclose(self.z_all, self.by_data[:, 2], rtol=0.0, atol=1e-15):
            raise ValueError('Bx and By z coordinates do not match.')
        expected_columns = 3 + len(self.available_times)
        if self.bx_data.shape[1] != expected_columns:
            raise ValueError(f'\nUnexpected number of columns.\nExpected: {expected_columns}\nFound:    {self.bx_data.shape[1]}')
        print(f'  Grid points : {len(self.x_all)}')
        print(f'  Time steps  : {len(self.available_times)}')
        print(f'  Columns     : {self.bx_data.shape[1]}')
        print(f'  Time range  : {self.available_times[0]:.6e} -> {self.available_times[-1]:.6e}')

    # Define the available map radius from the largest transverse distance
    # with at least one finite field value at any discharge time.
    def _calculate_map_radius(self):
        valid_field = np.any(np.isfinite(self.bx_data[:, 3:]), axis=1) | np.any(np.isfinite(self.by_data[:, 3:]), axis=1)
        if not np.any(valid_field):
            raise ValueError('No finite magnetic-field points were found in the imported map.')
        x = self.x_all[valid_field]
        y = self.y_all[valid_field]
        r = np.sqrt(x ** 2 + y ** 2)
        self.map_radius = float(np.max(r))

    # Select the nearest available discharge time and rebuild the interpolators
    # from the corresponding Bx and By columns.
    def set_discharge_time(self, discharge_time):
        discharge_time = float(discharge_time)
        self.time_index = int(np.argmin(np.abs(self.available_times - discharge_time)))
        self.discharge_time = discharge_time
        self.selected_time = float(self.available_times[self.time_index])
        self.selected_time_raw = float(self.available_times_raw[self.time_index])
        column = 3 + self.time_index
        bx = self.bx_data[:, column]
        by = self.by_data[:, column]
        valid = np.isfinite(self.x_all) & np.isfinite(self.y_all) & np.isfinite(bx) & np.isfinite(by)
        self.valid_mask = valid
        self.x_map = self.x_all[valid]
        self.y_map = self.y_all[valid]
        self.Bx_map = bx[valid]
        self.By_map = by[valid]
        if len(self.x_map) == 0:
            raise ValueError(f'No valid Bx/By points found for discharge time {discharge_time}.')
        points = np.column_stack((self.x_map, self.y_map))
        self.Bx_interpolator = LinearNDInterpolator(points, self.Bx_map, fill_value=0.0)
        self.By_interpolator = LinearNDInterpolator(points, self.By_map, fill_value=0.0)
        return self.selected_time

    # Alias retained for convenience when changing the selected timestep.
    def select_timestep(self, discharge_time):
        return self.set_discharge_time(discharge_time)

    def get_discharge_times(self):
        return self.available_times.copy()

    def get_discharge_times_raw(self):
        return self.available_times_raw.copy()

    # Query the interpolated field at transverse coordinates (x, y).
    # The x reflection matches the map/lens coordinate convention; offsets
    # move the lens relative to the beam, and uniform fields are then added.
    def get_field(self, discharge_time=None, x=0.0, y=0.0):
        if discharge_time is not None:
            selected_time = self.set_discharge_time(discharge_time)
        else:
            selected_time = self.selected_time
        x_local = -1 * (np.asarray(x) - self.offset_x)
        y_local = 1 * (np.asarray(y) - self.offset_y)
        Bx = self.Bx_interpolator(x_local, y_local)
        By = self.By_interpolator(x_local, y_local)
        Bx = np.nan_to_num(Bx, nan=0.0)
        By = np.nan_to_num(By, nan=0.0)
        Bx = self.field_polarity * Bx + self.uniform_bx
        By = 1 * By + self.uniform_by
        return (selected_time, Bx, By)

    # Return the selected map coordinates and total Bx/By values, including
    # the configured polarity and uniform external-field contributions.
    def get_field_map(self, discharge_time=None):
        if discharge_time is not None:
            self.set_discharge_time(discharge_time)
        return (self.field_polarity * self.x_map.copy(), self.y_map.copy(), self.field_polarity * self.Bx_map.copy() + self.uniform_bx, self.By_map.copy() + self.uniform_by, self.selected_time)

    # Visualise imported fields separately from the total fields after the
    # uniform external components have been added.
    def plot_field_map(self, figsize=(15, 9), marker_size=8, cmap='coolwarm'):
        """Plot imported and total Bx, By, and |B| maps for the selected time."""
        import matplotlib.pyplot as plt
        x_map, y_map, Bx_total, By_total, selected_time = self.get_field_map()
        Bx_imported = Bx_total - self.uniform_bx
        By_imported = By_total - self.uniform_by
        B_imported = np.hypot(Bx_imported, By_imported)
        B_total = np.hypot(Bx_total, By_total)
        fields = [(Bx_imported, 'Imported $B_x$ [T]'), (By_imported, 'Imported $B_y$ [T]'), (B_imported, 'Imported $|B|$ [T]'), (Bx_total, 'Total $B_x$ [T]'), (By_total, 'Total $B_y$ [T]'), (B_total, 'Total $|B|$ [T]')]
        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        for ax, (values, title) in zip(axes.flat, fields):
            points = ax.scatter(x_map * 1000000.0, y_map * 1000000.0, c=values, s=marker_size, cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel('x [µm]')
            ax.set_ylabel('y [µm]')
            ax.set_aspect('equal')
            fig.colorbar(points, ax=ax)
        fig.suptitle(f'Field map at t = {selected_time * 1000000000.0:g} ns')
        plt.show()
        return (fig, axes)

    def get_map_radius(self):
        return self.map_radius

    def get_radius(self):
        return self.radius

    # Convenient helper returning |B| = sqrt(Bx^2 + By^2) at a point.
    def get_field_magnitude(self, discharge_time=None, x=0.0, y=0.0):
        selected_time, Bx, By = self.get_field(discharge_time=discharge_time, x=x, y=y)
        B = np.sqrt(Bx ** 2 + By ** 2)
        return (selected_time, B)

    # Estimate how much the field magnitude divided by radius varies across
    # the map. This is a diagnostic metric, not a full particle-dynamics result.
    def get_relativistic_nonlinearity(self, discharge_time=None):
        if discharge_time is not None:
            self.set_discharge_time(discharge_time)
        x = self.field_polarity * self.x_map
        y = self.y_map
        Bx = self.field_polarity * self.Bx_map + self.uniform_bx
        By = 1 * self.By_map + self.uniform_by
        r = np.sqrt(x ** 2 + y ** 2)
        B = np.sqrt(Bx ** 2 + By ** 2)
        mask = r > 1e-12
        r = r[mask]
        B = B[mask]
        if len(r) == 0:
            return 0.0
        B_over_r = B / r
        mean_value = np.mean(np.abs(B_over_r))
        if mean_value == 0:
            return 0.0
        nonlinearity = np.std(B_over_r) / mean_value
        return float(nonlinearity)

    # Print the selected map, geometry, field ranges, and diagnostic metrics.
    def field_info(self):
        print()
        print('=' * 72)
        print('PLASMA LENS FIELD MAP')
        print('=' * 72)
        print(f'Selected time       : {self.selected_time:.6e}')
        print(f'Raw selected time   : {self.selected_time_raw:.6e}')
        print(f'Time index          : {self.time_index}')
        print(f'Number of points    : {len(self.x_map)}')
        print(f'Map radius          : {self.map_radius:.6e} m')
        print(f'Tracking radius     : {self.radius:.6e} m')
        print(f'Lens length         : {self.length:.6e} m')
        print(f'Longitudinal slices : {self.num_slice}')
        print()
        print(f'x range             : {self.x_map.min():.6e} -> {self.x_map.max():.6e} m')
        print(f'y range             : {self.y_map.min():.6e} -> {self.y_map.max():.6e} m')
        print()
        print(f'Imported-map polarity: {self.field_polarity:+d}')
        print(f'Uniform Bx          : {self.uniform_bx:.6e} T')
        print(f'Uniform By          : {self.uniform_by:.6e} T')
        print(f'Bx range            : {(self.field_polarity * self.Bx_map + self.uniform_bx).min():.6e} -> {(self.field_polarity * self.Bx_map + self.uniform_bx).max():.6e} T')
        print(f'By range            : {(1 * self.By_map + self.uniform_by).min():.6e} -> {(1 * self.By_map + self.uniform_by).max():.6e} T')
        print()
        print(f'Focusing gradient   : {self.get_focusing_gradient():.6e} T/m')
        print(f'Field nonlinearity  : {self.get_relativistic_nonlinearity():.6e}')
        print('=' * 72)
        print()

    # Track through a thick lens by splitting it into longitudinal slices, analogous to plasma_lens_nonlinear_thick.py.
    # At each slice, interpolate Bx/By at each particle and apply the magnetic
    # Lorentz-force angular kicks, with half-slice drifts at the entrance/exit.
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        slice_length = self.length / self.num_slice
        half_slice = slice_length / 2.0
        r2 = (beam.xs() - self.offset_x) ** 2 + (beam.ys() - self.offset_y) ** 2
        del beam[r2 > self.radius ** 2]
        if len(beam) == 0:
            return super().track(beam, savedepth, runnable, verbose)
        # Start with a half drift so the first kick is centred in the first slice.
        beam.transport(half_slice)
        for i in range(self.num_slice):
            r2 = (beam.xs() - self.offset_x) ** 2 + (beam.ys() - self.offset_y) ** 2
            del beam[r2 > self.radius ** 2]
            if len(beam) == 0:
                return super().track(beam, savedepth, runnable, verbose)
            x = -(beam.xs() - self.offset_x)
            y = beam.ys() - self.offset_y
            Bx = self.Bx_interpolator(x, y)
            By = self.By_interpolator(x, y)
            Bx = np.nan_to_num(Bx, nan=0.0)
            By = np.nan_to_num(By, nan=0.0)
            Bx = self.field_polarity * Bx + self.uniform_bx
            By = 1 * By + self.uniform_by
            # Apply magnetic kicks using the particle charge sign and energy.
            # The field components follow the transverse Lorentz-force convention.
            delta_xp = slice_length * By * beam.charge_sign() * SI.c / beam.Es()
            delta_yp = -slice_length * Bx * beam.charge_sign() * SI.c / beam.Es()
            beam.set_xps(beam.xps() + delta_xp)
            beam.set_yps(beam.yps() + delta_yp)
            if i < self.num_slice - 1:
                beam.transport(slice_length)
        beam.transport(half_slice)
        r2 = (beam.xs() - self.offset_x) ** 2 + (beam.ys() - self.offset_y) ** 2
        del beam[r2 > self.radius ** 2]
        return super().track(beam, savedepth, runnable, verbose)
