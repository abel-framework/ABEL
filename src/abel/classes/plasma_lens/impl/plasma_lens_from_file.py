
import re
import numpy as np
import scipy.constants as SI
from scipy.interpolate import LinearNDInterpolator, interp1d
from abel import PlasmaLens

class PlasmaLensFromFile(PlasmaLens):
    # This class uses a transverse magnetic-field map from file instead of
    # an analytic plasma-lens field. The selected 2D map is used along the
    # full lens length, which is divided into num_slice tracking steps.

    # Interpolators are rebuilt when an object is unpickled because scipy's
    # interpolator objects are not always safe to serialize directly.
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('interpolator_Bx', None)
        state.pop('interpolator_By', None)
        return state

    def __setstate__(self, state):
        """Restore the field map and rebuild interpolators after loading."""
        self.__dict__.update(state)

        if hasattr(self, 'data_Bx') and hasattr(self, 'selected_time'):
            self.set_discharge_time(self.selected_time)
        
    def __init__(self, filename_Bx=None, filename_By=None, discharge_time=None,
                 length=None, num_slice=None, offset_x=None, offset_y=None,
                 external_magnetic_field_x=None, external_magnetic_field_y=None,
                 imported_magnetic_field_polarity=None):
        # Configuration is checked and maps are loaded only when load() is called.
        super().__init__(length=length, radius=None, current=None,
                         offset_x=offset_x, offset_y=offset_y)
        self.filename_Bx = filename_Bx
        self.filename_By = filename_By
        self.discharge_time = discharge_time
        self.num_slice = num_slice
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.external_magnetic_field_x = external_magnetic_field_x
        self.external_magnetic_field_y = external_magnetic_field_y
        self.imported_magnetic_field_polarity = imported_magnetic_field_polarity
    

    def load(self, filename_Bx=None, filename_By=None, time_scale=None):
        """Validate settings, load both field maps, and build interpolators."""
        try:
            if filename_Bx is not None:
                self.filename_Bx = filename_Bx
            if filename_By is not None:
                self.filename_By = filename_By
            self.time_scale = 1.0 if time_scale is None else float(time_scale)

            if not np.isfinite(self.time_scale) or self.time_scale <= 0:
                raise ValueError('time_scale must be a finite value greater than 0.')
            if self.filename_Bx is None or self.filename_By is None:
                raise ValueError('Both filename_Bx and filename_By must be set before loading.')
            if self.length is None or float(self.length) <= 0:
                raise ValueError('length must be set to a value greater than 0 before loading.')

            self.length = float(self.length)
            self.num_slice = int(30 if self.num_slice is None else self.num_slice)
            if self.num_slice < 1:
                raise ValueError('num_slice must be >= 1.')
            self.offset_x = float(0.0 if self.offset_x is None else self.offset_x)
            self.offset_y = float(0.0 if self.offset_y is None else self.offset_y)
            self.external_magnetic_field_x = float(
                0.0 if self.external_magnetic_field_x is None else self.external_magnetic_field_x)
            self.external_magnetic_field_y = float(
                0.0 if self.external_magnetic_field_y is None else self.external_magnetic_field_y)
            self.imported_magnetic_field_polarity = int(
                1 if self.imported_magnetic_field_polarity is None
                else self.imported_magnetic_field_polarity)
            if self.imported_magnetic_field_polarity not in (-1, 1):
                raise ValueError('imported_magnetic_field_polarity must be +/-1.')

            self._load_field_map()
            self._calculate_map_radius()
            self.radius = self.map_radius
            if self.discharge_time is None:
                self.discharge_time = self.available_times[0]
            self.set_discharge_time(self.discharge_time)
            return self
        except Exception as exc:
            # Keep the original exception type and traceback, adding load context.
            exc.add_note('PlasmaLensFromFile.load() failed while configuring or loading the field map.')
            raise
        
    def _require_loaded(self):
        if not hasattr(self, 'interpolator_Bx') or not hasattr(self, 'interpolator_By'):
            try:
                self.load()
            except Exception as exc:
                raise RuntimeError(
                    'Field maps could not be loaded automatically. '
                    'Check the file paths and configuration.'
                ) from exc 
                
    @staticmethod
    def _read_time_values(filename):
        try:
            with open(filename, 'r') as file:
                header_line = next(
                    (line.strip() for line in file
                     if line.strip().startswith('%') and '@ t' in line), None)
        except OSError as exc:
            raise OSError(f'Could not open field-map file: {filename}') from exc
        if header_line is None:
            raise ValueError(f'Could not find COMSOL header in:\n{filename}')
        pattern = r'@\s*t\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'
        matches = re.findall(pattern, header_line)
        if not matches:
            raise ValueError(f'No discharge times found in:\n{filename}')
        return np.asarray([float(value) for value in matches], dtype=float)

    @staticmethod
    def _coordinate_count(filename):
        """Return the number of leading coordinate columns (2 or 3)."""
        with open(filename, 'r') as file:
            header = next((line.strip() for line in file
                           if line.strip().startswith('%') and '@ t' in line), None)
        if header is None:
            raise ValueError(f'Could not find COMSOL header in:\n{filename}')
        columns = header[1:].split()
        if columns[:3] == ['x', 'y', 'z']:
            return 3
        if columns[:2] in (['x', 'y'], ['cpl1x', 'cpl1y']):
            return 2
        raise ValueError(f'Expected x/y or x/y/z coordinates in:\n{filename}')

    @staticmethod
    def _read_data(filename):
        try:
            data = np.loadtxt(filename, comments='%', dtype=float)
        except Exception as exc:
            raise ValueError(f'Could not read field map:\n{filename}\n\n{exc}') from exc
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError(f'Field map must contain x, y, z and at least one field column:\n{filename}')
        return data

    def _load_field_map(self):
        self.data_Bx = self._read_data(self.filename_Bx)
        self.data_By = self._read_data(self.filename_By)
        if self.data_Bx.shape != self.data_By.shape:
            raise ValueError(f'Bx and By files have different shapes: Bx={self.data_Bx.shape}, By={self.data_By.shape}')

        data_Bx_time_values = self._read_time_values(self.filename_Bx)
        data_By_time_values = self._read_time_values(self.filename_By)
        if len(data_Bx_time_values) != len(data_By_time_values):
            raise ValueError('Bx and By contain different numbers of discharge times.')
        if not np.allclose(data_Bx_time_values, data_By_time_values, rtol=0.0, atol=1e-12):
            raise ValueError('Bx and By contain different discharge times.')

        self.data_available_time_values = data_Bx_time_values.copy()
        self.available_times = data_Bx_time_values * self.time_scale
        self.coordinate_count = self._coordinate_count(self.filename_Bx)
        if self.coordinate_count != self._coordinate_count(self.filename_By):
            raise ValueError('Bx and By use different coordinate layouts.')
        self.x_all, self.y_all = self.data_Bx[:, 0], self.data_Bx[:, 1]
        self.z_all = self.data_Bx[:, 2] if self.coordinate_count == 3 else None
        for index, axis in enumerate(('x', 'y', 'z')[:self.coordinate_count]):
            values = self.data_Bx[:, index]
            if not np.allclose(values, self.data_By[:, index], rtol=0.0, atol=1e-15):
                raise ValueError(f'Bx and By {axis} coordinates do not match.')
        expected_columns = self.coordinate_count + len(self.available_times)
        if self.data_Bx.shape[1] != expected_columns:
            raise ValueError(
                f'Unexpected number of columns in field map: expected {expected_columns}, '
                f'found {self.data_Bx.shape[1]}.')

    def _calculate_map_radius(self):
        valid_field = (np.any(np.isfinite(self.data_Bx[:, 3:]), axis=1)
                       | np.any(np.isfinite(self.data_By[:, 3:]), axis=1))
        if not np.any(valid_field):
            raise ValueError('No finite magnetic-field points were found in the imported map.')
        self.map_radius = float(np.max(np.hypot(self.x_all[valid_field], self.y_all[valid_field])))
        if not np.isfinite(self.map_radius) or self.map_radius <= 0:
            raise ValueError('Could not determine a positive radius from the imported field map.')

    def set_discharge_time(self, discharge_time):
        if not hasattr(self, 'data_Bx'):
            raise RuntimeError('Field maps are not loaded. Call load() first.')
        discharge_time = float(discharge_time)
        self.time_index = int(np.argmin(np.abs(self.available_times - discharge_time)))
        self.discharge_time = discharge_time
        self.selected_time = float(self.available_times[self.time_index])
        self.selected_time_raw = float(self.data_available_time_values[self.time_index])
        column = self.coordinate_count + self.time_index
        Bx, By = self.data_Bx[:, column], self.data_By[:, column]
        valid = np.isfinite(self.x_all) & np.isfinite(self.y_all) & np.isfinite(Bx) & np.isfinite(By)
        self.valid_mask = valid
        self.map_x, self.map_y = self.x_all[valid], self.y_all[valid]
        self.map_Bx, self.map_By = Bx[valid], By[valid]
        if len(self.map_x) < 3:
            raise ValueError(f'Fewer than three valid Bx/By points for discharge time {discharge_time}.')
        points = np.column_stack((self.map_x, self.map_y))
        try:
            if np.ptp(self.map_x) > 1e-15 and np.ptp(self.map_y) > 1e-15:
                self.interpolator_Bx = LinearNDInterpolator(points, self.map_Bx, fill_value=0.0)
                self.interpolator_By = LinearNDInterpolator(points, self.map_By, fill_value=0.0)
            else:
                # Some COMSOL exports contain a 1D line (one transverse
                # coordinate is constant), which cannot be triangulated in 2D.
                axis = self.map_x if np.ptp(self.map_x) > 1e-15 else self.map_y
                order = np.argsort(axis)
                axis = axis[order]
                bx, by = self.map_Bx[order], self.map_By[order]
                if np.any(np.diff(axis) <= 0):
                    raise ValueError('1D field-map coordinates must be unique.')
                bx_interp = interp1d(axis, bx, bounds_error=False, fill_value=0.0)
                by_interp = interp1d(axis, by, bounds_error=False, fill_value=0.0)
                coordinate = 0 if np.ptp(self.map_x) > 1e-15 else 1
                def line_interpolator(interp):
                    return lambda x, y: interp(x if coordinate == 0 else y)
                self.interpolator_Bx = line_interpolator(bx_interp)
                self.interpolator_By = line_interpolator(by_interp)
        except Exception as exc:
            raise ValueError('Could not construct field interpolators; check for duplicate or degenerate map coordinates.') from exc
        return self.selected_time

    def select_timestep(self, discharge_time):
        return self.set_discharge_time(discharge_time)

    def get_discharge_times(self):
        self._require_loaded()
        return self.available_times.copy()

    def get_discharge_times_raw(self):
        self._require_loaded()
        return self.data_available_time_values.copy()

    def get_field(self, discharge_time=None, x=0.0, y=0.0):
        self._require_loaded()
        selected_time = self.set_discharge_time(discharge_time) if discharge_time is not None else self.selected_time
        x_local = -(np.asarray(x) - self.offset_x)
        y_local = np.asarray(y) - self.offset_y
        Bx = np.nan_to_num(self.interpolator_Bx(x_local, y_local), nan=0.0)
        By = np.nan_to_num(self.interpolator_By(x_local, y_local), nan=0.0)
        Bx = self.imported_magnetic_field_polarity * Bx + self.external_magnetic_field_x
        By = By + self.external_magnetic_field_y
        return selected_time, Bx, By

    def get_field_map(self, discharge_time=None):
        self._require_loaded()
        if discharge_time is not None:
            self.set_discharge_time(discharge_time)
        return (self.imported_magnetic_field_polarity * self.map_x.copy(), self.map_y.copy(),
                self.imported_magnetic_field_polarity * self.map_Bx.copy() + self.external_magnetic_field_x,
                self.map_By.copy() + self.external_magnetic_field_y, self.selected_time)

    def plot_field_map(self, figsize=(15, 9), marker_size=8, cmap='coolwarm'):
        import matplotlib.pyplot as plt
        map_x, map_y, total_map_Bx, total_map_By, selected_time = self.get_field_map()
        imported_map_Bx = total_map_Bx - self.external_magnetic_field_x
        imported_map_By= total_map_By - self.external_magnetic_field_y
        fields = [(imported_map_Bx, 'Imported $B_x$ [T]'), (imported_map_By, 'Imported $B_y$ [T]'),
                  (np.hypot(imported_map_Bx, imported_map_By), 'Imported $|B|$ [T]'),
                  (total_map_Bx, 'Total $B_x$ [T]'), (total_map_By, 'Total $B_y$ [T]'),
                  (np.hypot(total_map_Bx, total_map_By), 'Total $|B|$ [T]')]
        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        for ax, (values, title) in zip(axes.flat, fields):
            points = ax.scatter(map_x * 1e6, map_y * 1e6, c=values, s=marker_size, cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel('x [µm]')
            ax.set_ylabel('y [µm]')
            ax.set_aspect('equal')
            fig.colorbar(points, ax=ax)
        fig.suptitle(f'Field map at t = {selected_time * 1e9:g} ns')
        plt.show()
        return fig, axes

    def get_map_radius(self):
        self._require_loaded()
        return self.map_radius

    def get_radius(self):
        self._require_loaded()
        return self.radius

    def get_field_magnitude(self, discharge_time=None, x=0.0, y=0.0):
        selected_time, Bx, By = self.get_field(discharge_time=discharge_time, x=x, y=y)
        return selected_time, np.hypot(Bx, By)

    def get_focusing_gradient(self):
        self._require_loaded()
        extent = max(np.ptp(self.map_x), np.ptp(self.map_y))
        if extent <= 0:
            return 0.0
        dx = min(1e-6, 0.001 * extent)
        _, bx0, by0 = self.get_field(x=0.0, y=0.0)
        _, bxx, byx = self.get_field(x=dx, y=0.0)
        _, bxy, byy = self.get_field(x=0.0, y=dx)
        dBy_dx = (float(np.asarray(byx)) - float(np.asarray(by0))) / dx
        dBx_dy = (float(np.asarray(bxy)) - float(np.asarray(bx0))) / dx
        return float(0.5 * (abs(dBy_dx) + abs(dBx_dy)))

    def get_relativistic_nonlinearity(self, discharge_time=None):
        self._require_loaded()
        if discharge_time is not None:
            self.set_discharge_time(discharge_time)
        x = self.imported_magnetic_field_polarity * self.map_x
        y = self.map_y
        Bx = self.imported_magnetic_field_polarity * self.map_Bx + self.external_magnetic_field_x
        By = self.map_By + self.external_magnetic_field_y
        r = np.hypot(x, y)
        mask = r > 1e-12
        if not np.any(mask):
            return 0.0
        B_over_r = np.hypot(Bx[mask], By[mask]) / r[mask]
        mean_value = np.mean(np.abs(B_over_r))
        return 0.0 if mean_value == 0 else float(np.std(B_over_r) / mean_value)

    def field_info(self):
        """Return field-map diagnostics without printing to the console."""
        self._require_loaded()
        total_map_Bx = self.imported_magnetic_field_polarity * self.map_Bx + self.external_magnetic_field_x
        total_map_By = self.map_By + self.external_magnetic_field_y
        return {
            'selected_time': self.selected_time,
            'raw_selected_time': self.selected_time_raw,
            'time_index': self.time_index,
            'number_of_points': len(self.map_x),
            'map_radius': self.map_radius,
            'tracking_radius': self.radius,
            'length': self.length,
            'num_slice': self.num_slice,
            'x_range': (float(self.map_x.min()), float(self.map_x.max())),
            'y_range': (float(self.map_y.min()), float(self.map_y.max())),
            'imported_magnetic_field_polarity': self.imported_magnetic_field_polarity,
            'external_magnetic_field_x': self.external_magnetic_field_x,
            'external_magnetic_field_y': self.external_magnetic_field_y,
            'Bx_range': (float(total_map_Bx.min()), float(total_map_Bx.max())),
            'By_range': (float(total_map_By.min()), float(total_map_By.max())),
            'focusing_gradient': self.get_focusing_gradient(),
            'field_nonlinearity': self.get_relativistic_nonlinearity(),
        }

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        self._require_loaded()
        slice_length = self.length / self.num_slice
        half_slice = slice_length / 2.0

        # Use the imported map radius as the tracking aperture.
        r2 = (beam.xs() - self.offset_x) ** 2 + (beam.ys() - self.offset_y) ** 2
        del beam[r2 > self.radius ** 2]
        if len(beam) == 0:
            return super().track(beam, savedepth, runnable, verbose)

        beam.transport(half_slice)
        for i in range(self.num_slice):
            r2 = (beam.xs() - self.offset_x) ** 2 + (beam.ys() - self.offset_y) ** 2
            del beam[r2 > self.radius ** 2]
            if len(beam) == 0:
                return super().track(beam, savedepth, runnable, verbose)

            x = -(beam.xs() - self.offset_x)
            y = beam.ys() - self.offset_y
            Bx = np.nan_to_num(self.interpolator_Bx(x, y), nan=0.0)
            By = np.nan_to_num(self.interpolator_By(x, y), nan=0.0)
            Bx = self.imported_magnetic_field_polarity * Bx + self.external_magnetic_field_x
            By = By + self.external_magnetic_field_y
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
