# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# License: GPL-3.0-or-later

"""ImpactX model for the CLEAR line downstream of the plasma lens."""

import numpy as np
import scipy.constants as SI
from abel.classes.spectrometer.spectrometer import Spectrometer
from abel.classes.plasma_lens.impl.plasma_lens_from_file import PlasmaLensFromFile
from pathlib import Path

#THESE VALUES ALL COME FROM THE MADX FILE ON THE CLEAR ACC MODELS REPOSITORY
_QUAD_LENGTH = 0.226
_CORRECTOR_LENGTH = 0.100
_BPM_LENGTH = 0.118
_ECQD_T_M_PER_A_M = 0.0572
_KJ_INTEGRATED_FIELD_T_M_PER_A = 0.248 * 0.0018
_SPECTROMETER_LENGTH = 1.7954
_SPECTROMETER_DIPOLE_LENGTH = 0.4713
_SPECTROMETER_DIPOLE_ANGLE_DEG = -22.75
_DUMP_BRANCH_LENGTH = 0.905
# Nominal currents for the CLEAR plasma-lens optics in plasmalens.str [A].
_NOMINAL_QUAD_CURRENTS = {"QDD0870": 0.0, "QFD0880": 0.0}
_CORRECTOR_NAMES = ("DHJ0840",)

class ImpactXSpectrometer(Spectrometer):
    """Common ImpactX tracking interface for spectrometer implementations."""
    def _run_lattice(self, lattice, beam, runnable=None, verbose=False, save_beams=True):
        from abel.wrappers.impactx.impactx_wrapper import run_impactx
        return run_impactx(
            lattice, beam, nom_energy=beam.energy(), runnable=runnable,
            keep_data=self.keep_data, save_beams=save_beams,
            space_charge=self.enable_space_charge, csr=self.enable_csr,
            isr=self.enable_isr, isr_on_ref_part=self.isr_on_ref_part,
            verbose=verbose,
        )
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        output, self.evolution = self._run_lattice(
            self.get_impactx_lattice(beam.energy()), beam, runnable, verbose,
            save_beams=self.use_monitors)
        return super().track(output, savedepth, runnable, verbose)
class SpectrometerCLEAR(ImpactXSpectrometer):
    """Track from the plasma-lens entrance through the CLEAR spectrometer."""
    def __init__(
        self,
        plasma_lens=None,
        discharge_time=80e-9,
        plasma_lens_length=20e-3,
        num_lens_slices=30,
        field_polarity=-1,
        external_bx=0.0,
        external_by=0.300,
        quad_currents=None,
        corrector_currents=None,
        num_slices=50,
        use_monitors=True,
        keep_data=False,
        enable_space_charge=False,
        enable_csr=False,
        enable_isr=False,
        isr_on_ref_part=True,
        aperture_diameter=None,
    ):
        super().__init__()
        self.num_slices = int(num_slices)
        self.num_lens_slices = int(num_lens_slices)
        self.use_monitors = bool(use_monitors)
        self.keep_data = bool(keep_data)
        self.enable_space_charge = bool(enable_space_charge)
        self.enable_csr = bool(enable_csr)
        self.enable_isr = bool(enable_isr)
        self.isr_on_ref_part = bool(isr_on_ref_part)
        self.aperture_diameter = (
            None if aperture_diameter is None else float(aperture_diameter))
        if self.aperture_diameter is not None and (
                not np.isfinite(self.aperture_diameter)
                or self.aperture_diameter <= 0):
            raise ValueError("aperture_diameter must be finite and positive.")
        if self.num_slices < 1 or self.num_lens_slices < 1:
            raise ValueError("num_slices and num_lens_slices must be positive.")
        if plasma_lens is None:
            map_dir = Path(__file__).resolve().parents[2] / "plasma_lens" / "impl" / "Argon"
            plasma_lens = PlasmaLensFromFile(
                filename_Bx=str(map_dir / "Bx_2D.txt"),
                filename_By=str(map_dir / "By_2D.txt"),
                discharge_time=discharge_time, length=plasma_lens_length,
                num_slice=num_lens_slices,
                imported_magnetic_field_polarity=field_polarity,
                external_magnetic_field_x=external_bx,
                external_magnetic_field_y=external_by,
            ).load()
        self.plasma_lens = plasma_lens
        self.quad_currents = dict(_NOMINAL_QUAD_CURRENTS)
        if quad_currents is not None:
            self.quad_currents.update(quad_currents)
        self.corrector_currents = {name: (0.0, 0.0) for name in _CORRECTOR_NAMES}
        if corrector_currents is not None:
            self.corrector_currents.update(corrector_currents)
    def get_length(self):
        """Return the modeled length from lens entrance to BPM0890 [m]."""
        return float(self.plasma_lens.length) / 2 + 0.292 + 2.04
    @staticmethod
    def _push_lens_reference(field_map, refpart, ds):
        p_total = np.sqrt(refpart.pt**2 - 1.0)
        step = 0.5 * ds / p_total
        refpart.x += step * refpart.px
        refpart.y += step * refpart.py
        refpart.z += step * refpart.pz
        refpart.t -= step * refpart.pt
        refpart.s += 0.5 * ds
        _, bx, by = field_map.get_field(x=refpart.x, y=refpart.y)
        q_over_mc = -SI.e / (SI.m_e * SI.c)
        refpart.px += q_over_mc * float(by) * ds
        refpart.py -= q_over_mc * float(bx) * ds
        pz2 = p_total**2 - refpart.px**2 - refpart.py**2
        if pz2 <= 0:
            raise ValueError("Reference particle lost forward momentum in the plasma lens.")
        refpart.pz = np.sqrt(pz2)
        refpart.x += step * refpart.px
        refpart.y += step * refpart.py
        refpart.z += step * refpart.pz
        refpart.t -= step * refpart.pt
        refpart.s += 0.5 * ds
    @staticmethod
    def _push_lens_particles(field_map, lens, pti, refpart):
        soa = pti.soa().to_xp()
        x, y = soa.real["position_x"], soa.real["position_y"]
        px, py = soa.real["momentum_x"], soa.real["momentum_y"]
        ds = lens.ds / lens.nslice
        x[:] += 0.5 * ds * px
        y[:] += 0.5 * ds * py
        _, bx, by = field_map.get_field(x=x, y=y)
        px[:] -= np.asarray(by) * ds / refpart.rigidity_Tm
        py[:] += np.asarray(bx) * ds / refpart.rigidity_Tm
        x[:] += 0.5 * ds * px
        y[:] += 0.5 * ds * py
    def _add_radial_aperture_checks(self, lattice):
        """Add a circular aperture check before and after each lattice element."""
        if self.aperture_diameter is None:
            return lattice
        from impactx import elements
        radius = self.aperture_diameter / 2
        def aperture():
            return elements.Aperture(
                aperture_x=radius, aperture_y=radius, shape="elliptical")
        checked = [aperture()]
        for element in lattice:
            checked.extend((element, aperture()))
        return checked
    def _quad_strengths(self, kinetic_energy_ev):
        """Convert downstream CLEAR quadrupole currents to ImpactX k1."""
        energy_gev = kinetic_energy_ev / SI.e / 1e9
        mass_gev = SI.m_e * SI.c**2 / SI.e / 1e9
        momentum_gev = np.sqrt(energy_gev * (energy_gev + 2 * mass_gev))
        brho = momentum_gev / 0.299792458
        gradients = {name: current * _ECQD_T_M_PER_A_M
                     for name, current in self.quad_currents.items()}
        return {
            "QDD0870": -gradients["QDD0870"] / brho,
            "QFD0880": +gradients["QFD0880"] / brho,
        }, brho
    def get_impactx_lattice(self, kinetic_energy_ev):
        """Build the CLEAR lattice from the lens entrance to BPM0890."""
        from impactx import elements
        strengths, brho = self._quad_strengths(kinetic_energy_ev)
        ns = self.num_slices
        lens_length = float(self.plasma_lens.length)
        btv810 = lens_length / 2 + 0.292
        bpm820 = btv810 + 0.362 - _BPM_LENGTH / 2
        corrector840 = btv810 + 0.362 + 0.540 - _CORRECTOR_LENGTH / 2 + 0.100
        q870 = btv810 + 1.390 + 1.365 - 1.465
        q880 = q870 + 0.450
        bpm890 = q880 + 0.200 - _BPM_LENGTH / 2 + 0.100
        items = [(0.0, 0.0, 0, lambda: elements.BeamMonitor(
            name="BTV805", backend="h5", encoding="g"))]
        lens = elements.Programmable(name="PlasmaLensFromFile", ds=lens_length,
                                     nslice=self.num_lens_slices)
        lens.beam_particles = lambda pti, refpart: self._push_lens_particles(
            self.plasma_lens, lens, pti, refpart)
        lens.ref_particle = lambda refpart: self._push_lens_reference(
            self.plasma_lens, refpart, lens.ds / lens.nslice)
        items.append((lens_length / 2, lens_length, 2, lambda: lens))
        def add_monitor(name, center):
            length = _BPM_LENGTH if name.startswith("BPM") else 0.0
            def make_monitor():
                monitor = elements.BeamMonitor(name=name, backend="h5", encoding="g")
                if length == 0:
                    return monitor
                return [elements.Drift(ds=length / 2, nslice=ns), monitor,
                        elements.Drift(ds=length / 2, nslice=ns)]
            items.append((center, length, 1, make_monitor))
        for name, center in (("QDD0870", q870), ("QFD0880", q880)):
            items.append((center, _QUAD_LENGTH, 2,
                          lambda name=name: elements.ChrQuad(
                              name=name, ds=_QUAD_LENGTH, k=strengths[name], nslice=ns)))
        hkick, vkick = self.corrector_currents["DHJ0840"]
        items.append((corrector840, _CORRECTOR_LENGTH, 2, lambda: [
            elements.Drift(ds=_CORRECTOR_LENGTH / 2, nslice=ns),
            elements.Kicker(name="DHJ0840", xkick=float(hkick) * _KJ_INTEGRATED_FIELD_T_M_PER_A / brho,
                            ykick=float(vkick) * _KJ_INTEGRATED_FIELD_T_M_PER_A / brho,
                            unit="dimensionless"),
            elements.Drift(ds=_CORRECTOR_LENGTH / 2, nslice=ns),
        ]))
        items.append((btv810, 0.0, 0, lambda: elements.BeamMonitor(
            name="BTV0810", backend="h5", encoding="g")))
        for name, center in (("BPM0820", bpm820), ("BPM0890", bpm890)):
            if self.use_monitors:
                add_monitor(name, center)
        items.sort(key=lambda item: (item[0] - item[1] / 2, item[2]))
        lattice, position = [], 0.0
        for center, length, _, create in items:
            start = center - length / 2
            if start > position:
                lattice.append(elements.Drift(ds=start - position, nslice=ns))
            elif start < position - 1e-10:
                raise ValueError(f"Overlapping CLEAR elements near s={start:.6f} m.")
            element = create()
            lattice.extend(element if isinstance(element, list) else [element])
            position = center + length / 2
        if position < self.get_length():
            lattice.append(elements.Drift(ds=self.get_length() - position, nslice=ns))
        return self._add_radial_aperture_checks(lattice)
    @staticmethod
    def _truncate_lattice_at_monitor(lattice, monitor_name):
        for index, element in enumerate(lattice):
            if getattr(element, "name", None) == monitor_name:
                return lattice[:index + 1]
        raise ValueError(f"Monitor {monitor_name} is not in the lattice.")
    def get_impactx_lattice_to_screen(
            self, kinetic_energy_ev, screen="BTV910"):
        """Build the entrance-to-screen lattice for a selected branch.
        ``screen`` is ``"BTV910"`` for the straight dump (spectrometer off),
        or ``"BTV930"`` for the bent spectrometer branch.
        """
        from impactx import elements
        lattice = self.get_impactx_lattice(kinetic_energy_ev)
        if screen == "BTV910":
            lattice.extend(self._add_radial_aperture_checks([
                elements.ExactDrift(ds=_DUMP_BRANCH_LENGTH, nslice=self.num_slices,
                                    name="CA_INAIR_to_BTV910"),
                elements.BeamMonitor(name="BTV910", backend="h5", encoding="g"),
            ]))
            return lattice
        if screen != "BTV930":
            raise ValueError("screen must be 'BTV910' or 'BTV930'.")
        _, brho = self._quad_strengths(kinetic_energy_ev)
        dipole_length = _SPECTROMETER_DIPOLE_LENGTH
        theta_deg = _SPECTROMETER_DIPOLE_ANGLE_DEG
        theta = np.deg2rad(theta_deg)
        dipole_arc_length = theta * (0.5 * dipole_length) / np.sin(theta / 2)
        dipole_center = (34.899994842 - 13.5894 - 21.0) + dipole_arc_length / 2
        dipole_start = dipole_center - dipole_length / 2
        dipole_end = dipole_center + dipole_length / 2
        bpm0920_center = dipole_center + 0.7436
        bpm0920_length = _BPM_LENGTH
        lattice.extend(self._add_radial_aperture_checks([
            elements.ExactDrift(ds=dipole_start, nslice=self.num_slices),
            elements.ExactSbend(
                name="BHB0900", ds=dipole_length, phi=theta_deg,
                B=theta * brho / dipole_length, nslice=self.num_slices,
            ),
            elements.ExactDrift(
                ds=bpm0920_center - bpm0920_length / 2 - dipole_end,
                nslice=self.num_slices),
            elements.ExactDrift(ds=bpm0920_length / 2, nslice=self.num_slices),
            elements.BeamMonitor(name="BPM0920", backend="h5", encoding="g"),
            elements.ExactDrift(ds=bpm0920_length / 2, nslice=self.num_slices),
            elements.ExactDrift(
                ds=_SPECTROMETER_LENGTH - bpm0920_center - bpm0920_length / 2,
                nslice=self.num_slices),
            elements.BeamMonitor(name="BTV0930", backend="h5", encoding="g"),
        ]))
        return lattice
    def track_from_plasma_lens_entrance(self, beam, spectrometer_on=True, verbose=False,
                                        runnable=None):
        """Track a beam from BTV805 through the lens to a screen.
        ``spectrometer_on`` selects BTV930 or the straight BTV910 branch.
        """
        screen = "BTV930" if spectrometer_on else "BTV910"
        target_names = {"BTV0810", "BPM0820", "BPM0890", "BPM0920", screen}
        lattice = self.get_impactx_lattice_to_screen(
            beam.energy(), screen)
        targets = [
            getattr(element, "name") for element in lattice
            if getattr(element, "name", None) in target_names
        ]
        beams = {"BTV805 (lens entrance)": beam}
        for monitor_name in targets:
            monitor_lattice = self.get_impactx_lattice_to_screen(
                beam.energy(), screen)
            lattice_to_monitor = self._truncate_lattice_at_monitor(
                monitor_lattice, monitor_name)
            monitor_beam, _ = self._run_lattice(
                lattice_to_monitor, beam, runnable, verbose, save_beams=False)
            label = "BTV810" if monitor_name == "BTV0810" else monitor_name
            if monitor_name == screen:
                label = f"{screen} (spectrometer {'ON' if spectrometer_on else 'OFF'})"
            beams[label] = monitor_beam
        return beams
    def plot_lattice_survey(self, kinetic_energy_ev):
        """Plot the common line and both screen branches with CLEAR element colors."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        lattice = self.get_impactx_lattice(kinetic_energy_ev)
        items, common_end = [], 0.0
        for element in lattice:
            length = float(getattr(element, "ds", 0.0))
            name = getattr(element, "name", "") or ""
            center = common_end + length / 2
            if name.startswith(("Q", "DHJ")):
                items.append(("magnet", name, center))
            elif name.startswith(("BTV", "BPM")):
                items.append(("screen", name, center))
            elif name.startswith("PlasmaLens"):
                items.append(("lens", name, center, length))
            common_end += length
        fig, ax = plt.subplots(figsize=(17, 8), layout="constrained")
        ax.plot([0, common_end], [0, 0], color="black", lw=2.2,
                label="Common line from BTV805")
        lens = next((item for item in items if item[0] == "lens"), None)
        if lens:
            _, _, center, length = lens
            ax.plot([center - length / 2, center + length / 2], [0, 0],
                    color="tab:cyan", lw=7, solid_capstyle="butt", zorder=3)
            ax.annotate("Plasma lens", (center, 0), xytext=(0, 7),
                        textcoords="offset points", ha="center", color="teal", fontsize=8)
        magnets = [(name, pos) for kind, name, pos, *rest in items if kind == "magnet"]
        for i, (name, pos) in enumerate(magnets):
            quad = name.startswith("Q")
            color, marker = (("tab:purple", "^") if quad else ("tab:orange", "D"))
            ax.scatter(pos, 0, color=color, marker=marker, s=36, zorder=4)
            height = 0.12 + 0.035 * (i % 3)
            ax.vlines(pos, 0.015, height, color=color, lw=0.8)
            ax.annotate(name, (pos, height), xytext=(0, 2), textcoords="offset points",
                        rotation=55, ha="left", va="bottom", fontsize=7, color=color)
        screens = [(name, pos) for kind, name, pos, *rest in items if kind == "screen"]
        for i, (name, pos) in enumerate(screens):
            bpm = name.startswith("BPM")
            color = "tab:blue" if bpm else "tab:red"
            ax.scatter(pos, 0, color=color, marker="s", s=24, zorder=5)
            height = -0.12 - 0.035 * (i % 3)
            ax.vlines(pos, -0.015, height, color=color, lw=0.7)
            label = f"{name}\ns={pos:.2f} m" if bpm else name
            ax.annotate(label, (pos, height), xytext=(0, -2), textcoords="offset points",
                        rotation=55, ha="right", va="top", fontsize=7, color=color)
        dump_end = common_end + _DUMP_BRANCH_LENGTH
        ax.plot([common_end, dump_end], [0, 0], color="tab:green", lw=2,
                label="CA.INAIR branch")
        ax.scatter(dump_end, 0, marker="s", color="tab:red", s=28, zorder=5)
        ax.annotate("BTV910", (dump_end, 0), xytext=(0, -18),
                    textcoords="offset points", ha="center", fontsize=8)
        theta = np.deg2rad(_SPECTROMETER_DIPOLE_ANGLE_DEG)
        dipole_length = _SPECTROMETER_DIPOLE_LENGTH
        arc_length = theta * (0.5 * dipole_length) / np.sin(theta / 2)
        dipole_center = 34.899994842 - 13.5894 - 21.0 + arc_length / 2
        dipole_start = dipole_center - dipole_length / 2
        dipole_start_x = common_end + dipole_start
        radius = dipole_length / theta
        arc_t = np.linspace(0, theta, 80)
        arc_x = dipole_start_x + radius * np.sin(arc_t)
        arc_y = radius * (1 - np.cos(arc_t))
        remaining = _SPECTROMETER_LENGTH - dipole_start - dipole_length
        end_x = arc_x[-1] + remaining * np.cos(theta)
        end_y = arc_y[-1] + remaining * np.sin(theta)
        ax.plot([common_end, dipole_start_x], [0, 0], color="tab:blue", lw=2,
                label="CA.SPECT branch")
        ax.plot(arc_x, arc_y, color="tab:blue", lw=3)
        ax.annotate("BHB0900", (np.mean(arc_x), np.mean(arc_y)), xytext=(0, 10),
                    textcoords="offset points", ha="center", color="tab:orange", fontsize=8)
        ax.plot([arc_x[-1], end_x], [arc_y[-1], end_y], color="tab:blue", lw=2)
        bpm_s = dipole_start + dipole_length + (0.7436 - dipole_length / 2)
        bpm_x = arc_x[-1] + (bpm_s - dipole_start - dipole_length) * np.cos(theta)
        bpm_y = arc_y[-1] + (bpm_s - dipole_start - dipole_length) * np.sin(theta)
        ax.scatter(bpm_x, bpm_y, marker="s", color="tab:blue", s=25, zorder=5)
        ax.annotate(f"BPM0920\ns={common_end + bpm_s:.2f} m", (bpm_x, bpm_y),
                    xytext=(0, -18), textcoords="offset points", ha="center", fontsize=8)
        ax.scatter(end_x, end_y, marker="s", color="tab:red", s=28, zorder=5)
        ax.annotate("BTV930", (end_x, end_y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
        legend = [
            Line2D([0], [0], marker="^", color="none", markerfacecolor="tab:purple", label="Quadrupole"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="tab:orange", label="Corrector"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="tab:blue", label="BPM"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="tab:red", label="BTV / screen"),
            Line2D([0], [0], color="tab:cyan", lw=7, label="Plasma lens"),
            Line2D([0], [0], color="black", lw=2, label="Common line"),
            Line2D([0], [0], color="tab:green", lw=2, label="CA.INAIR branch"),
            Line2D([0], [0], color="tab:blue", lw=2, label="CA.SPECT branch"),
        ]
        ax.set(xlabel="survey x [m]", ylabel="survey y [m]",
               title="CLEAR sublattice survey: BTV805 to BTV910 / BTV930",
               ylim=(-0.55, 0.55))
        ax.grid(True, alpha=0.25)
        ax.legend(handles=legend, loc="upper left", ncol=2, fontsize=8)
        return fig, ax

