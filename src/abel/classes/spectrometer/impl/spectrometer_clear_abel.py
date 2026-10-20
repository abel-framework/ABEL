# This file is part of ABEL
# Copyright 2025, The ABEL Authors
# License: GPL-3.0-or-later

"""ImpactX first implementation of the CLEAR straight plasma-lens section.
The element layout follows ``CA.STRAIGHT`` in the CERN acc-models-clear
``clear.seqx`` sequence, from BTV0390 to BPM0890. Nominal quadrupole currents
are taken from ``scenarios/plasmalens/plasmalens.str``. PLC0800M is replaced
by ABEL's imported magnetic-field lens.
"""

from pathlib import Path
import numpy as np
import scipy.constants as SI
from abel.classes.spectrometer.spectrometer import Spectrometer
from abel.classes.plasma_lens.impl.plasma_lens_from_file import PlasmaLensFromFile

#THESE VALUES ALL COME FROM THE MADX FILE ON THE CLEAR ACC MODELS REPOSITORY
_LATTICE_LENGTH = (
    1.4664 + 0.500 + 0.500 + 2.468 + 3.820 + 0.530 + 0.450 + 0.450
    + 1.365 + 1.390 + 0.450 + 0.200
)
_QUAD_LENGTH = 0.226
_QUAD_MECHANICAL_LENGTH = 0.200
_CORRECTOR_LENGTH = 0.100
_BPM_LENGTH = 0.118
_ECQD_T_M_PER_A_M = 0.0572
_KJ_INTEGRATED_FIELD_T_M_PER_A = 0.248 * 0.0018
_UPSTREAM_QUAD_CURRENTS = {
    "QFD0350": 0.09788539711,
    "QDD0355": 1.897039703,
    "QFD0360": 0.4342097268,
}
_SPECTROMETER_LENGTH = 1.7954
_SPECTROMETER_DIPOLE_LENGTH = 0.4713
_SPECTROMETER_DIPOLE_ANGLE_DEG = -22.75
_DUMP_BRANCH_LENGTH = 0.905

# Nominal currents for the CLEAR plasma-lens optics in plasmalens.str [A].
_NOMINAL_QUAD_CURRENTS = {
    "QFD0510": 4.64082096,
    "QDD0515": 47.97442317,
    "QFD0520": 50.64090629,
    "QFD0760": 76.55068446,
    "QDD0765": 115.5896332,
    "QFD0770": 57.21069551,
    "QDD0870": 0.0,
    "QFD0880": 0.0,
}
_CORRECTOR_NAMES = ("DHJ0540", "DHJ0590", "DHJ0710", "DHJ0780", "DHJ0840")


def _push_lens_reference(field_map, refpart, ds):
    """Push ImpactX's reference particle through one imported-field slice."""
    p_total = np.sqrt(refpart.pt**2 - 1.0)

    def drift(step_s):
        step = step_s / p_total
        refpart.x += step * refpart.px
        refpart.y += step * refpart.py
        refpart.z += step * refpart.pz
        refpart.t -= step * refpart.pt
        refpart.s += step_s

    drift(0.5 * ds)
    _, bx, by = field_map.get_field(x=refpart.x, y=refpart.y)
    q_over_mc = -SI.e / (SI.m_e * SI.c)
    refpart.px += q_over_mc * float(by) * ds
    refpart.py -= q_over_mc * float(bx) * ds
    pz2 = p_total**2 - refpart.px**2 - refpart.py**2
    if pz2 <= 0:
        raise ValueError("Reference particle lost forward momentum in the imported plasma lens.")
    refpart.pz = np.sqrt(pz2)
    drift(0.5 * ds)


def _push_lens_particles(field_map, lens, pti, refpart):
    """Apply one paraxial drift/kick/drift slice to an ImpactX particle tile."""
    soa = pti.soa().to_xp()
    x, y = soa.real["position_x"], soa.real["position_y"]
    px, py = soa.real["momentum_x"], soa.real["momentum_y"]
    ds = lens.ds / lens.nslice

    x[:] += 0.5 * ds * px
    y[:] += 0.5 * ds * py
    _, bx, by = field_map.get_field(x=x, y=y)
    brho = refpart.rigidity_Tm
    px[:] -= np.asarray(by) * ds / brho
    py[:] += np.asarray(bx) * ds / brho
    x[:] += 0.5 * ds * px
    y[:] += 0.5 * ds * py


class SpectrometerCLEARAbel(Spectrometer):
    """Track through the CLEAR CA.STRAIGHT lattice with a map-based lens.

    Parameters
    ----------
    plasma_lens : PlasmaLensFromFile here defaults.
        A configured and loaded field-map lens. If omitted, the Argon maps
        loaded using the settings below.
    discharge_time : 
        Requested map discharge time in seconds (defaults to 80 ns).
    plasma_lens_length : 
        Physical map-lens length in metres (defaulst to 20 mm).
    num_lens_slices : int
        Number of integration slices through the map lens.
    field_polarity : int
        Imported map polarity, either -1 or +1.
    external_bx, external_by : 
        Uniform external field added to the imported map, in tesla.
    aperture_diameter : 
        Optional circular CLEAR beam-pipe aperture diameter in metres. When
        configured, aperture checks are placed at each lattice element bdry.
        Apertures disabled by default.
    quad_currents : dict, optional
        Quadrupole supply currents [A], keyed by MAD-X element name. Defaults
        to the nominal currents in ``scenarios/plasmalens/plasmalens.str`` in the acc models clear repository.
    corrector_currents : dict, optional
        Horizontal and vertical currents [A], keyed by corrector name and
        supplied as ``(horizontal, vertical)`` pairs. All defaulting to zero.
    num_slices : int
        ImpactX space-charge slices per quadrupole and drift.
    use_monitors : bool
        Includes all BPM and BTV diagnostics in the sequence.
        """

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
        self.num_lens_slices = int(num_lens_slices)
        self.num_slices = int(num_slices)
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

        if self.num_lens_slices < 1 or self.num_slices < 1:
            raise ValueError("num_lens_slices and num_slices must be positive.")

        if plasma_lens is None:
            map_dir = Path(__file__).resolve().parents[2] / "plasma_lens" / "impl" / "Argon"
            plasma_lens = PlasmaLensFromFile(
                filename_Bx=str(map_dir / "Bx_2D.txt"),
                filename_By=str(map_dir / "By_2D.txt"),
                discharge_time=discharge_time,
                length=plasma_lens_length,
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
        """Return the MAD-X CA.STRAIGHT length from BTV0390 to BPM0890 [m]."""
        return _LATTICE_LENGTH

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

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        """Track the input beam through CA.STRAIGHT with ImpactX."""
        from abel.wrappers.impactx.impactx_wrapper import run_impactx

        lattice = self.get_impactx_lattice(beam.energy())
        output, self.evolution = run_impactx(
            lattice,
            beam,
            nom_energy=beam.energy(),
            runnable=runnable,
            keep_data=self.keep_data,
            save_beams=self.use_monitors,
            space_charge=self.enable_space_charge,
            csr=self.enable_csr,
            isr=self.enable_isr,
            isr_on_ref_part=self.isr_on_ref_part,
            verbose=verbose,
        )
        return super().track(output, savedepth, runnable, verbose)

    def _quad_strengths(self, kinetic_energy_ev):
        """Convert MAD-X nominal supply currents to ImpactX k1 [1/m^2]."""
        energy_gev = kinetic_energy_ev / SI.e / 1e9
        mass_gev = SI.m_e * SI.c**2 / SI.e / 1e9
        momentum_gev = np.sqrt(energy_gev * (energy_gev + 2 * mass_gev))
        brho = momentum_gev / 0.299792458
        gradients = {name: current * _ECQD_T_M_PER_A_M
                     for name, current in self.quad_currents.items()}
        return {
            "QFD0510": +gradients["QFD0510"] / brho,
            "QDD0515": -gradients["QDD0515"] / brho,
            "QFD0520": +gradients["QFD0520"] / brho,
            "QFD0760": +gradients["QFD0760"] / brho,
            "QDD0765": -gradients["QDD0765"] / brho,
            "QFD0770": +gradients["QFD0770"] / brho,
            "QDD0870": -gradients["QDD0870"] / brho,
            "QFD0880": +gradients["QFD0880"] / brho,
        }, brho

    def get_impactx_lattice(self, kinetic_energy_ev):
        """Build the ImpactX element list for the CLEAR straight section.

        Element centre positions follow the AT placements in the MAD-X
        CA.STRAIGHT sequence. Finite-length quadrupoles and correctors are
        represented at those positions.
        """
        from impactx import elements

        strengths, brho = self._quad_strengths(kinetic_energy_ev)
        ns = self.num_slices
        items = []

        def add_monitor(name, position):
            if self.use_monitors:
                length = _BPM_LENGTH if name.startswith("BPM") else 0.0

                def make_monitor(name=name, length=length):
                    monitor = elements.BeamMonitor(name=name, backend="h5", encoding="g")
                    if length == 0:
                        return monitor
                    return [
                        elements.Drift(ds=length / 2, nslice=ns, name=f"{name}_upstream"),
                        monitor,
                        elements.Drift(ds=length / 2, nslice=ns, name=f"{name}_downstream"),
                    ]

                items.append((position, length, 1, make_monitor))

        def add_quad(name, position):
            items.append((position, _QUAD_LENGTH, 2, lambda name=name: elements.ChrQuad(
                name=name, ds=_QUAD_LENGTH, k=strengths[name], nslice=ns)))

        def add_corrector(name, position):
            hkick, vkick = self.corrector_currents[name]
            # MAD-X DHJ calibration: integrated field / rigidity [rad/A].
            xkick = float(hkick) * _KJ_INTEGRATED_FIELD_T_M_PER_A / brho
            ykick = float(vkick) * _KJ_INTEGRATED_FIELD_T_M_PER_A / brho
            def make_corrector(name=name, xkick=xkick, ykick=ykick):
                return [
                    elements.Drift(ds=_CORRECTOR_LENGTH / 2, nslice=ns,
                                   name=f"{name}_upstream"),
                    elements.Kicker(xkick=xkick, ykick=ykick,
                                    unit="dimensionless", name=name),
                    elements.Drift(ds=_CORRECTOR_LENGTH / 2, nslice=ns,
                                   name=f"{name}_downstream"),
                ]

            items.append((position, _CORRECTOR_LENGTH, 2, make_corrector))

        # Absolute centres are reconstructed from the CA.STRAIGHT AT/FROM
        # expressions in clear.seqx; origin is the BTV0390 exit.
        q510 = 1.4664 - _QUAD_MECHANICAL_LENGTH / 2
        q515 = q510 + 0.500
        q520 = q515 + 0.500
        bpm530 = q520 + 0.372 + _QUAD_MECHANICAL_LENGTH / 2 - _BPM_LENGTH / 2
        dhj540 = bpm530 + 0.200 + _BPM_LENGTH / 2 - _CORRECTOR_LENGTH / 2
        dhj590 = dhj540 + 1.227
        bpm595 = dhj590 + 0.253 + _CORRECTOR_LENGTH / 2 - _BPM_LENGTH / 2
        btv620 = q520 + 2.468 + _QUAD_MECHANICAL_LENGTH / 2
        bpm690 = btv620 + 2.874 - _BPM_LENGTH / 2
        dhj710 = bpm690 + 0.564 + _BPM_LENGTH / 2 - _CORRECTOR_LENGTH / 2
        btv730 = btv620 + 3.820
        q760 = btv730 + 0.530 - _QUAD_MECHANICAL_LENGTH / 2
        q765 = q760 + 0.450
        q770 = q765 + 0.450
        dhj780 = q770 + 0.533 + _QUAD_MECHANICAL_LENGTH / 2 - _CORRECTOR_LENGTH / 2
        lens_center = q770 + 1.365 + _QUAD_MECHANICAL_LENGTH / 2 - 0.292
        btv805 = lens_center - float(self.plasma_lens.length) / 2
        btv810 = q770 + 1.365 + _QUAD_MECHANICAL_LENGTH / 2
        bpm820 = btv810 + 0.362 - _BPM_LENGTH / 2
        dhj840 = btv810 + 0.362 + 0.540 - _CORRECTOR_LENGTH / 2 + _QUAD_MECHANICAL_LENGTH / 2
        q870 = q770 + 1.390 + 1.365
        q880 = q870 + 0.450
        bpm890 = q880 + 0.200 - _BPM_LENGTH / 2 + _QUAD_MECHANICAL_LENGTH / 2

        for name, position in (("QFD0510", q510), ("QDD0515", q515),
                               ("QFD0520", q520), ("QFD0760", q760),
                               ("QDD0765", q765), ("QFD0770", q770),
                               ("QDD0870", q870), ("QFD0880", q880)):
            add_quad(name, position)
        for name, position in (("DHJ0540", dhj540), ("DHJ0590", dhj590),
                               ("DHJ0710", dhj710), ("DHJ0780", dhj780),
                               ("DHJ0840", dhj840)):
            add_corrector(name, position)

        # All installed BPMs and screens in CA.STRAIGHT.
        for name, position in (("BPM0530", bpm530), ("BPM0595", bpm595),
                               ("BPM0690", bpm690), ("BPM0820", bpm820),
                               ("BPM0890", bpm890), ("BTV0620", btv620),
                               ("BTV0730", btv730), ("BTV0800", lens_center),
                               ("BTV805", btv805),
                               ("BTV0810", btv810)):
            add_monitor(name, position)

        # The MAD-X lens marker is at the centre of BTV0800. Splitting the
        # field lens there keeps that screen at the same physical position.
        lens_length = float(self.plasma_lens.length)
        if lens_length <= 0 or lens_length >= 2 * (btv810 - lens_center):
            raise ValueError("Plasma-lens length must be positive and fit before BTV0810.")
        for half in (-1, +1):
            ds = lens_length / 2
            lens = elements.Programmable(
                name=f"PlasmaLensFromFile_{'upstream' if half < 0 else 'downstream'}",
                ds=ds,
                nslice=max(1, int(np.ceil(self.num_lens_slices / 2))),
            )
            lens.beam_particles = lambda pti, refpart, lens=lens: _push_lens_particles(
                self.plasma_lens, lens, pti, refpart)
            lens.ref_particle = lambda refpart, ds=ds: _push_lens_reference(
                self.plasma_lens, refpart, ds / lens.nslice)
            items.append((lens_center + half * lens_length / 4, ds, 2, lambda lens=lens: lens))

        # Monitors at a finite element boundary can differ from that boundary
        # by a few floating-point ulps (for example BTV805 at the lens entrance).
        # Round the sort position so the monitor priority resolves the tie
        # before the finite-length element is appended.
        items.sort(key=lambda item: (round(item[0] - item[1] / 2, 12),
                                     item[2], item[0]))
        lattice = []
        s = 0.0
        for center, length, _, create_element in items:
            start = center - length / 2
            if start < s - 1e-10:
                raise ValueError(f"Overlapping CLEAR elements near s={start:.6f} m.")
            if start > s + 1e-12:
                lattice.append(elements.Drift(ds=start - s, nslice=ns,
                                              name=f"CLEAR_drift_{len(lattice):03d}"))
                s = start
            element = create_element()
            if isinstance(element, list):
                lattice.extend(element)
            else:
                lattice.append(element)
            s = center + length / 2

        if s > _LATTICE_LENGTH + 1e-8:
            raise ValueError("CLEAR element positions exceed the MAD-X section length.")
        if s < _LATTICE_LENGTH - 1e-12:
            lattice.append(elements.Drift(ds=_LATTICE_LENGTH - s, nslice=ns,
                                          name="CLEAR_final_drift"))
        return self._add_radial_aperture_checks(lattice)

    def get_impactx_lattice_from_monitor(self, kinetic_energy_ev, monitor_name):
        """

        The returned lattice starts with the selected monitor, followed by
        every downstream element through to BPM0890. It is useful when the input
        beam has already been tracked to a location, for example BTV805 at
        the plasma-lens entrance during measurements.
        """
        lattice = self.get_impactx_lattice(kinetic_energy_ev)
        for index, element in enumerate(lattice):
            if getattr(element, "name", None) == monitor_name:
                return lattice[index:]
        available = [getattr(element, "name", None) or "<unnamed>" for element in lattice
                     if (getattr(element, "name", None) or "").startswith(("BTV", "BPM"))]
        raise ValueError(
            f"Monitor {monitor_name} is not present in the CLEAR lattice. "
            f"Available monitors: {available}"
        )

    @staticmethod
    def _truncate_lattice_at_monitor(lattice, monitor_name):
        """Return a fresh-lattice prefix ending at the requested monitor."""
        for index, element in enumerate(lattice):
            if getattr(element, "name", None) == monitor_name:
                return lattice[:index + 1]
        available = [getattr(element, "name", None) or "<unnamed>" for element in lattice
                     if (getattr(element, "name", None) or "").startswith(("BTV", "BPM"))]
        raise ValueError(
            f"Monitor {monitor_name} is not present in the requested lattice. "
            f"Available monitors: {available}"
        )

    def _get_qfd0350_to_ca_straight_lattice(self, kinetic_energy_ev):
        """Build the upstream model from QFD0350 entrance to CA.STRAIGHT."""
        from impactx import elements

        energy_gev = kinetic_energy_ev / SI.e / 1e9
        mass_gev = SI.m_e * SI.c**2 / SI.e / 1e9
        momentum_gev = np.sqrt(energy_gev * (energy_gev + 2 * mass_gev))
        brho = momentum_gev / 0.299792458
        gradients = {name: current * _ECQD_T_M_PER_A_M
                     for name, current in _UPSTREAM_QUAD_CURRENTS.items()}
        k350 = +gradients["QFD0350"] / brho
        k355 = -gradients["QDD0355"] / brho
        k360 = +gradients["QFD0360"] / brho
        ns = self.num_slices

        lattice = [
            elements.ChrQuad(name="QFD0350", ds=0.226, k=k350, nslice=ns),
            elements.Drift(ds=0.214, nslice=ns, name="drift_350_355"),
            elements.ChrQuad(name="QDD0355", ds=0.226, k=k355, nslice=ns),
            elements.Drift(ds=0.214, nslice=ns, name="drift_355_360"),
            elements.ChrQuad(name="QFD0360", ds=0.226, k=k360, nslice=ns),
            # DHG0385 is 0.703 m after QFD0360's centre; its nominal current is 0 A.
            elements.Drift(ds=0.5175, nslice=ns, name="drift_to_DHG0385"),
            elements.Drift(ds=0.0725, nslice=ns, name="DHG0385_upstream"),
            elements.Kicker(name="DHG0385", xkick=0.0, ykick=0.0,
                            unit="dimensionless"),
            elements.Drift(ds=0.0725, nslice=ns, name="DHG0385_downstream"),
            elements.Drift(ds=1.4345, nslice=ns, name="drift_to_BTV0390"),
            elements.BeamMonitor(name="BTV0390", backend="h5", encoding="g"),
            # BTV0390 is 0.4436 m upstream of the CA.STRAIGHT sequence origin.
            elements.Drift(ds=0.4436, nslice=ns,
                           name="BTV0390_to_CA_STRAIGHT"),
        ]
        return self._add_radial_aperture_checks(lattice)

    def get_impactx_lattice_from_btv805_to_screen(
            self, kinetic_energy_ev, screen="BTV910"):
        """Build a BTV805-start sublattice via CA.STRAIGHT to one screen.
        ``screen`` is ``"BTV910"`` for the straight dump (spectrometer off),
        or ``"BTV930"`` for the bent spectrometer branch.
        """
        from impactx import elements

        lattice = self.get_impactx_lattice_from_monitor(
            kinetic_energy_ev, "BTV805")
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

    def track_from_btv805(self, beam, spectrometer_on=True, verbose=False,
                          runnable=None):
        """Track from BTV805 and return beam distributions at BPMs and screens ``spectrometer_on`` selects
        the bent BTV930 or straight BTV910 branch.
        """
        from abel.wrappers.impactx.impactx_wrapper import run_impactx

        screen = "BTV930" if spectrometer_on else "BTV910"
        target_names = {"BTV0810", "BPM0820", "BPM0890", "BPM0920", screen}
        lattice = self.get_impactx_lattice_from_btv805_to_screen(
            beam.energy(), screen)
        targets = [
            getattr(element, "name") for element in lattice
            if getattr(element, "name", None) in target_names
        ]
        beams = {"BTV805 (lens entrance)": beam}
        for monitor_name in targets:
            monitor_lattice = self.get_impactx_lattice_from_btv805_to_screen(
                beam.energy(), screen)
            lattice_to_monitor = self._truncate_lattice_at_monitor(
                monitor_lattice, monitor_name)
            monitor_beam, _ = run_impactx(
                lattice_to_monitor, beam, nom_energy=beam.energy(),
                runnable=runnable, keep_data=self.keep_data, save_beams=False,
                space_charge=self.enable_space_charge, csr=self.enable_csr,
                isr=self.enable_isr, isr_on_ref_part=self.isr_on_ref_part,
                verbose=verbose,
            )
            label = "BTV810" if monitor_name == "BTV0810" else monitor_name
            if monitor_name == screen:
                label = f"{screen} (spectrometer {'ON' if spectrometer_on else 'OFF'})"
            beams[label] = monitor_beam
        return beams

    def track_from_qfd0350(self, beam, spectrometer_on=True, verbose=False,
                           runnable=None):
        """Track from QFD0350 entrance to BTV805, then down a selected branch."""
        from abel.wrappers.impactx.impactx_wrapper import run_impactx

        full_line = self._get_qfd0350_to_ca_straight_lattice(beam.energy())
        full_line.extend(self.get_impactx_lattice(beam.energy()))
        to_btv805 = self._truncate_lattice_at_monitor(full_line, "BTV805")
        beam805, _ = run_impactx(
            to_btv805, beam, nom_energy=beam.energy(), runnable=runnable,
            keep_data=self.keep_data, save_beams=False,
            space_charge=self.enable_space_charge, csr=self.enable_csr,
            isr=self.enable_isr, isr_on_ref_part=self.isr_on_ref_part,
            verbose=verbose,
        )
        return self.track_from_btv805(
            beam805, spectrometer_on=spectrometer_on, verbose=verbose,
            runnable=runnable)

    def track_clear_example(self, beam, spectrometer_on=True,
                            generate_at_btv805=False, verbose=False,
                            runnable=None):
        """Track the CLEAR example from QFD0350 or directly from BTV805."""
        if generate_at_btv805:
            return self.track_from_btv805(
                beam, spectrometer_on=spectrometer_on, verbose=verbose,
                runnable=runnable)
        return self.track_from_qfd0350(
            beam, spectrometer_on=spectrometer_on, verbose=verbose,
            runnable=runnable)

    @staticmethod
    def _select_profile_monitors(station_beams, prefix, requested=None):
        """Select matching monitor beams, allowing labels with descriptive suffixes."""
        available = [name for name in station_beams if name.startswith(prefix)]
        if requested is None:
            return [(name, station_beams[name]) for name in available]
        if isinstance(requested, str):
            requested = [requested]

        selected = []
        for query in requested:
            matches = [name for name in available
                       if name == query or name.startswith(f"{query} ")]
            if not matches:
                raise ValueError(
                    f"{query} was not found among {prefix} monitors. "
                    f"Available: {available}")
            selected.extend((name, station_beams[name]) for name in matches)
        return selected

    def plot_btv_profiles(self, station_beams, btvs=None, bins=90,
                          common_limits=True):
        """Plot transverse x-y images for selected BTV screens.
           Screen names to plot, such as "BTV805" or
            ["BTV805", "BTV810", "BTV910"]. Defaults to all BTVs after plasma lens.
            Uses the same x-y range for all selected screens.
        """
        import matplotlib.pyplot as plt

        selected = self._select_profile_monitors(station_beams, "BTV", btvs)
        if not selected:
            raise ValueError("No BTV screen beams were supplied to plot.")
        profile_data = [
            (name, np.asarray(beam.xs()) * 1e6, np.asarray(beam.ys()) * 1e6)
            for name, beam in selected
        ]
        hist_range = None
        if common_limits:
            all_x = np.concatenate([x for _, x, _ in profile_data])
            all_y = np.concatenate([y for _, _, y in profile_data])
            hist_range = [
                (float(np.min(all_x)), float(np.max(all_x))),
                (float(np.min(all_y)), float(np.max(all_y))),
            ]

        ncols = min(3, len(profile_data))
        nrows = int(np.ceil(len(profile_data) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False,
            layout="constrained", sharex=True, sharey=True)
        for ax, (name, x_um, y_um) in zip(axes.flat, profile_data):
            ax.hist2d(x_um, y_um, bins=bins, range=hist_range, cmap="viridis")
            ax.set_title(
                f"{name}\n$\\sigma_x$={np.std(x_um):.1f} µm, "
                f"$\\sigma_y$={np.std(y_um):.1f} µm")
            ax.set_xlabel("x [µm]")
            ax.set_ylabel("y [µm]")
            ax.set_aspect("equal")
        for ax in axes.flat[len(profile_data):]:
            ax.set_visible(False)
        fig.suptitle(f"CLEAR transverse profiles: {', '.join(n for n, _, _ in profile_data)}")
        return fig, axes

    def plot_bpm_centroids(self, station_beams, bpms=None):
        """Plot selected BPM centroids and RMS sizes.
              BPM names to plot. Defaults to all returned BPMs, in beamline order towards downstream.
        """
        import matplotlib.pyplot as plt

        selected = self._select_profile_monitors(station_beams, "BPM", bpms)
        if not selected:
            raise ValueError("No BPM beam snapshots exists supplied to plot.")
        names = [name for name, _ in selected]
        x_mean = np.array([np.mean(beam.xs()) for _, beam in selected]) * 1e3
        y_mean = np.array([np.mean(beam.ys()) for _, beam in selected]) * 1e3
        x_rms = np.array([np.std(beam.xs()) for _, beam in selected]) * 1e3
        y_rms = np.array([np.std(beam.ys()) for _, beam in selected]) * 1e3

        print("BPM beam positions and RMS sizes:")
        print("BPM       x centroid [mm]  y centroid [mm]  sigma_x [µm]  sigma_y [µm]")
        for name, xm, ym, sx, sy in zip(
                names, x_mean, y_mean, x_rms * 1e3, y_rms * 1e3):
            print(f"{name:<9} {xm:>15.6g} {ym:>15.6g} {sx:>13.5g} {sy:>13.5g}")

        fig, axes = plt.subplots(
            2, 1, figsize=(max(7, 1.5 * len(names)), 6), sharex=True,
            layout="constrained")
        axes[0].errorbar(names, x_mean, yerr=x_rms, fmt="o-", capsize=4)
        axes[0].axhline(0, color="black", lw=0.8, alpha=0.5)
        axes[0].set_ylabel("horizontal x [mm]")
        axes[0].set_title("BPM centroid in beamline order; bars show ±1 RMS")
        axes[1].errorbar(names, y_mean, yerr=y_rms, fmt="o-", capsize=4,
                         color="tab:orange")
        axes[1].axhline(0, color="black", lw=0.8, alpha=0.5)
        axes[1].set_ylabel("vertical y [mm]")
        axes[1].set_xlabel("BPM")
        plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
        return fig, axes

    def plot_lattice_survey(self, kinetic_energy_ev):
        """Plot the BTV805-start lattice and both downstream sub lattices."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        lattice = self.get_impactx_lattice_from_monitor(
            kinetic_energy_ev, "BTV805")
        common_end = sum(float(getattr(element, "ds", 0.0)) for element in lattice)
        items = []
        s_position = 0.0
        for element in lattice:
            name = getattr(element, "name", None) or ""
            ds = float(getattr(element, "ds", 0.0))
            center = s_position + ds / 2
            if name.startswith(("Q", "DH")) and not name.endswith(("_upstream", "_downstream")):
                items.append(("magnet", name, center))
            elif name.startswith(("BTV", "BPM")):
                items.append(("marker", name, center))
            s_position += ds

        fig, ax = plt.subplots(figsize=(17, 8), layout="constrained")
        ax.plot([0, common_end], [0, 0], color="black", lw=2.2,
                label="Common line from BTV805")
        lens_center = next(pos for kind, name, pos in items if name == "BTV0800")
        lens_length = float(self.plasma_lens.length)
        ax.plot([lens_center-lens_length/2, lens_center+lens_length/2], [0, 0],
                color="tab:cyan", lw=7, solid_capstyle="butt", zorder=3)
        ax.annotate("ABEL plasma lens", (lens_center, 0), xytext=(0, 7),
                    textcoords="offset points", ha="center", color="teal", fontsize=8)
        items.append(("marker", "PLC0800M (plasma lens centre)", lens_center))

        magnet_items = [(name, pos) for kind, name, pos in items if kind == "magnet"]
        for i, (name, pos) in enumerate(magnet_items):
            is_quad = name.startswith("Q")
            color = "tab:purple" if is_quad else "tab:orange"
            marker = "^" if is_quad else "D"
            ax.scatter([pos], [0], color=color, marker=marker, s=36, zorder=4)
            height = 0.12 + 0.035 * (i % 3)
            ax.vlines(pos, 0.015, height, color=color, lw=0.8)
            ax.annotate(name, (pos, height), xytext=(0, 2), textcoords="offset points",
                        rotation=55, ha="left", va="bottom", fontsize=7, color=color)

        marker_items = [(name, pos) for kind, name, pos in items if kind == "marker"]
        for i, (name, pos) in enumerate(marker_items):
            color = "tab:blue" if name.startswith("BPM") else "tab:red"
            ax.scatter([pos], [0], color=color, marker="s", s=24, zorder=5)
            height = -0.12 - 0.035 * (i % 3)
            ax.vlines(pos, -0.015, height, color=color, lw=0.7)
            label = f"{name}\ns={pos:.2f} m" if name.startswith("BPM") else name
            ax.annotate(label, (pos, height), xytext=(0, -2), textcoords="offset points",
                        rotation=55, ha="right", va="top", fontsize=7, color=color)

        # Straight dump branch to BTV910.
        dump_end = common_end + _DUMP_BRANCH_LENGTH
        ax.plot([common_end, dump_end], [0, 0], color="tab:green", lw=2,
                label="CA.INAIR branch")
        ax.scatter([dump_end], [0], marker="s", color="tab:red", s=28, zorder=5)
        ax.annotate("BTV910", (dump_end, 0), xytext=(0, -18),
                    textcoords="offset points", ha="center", fontsize=8)

        # Bent spectrometer branch to BPM0920 and BTV930.
        dipole_length = _SPECTROMETER_DIPOLE_LENGTH
        theta_deg = _SPECTROMETER_DIPOLE_ANGLE_DEG
        theta = np.deg2rad(theta_deg)
        arc_length = theta * (0.5 * dipole_length) / np.sin(theta / 2)
        dipole_center = (34.899994842 - 13.5894 - 21.0) + arc_length / 2
        dipole_start = dipole_center - dipole_length / 2
        dipole_end = dipole_center + dipole_length / 2
        radius = dipole_length / theta
        bend_x0 = common_end + dipole_start
        arc_t = np.linspace(0.0, theta, 100)
        arc_x = bend_x0 + radius * np.sin(arc_t)
        arc_y = radius * (1.0 - np.cos(arc_t))
        ax.plot([common_end, bend_x0], [0, 0], color="tab:blue", lw=2,
                label="CA.SPECT branch")
        ax.plot(arc_x, arc_y, color="tab:blue", lw=3)
        ax.annotate("BHB0900", (np.mean(arc_x), np.mean(arc_y)), xytext=(0, 10),
                    textcoords="offset points", ha="center", color="tab:orange", fontsize=8)
        remaining = _SPECTROMETER_LENGTH - dipole_start - dipole_length
        end_x = arc_x[-1] + remaining * np.cos(theta)
        end_y = arc_y[-1] + remaining * np.sin(theta)
        ax.plot([arc_x[-1], end_x], [arc_y[-1], end_y], color="tab:blue", lw=2)
        bpm_s = dipole_start + dipole_length + (0.7436 - dipole_length / 2)
        bpm_x = arc_x[-1] + (bpm_s - dipole_start - dipole_length) * np.cos(theta)
        bpm_y = arc_y[-1] + (bpm_s - dipole_start - dipole_length) * np.sin(theta)
        ax.scatter([bpm_x], [bpm_y], marker="s", color="tab:blue", s=25, zorder=5)
        ax.annotate(f"BPM0920\ns={common_end + bpm_s:.2f} m", (bpm_x, bpm_y), xytext=(0, -18),
                    textcoords="offset points", ha="center", fontsize=8)
        ax.scatter([end_x], [end_y], marker="s", color="tab:red", s=28, zorder=5)
        ax.annotate("BTV0930", (end_x, end_y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8)

        legend_items = [
            Line2D([0], [0], marker="^", color="none", markerfacecolor="tab:purple", label="Quadrupole"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="tab:orange", label="Corrector"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="tab:blue", label="BPM"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="tab:red", label="BTV / screen"),
            Line2D([0], [0], color="tab:cyan", lw=7, label="ABEL plasma lens"),
            Line2D([0], [0], color="black", lw=2, label="Common line"),
            Line2D([0], [0], color="tab:green", lw=2, label="CA.INAIR branch"),
            Line2D([0], [0], color="tab:blue", lw=2, label="CA.SPECT branch"),
        ]
        ax.set_xlabel("survey x [m]")
        ax.set_ylabel("survey y [m]")
        ax.set_title("CLEAR sublattice survey: BTV805 to BTV910 / BTV930")
        ax.grid(True, alpha=0.25)
        ax.legend(handles=legend_items, loc="upper left", ncol=2, fontsize=8)
        ax.set_ylim(-0.55, 0.55)
        return fig, ax
