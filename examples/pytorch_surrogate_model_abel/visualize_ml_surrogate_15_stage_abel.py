#!/usr/bin/env python3
#
# ABEL visualization for the 15-stage PyTorch surrogate example.
# The figure layout follows the ImpactX example script:
# https://impactx.readthedocs.io/en/latest/usage/examples/pytorch_surrogate_model/README.html

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpmd_api as io
import pandas as pd
from matplotlib import pyplot as plt
from scipy.constants import c, e, m_e


EBEAM_LPA_Z0 = -107e-6


@dataclass
class BeamMoments:
    name: str
    label: str
    location_m: float
    mean_energy_gev: float
    emit_nx_m: float
    emit_ny_m: float
    sigma_x_m: float
    sigma_y_m: float
    sigma_xp: float
    sigma_yp: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the ABEL ML surrogate benchmark.")
    parser.add_argument(
        "--file-path",
        "-f",
        type=Path,
        default=Path(__file__).resolve().parent / "final.h5",
        help="OpenPMD/HDF5 file written by run_ml_surrogate_15_stage_abel.py",
    )
    parser.add_argument(
        "--save-png", action="store_true", help="non-interactive run: save to PNGs"
    )
    parser.add_argument(
        "--num-stages", "-n", type=int, default=15, help="number of stages to plot"
    )
    parser.add_argument(
        "--stages_to_plot",
        "-s",
        type=int,
        help="stage-exit phase-space plots to create",
        nargs="*",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=None,
        help="optional maximum particles to scatter per phase-space plot",
    )
    return parser.parse_args()


def load_beam_dataframe(series: io.Series, particles) -> pd.DataFrame:
    xs = particles["position"]["x"].load_chunk()
    ys = particles["position"]["y"].load_chunk()
    if "t" in particles["position"]:
        zs = particles["position"]["t"].load_chunk()
        px = particles["momentum"]["x"].load_chunk()
        py = particles["momentum"]["y"].load_chunk()
        pt = particles["momentum"]["t"].load_chunk()
        reference_beta_gamma = particles.get_attribute("abel_reference_beta_gamma")
        series.flush()
        beta_gamma_x = px * reference_beta_gamma
        beta_gamma_y = py * reference_beta_gamma
        beta_gamma_z = (1.0 + pt) * reference_beta_gamma
        gamma = np.sqrt(1.0 + beta_gamma_x**2 + beta_gamma_y**2 + beta_gamma_z**2)
        xprime = beta_gamma_x / beta_gamma_z
        yprime = beta_gamma_y / beta_gamma_z
    else:
        zs = particles["position"]["z"].load_chunk()
        uxs = particles["momentum"]["x"].load_chunk()
        uys = particles["momentum"]["y"].load_chunk()
        uzs = particles["momentum"]["z"].load_chunk()
        series.flush()
        reference_beta_gamma = np.mean(uzs / c)
        px = (uxs / c) / reference_beta_gamma
        py = (uys / c) / reference_beta_gamma
        pt = (uzs / c - reference_beta_gamma) / reference_beta_gamma
        beta_gamma_x = uxs / c
        beta_gamma_y = uys / c
        beta_gamma_z = uzs / c
        gamma = np.sqrt(1.0 + beta_gamma_x**2 + beta_gamma_y**2 + beta_gamma_z**2)
        xprime = uxs / uzs
        yprime = uys / uzs

    df = pd.DataFrame(
        {
            "position_x": xs,
            "position_y": ys,
            "position_t": zs,
            "momentum_x": px,
            "momentum_y": py,
            "momentum_t": pt,
            "xprime": xprime,
            "yprime": yprime,
            "beta_gamma_x": beta_gamma_x,
            "beta_gamma_y": beta_gamma_y,
            "beta_gamma_z": beta_gamma_z,
            "gamma": gamma,
        }
    )
    df.attrs["reference_beta_gamma"] = float(reference_beta_gamma)
    return df


def list_beam_species(series: io.Series) -> list[str]:
    step = list(series.iterations)[0]
    return sorted(series.iterations[step].particles)


def load_all_beams(path: Path, num_stages: int) -> tuple[dict[str, pd.DataFrame], dict[str, float], dict[str, str]]:
    series = io.Series(str(path), io.Access.read_only)
    step = list(series.iterations)[0]
    particles = series.iterations[step].particles
    beam_names = list_beam_species(series)

    expected = ["beam_000_source"] + [
        f"beam_{stage_i:03d}_stage_{stage_i}" for stage_i in range(1, num_stages + 1)
    ]
    missing = [name for name in expected if name not in beam_names]
    if missing:
        raise FileNotFoundError(
            "The selected H5 file does not contain the expected ABEL stage snapshots: "
            + ", ".join(missing)
        )

    beams: dict[str, pd.DataFrame] = {}
    locations: dict[str, float] = {}
    labels: dict[str, str] = {}
    for name in expected:
        species = particles[name]
        beams[name] = load_beam_dataframe(series, species)
        locations[name] = species.get_attribute("abel_location_m")
        labels[name] = species.get_attribute("abel_snapshot_label")

    series.close()
    return beams, locations, labels


def rms(values) -> float:
    return float(np.std(values, ddof=0))


def emittance(values_a, values_b) -> float:
    cov = np.cov(values_a, values_b, ddof=0)
    return float(np.sqrt(np.linalg.det(cov)))


def calculate_moments(
    beams: dict[str, pd.DataFrame], locations: dict[str, float], labels: dict[str, str]
) -> list[BeamMoments]:
    rows = []
    for name, beam in beams.items():
        rows.append(
            BeamMoments(
                name=name,
                label=labels[name],
                location_m=locations[name],
                mean_energy_gev=float(np.mean(beam["gamma"]) * m_e * c**2 / e * 1.0e-9),
                emit_nx_m=emittance(beam["position_x"], beam["beta_gamma_x"]),
                emit_ny_m=emittance(beam["position_y"], beam["beta_gamma_y"]),
                sigma_x_m=rms(beam["position_x"]),
                sigma_y_m=rms(beam["position_y"]),
                sigma_xp=rms(beam["xprime"]),
                sigma_yp=rms(beam["yprime"]),
            )
        )
    return rows


def plot_moments(rows: list[BeamMoments], save_png: bool) -> None:
    fig, axT = plt.subplots(2, 2, figsize=(10, 8))
    ymarker = "^"
    s = np.array([row.location_m for row in rows])

    ax = axT[0][0]
    ax.plot(s, np.array([row.emit_nx_m for row in rows]) * 1e6, "bo", label="x")
    ax.plot(
        s,
        np.array([row.emit_ny_m for row in rows]) * 1e6,
        "r",
        marker=ymarker,
        linestyle="None",
        label="y",
    )
    ax.legend()
    ax.set_xlabel("s (m)")
    ax.set_ylabel(r"emittance (mm-mrad)")

    ax = axT[0][1]
    ax.plot(s, np.array([row.mean_energy_gev for row in rows]), "go")
    ax.set_xlabel("s (m)")
    ax.set_ylabel(r"mean energy (GeV)")

    ax = axT[1][0]
    ax.plot(s, np.array([row.sigma_x_m for row in rows]) * 1e6, "bo", label="x")
    ax.plot(
        s,
        np.array([row.sigma_y_m for row in rows]) * 1e6,
        "r",
        marker=ymarker,
        linestyle="None",
        label="y",
    )
    ax.legend()
    ax.set_xlabel("s (m)")
    ax.set_ylabel(r"beam width ($\mu$m)")

    ax = axT[1][1]
    ax.semilogy(s, np.array([row.sigma_xp for row in rows]) * 1e3, "bo", label="x")
    ax.semilogy(
        s,
        np.array([row.sigma_yp for row in rows]) * 1e3,
        "r",
        marker=ymarker,
        linestyle="None",
        label="y",
    )
    ax.legend()
    ax.set_xlabel("s (m)")
    ax.set_ylabel(r"divergence (mrad)")

    plt.tight_layout()
    if save_png:
        plt.savefig("lpa_ml_surrogate_moments.png")
        plt.close(fig)
    else:
        plt.show()


def maybe_sample(beam: pd.DataFrame, max_particles: int | None) -> pd.DataFrame:
    if max_particles is None or len(beam) <= max_particles:
        return beam
    return beam.sample(n=max_particles, random_state=1)


def plot_beam_df(
    beam_at_step,
    axT,
    unit=1e6,
    unit_z=1e6,
    unit_label=r"$\mu$m",
    unit_z_label=r"$\xi$ ($\mu$m)",
    alpha=0.6,
    cmap=None,
    color="red",
    size=0.1,
    t_offset=0.0,
    label=None,
    z_ticks=None,
):
    ax = axT[0][0]
    ax.scatter(
        beam_at_step.position_x.multiply(unit),
        beam_at_step.position_y.multiply(unit),
        c=color,
        s=size,
        alpha=alpha,
        cmap=cmap,
    )
    ax.set_xlabel(r"x (%s)" % unit_label)
    ax.set_ylabel(r"y (%s)" % unit_label)
    ax.axes.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))

    ax = axT[0][1]
    ax.scatter(
        beam_at_step.position_t.multiply(unit_z) - t_offset,
        beam_at_step.position_x.multiply(unit),
        c=color,
        s=size,
        alpha=alpha,
        cmap=cmap,
    )
    ax.set_xlabel(r"%s" % unit_z_label)
    ax.set_ylabel(r"x (%s)" % unit_label)
    ax.axes.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    if z_ticks is not None:
        ax.set_xticks(z_ticks)

    ax = axT[0][2]
    ax.scatter(
        beam_at_step.position_t.multiply(unit_z) - t_offset,
        beam_at_step.position_y.multiply(unit),
        c=color,
        s=size,
        alpha=alpha,
        cmap=cmap,
    )
    ax.set_xlabel(r"%s" % unit_z_label)
    ax.set_ylabel(r"y (%s)" % unit_label)
    ax.axes.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    if z_ticks is not None:
        ax.set_xticks(z_ticks)

    ax = axT[1][0]
    ax.scatter(
        beam_at_step.momentum_x,
        beam_at_step.momentum_y,
        c=color,
        s=size,
        alpha=alpha,
        cmap=cmap,
    )
    ax.set_xlabel("px")
    ax.set_ylabel("py")
    ax.axes.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))

    ax = axT[1][1]
    ax.scatter(
        beam_at_step.momentum_t,
        beam_at_step.momentum_x,
        c=color,
        s=size,
        alpha=alpha,
        cmap=cmap,
    )
    ax.set_xlabel("pt")
    ax.set_ylabel("px")
    ax.axes.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))

    ax = axT[1][2]
    ax.scatter(
        beam_at_step.momentum_t,
        beam_at_step.momentum_y,
        c=color,
        s=size,
        alpha=alpha,
        label=label,
        cmap=cmap,
    )
    if label is not None:
        ax.legend()
    ax.set_xlabel("pt")
    ax.set_ylabel("py")
    ax.axes.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))

    ax = axT[2][0]
    ax.scatter(
        beam_at_step.position_x.multiply(unit),
        beam_at_step.momentum_x,
        c=color,
        s=size,
        alpha=alpha,
        cmap=cmap,
    )
    ax.set_xlabel(r"x (%s)" % unit_label)
    ax.set_ylabel("px")
    ax.axes.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))

    ax = axT[2][1]
    ax.scatter(
        beam_at_step.position_y.multiply(unit),
        beam_at_step.momentum_y,
        c=color,
        s=size,
        alpha=alpha,
        cmap=cmap,
    )
    ax.set_xlabel(r"y (%s)" % unit_label)
    ax.set_ylabel("py")
    ax.axes.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))

    ax = axT[2][2]
    ax.scatter(
        beam_at_step.position_t.multiply(unit_z) - t_offset,
        beam_at_step.momentum_t,
        c=color,
        s=size,
        alpha=alpha,
        cmap=cmap,
    )
    ax.set_xlabel(r"%s" % unit_z_label)
    ax.set_ylabel("pt")
    ax.axes.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    if z_ticks is not None:
        ax.set_xticks(z_ticks)
    plt.tight_layout()


def to_t_global(beam: pd.DataFrame, ref_pz: float, ref_z: float) -> None:
    ref_pt = -np.sqrt(1.0 + ref_pz**2)
    dx = beam["position_x"].to_numpy(copy=True)
    dy = beam["position_y"].to_numpy(copy=True)
    dt = beam["position_t"].to_numpy(copy=True)
    dpx = beam["momentum_x"].to_numpy(copy=True)
    dpy = beam["momentum_y"].to_numpy(copy=True)
    dpt = beam["momentum_t"].to_numpy(copy=True)

    denominator = ref_pt + ref_pz * dpt
    new_x = dx + ref_pz * dpx * dt / denominator
    new_y = dy + ref_pz * dpy * dt / denominator
    pz = np.sqrt(-1.0 + denominator**2 - (ref_pz * dpx) ** 2 - (ref_pz * dpy) ** 2)
    new_t = dt * pz / denominator + ref_z
    beam["position_x"] = new_x
    beam["position_y"] = new_y
    beam["position_t"] = new_t
    beam["momentum_x"] = dpx * ref_pz
    beam["momentum_y"] = dpy * ref_pz
    beam["momentum_t"] = pz


def plot_phase_space(
    beam: pd.DataFrame,
    title: str,
    output_name: str,
    save_png: bool,
    max_particles: int | None,
    reference_ct: float,
) -> None:
    beam_to_plot = beam.copy()
    ref_pz = beam_to_plot.attrs["reference_beta_gamma"]
    to_t_global(beam_to_plot, ref_pz=ref_pz, ref_z=reference_ct + EBEAM_LPA_Z0)
    beam_to_plot = maybe_sample(beam_to_plot, max_particles)
    t_offset = reference_ct * 1e6
    fig, axT = plt.subplots(3, 3, figsize=(10, 8))
    fig.suptitle(title)
    plot_beam_df(
        beam_to_plot,
        axT,
        alpha=0.6,
        color="red",
        unit_z=1e6,
        unit_z_label=r"$\xi$ ($\mu$m)",
        t_offset=t_offset,
        z_ticks=[-107.3, -106.6],
    )
    if save_png:
        plt.savefig(output_name)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    if args.num_stages < 1:
        raise ValueError("--num-stages must be positive.")

    beams, locations, labels = load_all_beams(args.file_path, args.num_stages)
    rows = calculate_moments(beams, locations, labels)
    plot_moments(rows, args.save_png)

    plot_phase_space(
        beams["beam_000_source"],
        f"initially, ct={locations['beam_000_source']:.2f} m",
        "initial_phase_spaces.png",
        args.save_png,
        args.max_particles,
        locations["beam_000_source"],
    )

    if args.stages_to_plot is not None:
        for stage_i in args.stages_to_plot:
            if stage_i < 1 or stage_i > args.num_stages:
                raise ValueError(
                    f"Stage {stage_i} is outside the available range 1..{args.num_stages}."
                )
            name = f"beam_{stage_i:03d}_stage_{stage_i}"
            plot_phase_space(
                beams[name],
                f"stage {stage_i}, ct={locations[name]:.2f} m",
                f"stage_{stage_i - 1}_phase_spaces.png",
                args.save_png,
                args.max_particles,
                locations[name],
            )


if __name__ == "__main__":
    main()
