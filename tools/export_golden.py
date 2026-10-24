#!/usr/bin/env python3
"""Export golden fixtures from the Python implementation for the C++ port.

Run from the repository root with the Python environment that the reference
numbers come from::

    .venv/bin/python tools/export_golden.py                    # synthetic goldens
    .venv/bin/python tools/export_golden.py --kitti-ref DIR    # + KITTI references

Everything is deterministic: inputs are generated from fixed seeds and stored
alongside the outputs, so the C++ tests never need NumPy's RNG.

Output layout (``tests/golden/``):

``testdata/<test_module>.json``
    Random data each Python test consumed, recorded per node id by
    ``tools/golden_recorder.py`` and replayed by the C++ test of the same name.
``<module>.json``
    Reference inputs and outputs of the library functions, compared by the
    C++ golden tests at tight tolerances (exact where the computation is the
    same OpenCV code).
``kitti/``
    First-300-frame runs of KITTI 00 and 05 and full-sequence references,
    copied from Python runs of ``scripts/run_slam.py``.

Arrays use the same exact base64 encoding as the recorder. Floats that must be
compared bit-for-bit are stored as ``float.hex()`` strings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "tools"))

import cv2  # noqa: E402
import yaml  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import golden_recorder  # noqa: E402
from monocular_slam.cli.common import _parse_scalar  # noqa: E402
from monocular_slam.config import Config, ConfigError  # noqa: E402

GOLDEN = REPO / "tests" / "golden"


def enc(value):
    """Exact array encoding (never content-shared: module goldens are small)."""
    return golden_recorder.encode(value, share=False)


def fhex(x: float) -> str:
    return float(x).hex()


def write(name: str, payload: dict) -> None:
    path = GOLDEN / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(REPO)} ({path.stat().st_size / 1024:.1f} KB)")


# --------------------------------------------------------------------------- #
# Python-compatibility primitives (float repr, round, format, json, numpy)
# --------------------------------------------------------------------------- #


def export_compat() -> None:
    rng = np.random.default_rng(1234)
    specials = [
        0.0, -0.0, 1.0, -1.0, 0.1, 0.2, 0.3, 1 / 3, 2 / 3, 1e-4, 1e-5, 9.999e-5, 1e15, 1e16,
        1e17, 1.5e16, 123456789012345678.0, 5e-324, 2.2250738585072014e-308,
        1.7976931348623157e308, 100.0, 1234.5, 0.85, 0.001, 3.14159, 2.675, 1e22, 1e-7,
        12.0, 4541.0, 99.8, 68.04, -9.0, 0.5, 1.345,
    ]
    values = list(specials)
    for _ in range(500):
        mantissa = rng.standard_normal()
        exponent = int(rng.integers(-25, 25))
        values.append(float(mantissa * 10.0**exponent))
    values += [float(v) for v in rng.integers(-10**6, 10**6, 40)]
    values += [round(float(v), int(d)) for v, d in zip(rng.uniform(-1000, 1000, 100),
                                                        rng.integers(0, 9, 100))]
    repr_cases = [[fhex(v), repr(v)] for v in values]
    repr_cases += [["nan", repr(float("nan"))], ["inf", repr(float("inf"))],
                   ["-inf", repr(float("-inf"))]]

    round_cases = []
    tie_values = [0.5, 1.5, 2.5, -0.5, -1.5, 0.125, 0.375, 2.675, 1.005, 0.045, 1e-9, 123.4567]
    for v in tie_values + [float(x) for x in rng.uniform(-500, 500, 80)]:
        for digits in range(0, 9):
            round_cases.append([fhex(v), digits, fhex(round(v, digits))])
    round_cases.append(["nan", 4, "nan"])
    round_cases.append(["inf", 4, "inf"])

    format_cases = []
    specs = [".0f", ".1f", ".2f", ".3f", ".4f", ".6f", ".2e", ".12e", ".4g", ".3g", "g", "9.3f",
             "10.3f", "7d"]
    sample = specials + [float(v) for v in rng.standard_normal(40) * 1000]
    for v in sample:
        for spec in specs:
            if spec.endswith("d"):
                continue
            # [value, spec, f"{v:spec}", "%spec" % v]
            format_cases.append([fhex(v), spec, format(v, spec), ("%" + spec) % v])
    int_cases = [[int(n), f"{int(n):,}", f"{int(n):7d}"] for n in
                 [0, 7, 999, 1000, 4541, 123456, 1234567, -1234567, 10**12]]

    stats_cases = []
    for n in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 100, 127, 128, 129, 255, 256, 257, 1000,
              1201]:
        data = rng.standard_normal(n) * 10.0 + 3.0
        positive = np.abs(data) + 0.1
        stats_cases.append({
            "data": enc(data),
            "sum": fhex(np.sum(data)),
            "mean": fhex(np.mean(data)),
            "std": fhex(np.std(data)),
            "median": fhex(np.median(data)),
            "min": fhex(np.min(data)),
            "max": fhex(np.max(data)),
            "rms": fhex(np.sqrt(np.mean(data**2))),
            "positive": enc(positive),
            "percentile_20_80": [fhex(v) for v in np.percentile(positive, [20, 80])],
            "percentile_50": fhex(np.percentile(positive, 50)),
        })

    # np.argsort (default kind) is NOT stable; the C++ port reproduces NumPy's
    # introsort so ties come out in the same order. Arrays with many ties.
    argsort_cases = []
    for n in [1, 2, 3, 5, 10, 15, 16, 17, 20, 33, 50, 100, 257, 500, 3000]:
        data = np.round(rng.standard_normal(n), 1)
        argsort_cases.append({"data": enc(data), "argsort": enc(np.argsort(data)),
                              "argsort_neg": enc(np.argsort(-data))})
    data = rng.standard_normal(200)
    data[::7] = np.nan
    argsort_cases.append({"data": enc(data), "argsort": enc(np.argsort(data)),
                          "argsort_neg": enc(np.argsort(-data))})

    json_objects = [
        {"a": 1, "b": 2.5, "c": None, "d": True, "e": False, "f": "text"},
        {"nested": {"list": [1, 2.0, 3e-7, -0.0], "empty_list": [], "empty_dict": {}},
         "nan": float("nan"), "inf": float("inf"), "ninf": float("-inf")},
        {"unicode": "café — ±", "escapes": "quote\" backslash\\ tab\t newline\n \x01",
         "list_of_dicts": [{"x": 1}, {"y": [None, True]}]},
        [1, [2, [3, []]], {}],
        {"big": 1e16, "small": 1e-5, "int": 4541, "neg": -12, "float_int": 100.0},
    ]
    json_cases = [[json.dumps(o), json.dumps(o, indent=2)] for o in json_objects]

    csv_rows = [
        {"index": 1, "status": "ok", "reason": "", "ratio": 0.5, "flag": True, "nan": float("nan")},
        {"index": 2, "status": "failed", "reason": "has,comma", "ratio": 1e-5, "flag": False,
         "nan": float("inf")},
        {"index": 3, "status": 'quote"inside', "reason": "line\nbreak", "ratio": -0.0,
         "flag": False, "nan": 12.0},
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    csv_input = [[(k, fhex(v) if isinstance(v, float) else v) for k, v in row.items()]
                 for row in csv_rows]

    savetxt_rows = rng.standard_normal((5, 12)) * np.array([1, 1e-3, 1e3, 1e-12] * 3)
    savetxt_buffer = io.StringIO()
    np.savetxt(savetxt_buffer, savetxt_rows, fmt="%.12e")

    write("compat", {
        "repr": repr_cases,
        "round": round_cases,
        "format": format_cases,
        "int_format": int_cases,
        "stats": stats_cases,
        "argsort": argsort_cases,
        "json": json_cases,
        "csv": {"rows": csv_input, "text": buffer.getvalue()},
        "savetxt": {"rows": enc(savetxt_rows), "text": savetxt_buffer.getvalue()},
    })


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": repr(value)}
    if isinstance(value, tuple):
        return {"tuple": [_jsonable(v) for v in value]}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _typed(value):
    """Value plus its Python type name, for coercion goldens."""
    return {"type": type(value).__name__, "value": _jsonable(value)}


def export_config() -> None:
    kitti_yaml = REPO / "configs" / "kitti.yaml"
    odd_overrides = {
        "dataset.path": "with: colon",
        "dataset.camera": "yes",
        "output.root": "~",
        "output.run_name": "{sequence}_run",
        "output.plot_format": "-dash",
        "features.detector": "#hash",
        "matcher.matcher": "tail #comment",
        "odometry.ransac_method": " lead",
        "odometry.scale_source": "trail ",
        "runtime.log_level": "1e3",
        "dataset.sequence": "07",
        "evaluation.alignment": "null",
        "evaluation.rpe_deltas": [2, 4],
        "evaluation.drift_segment_lengths_m": [50, 1e-5, 1e16, 0.1],
        "odometry.min_scale": 1e-7,
        "odometry.max_scale": 1e16,
        "pose_graph.huber_k": 2.0,
        "dataset.max_frames": 300,
        "matcher.max_distance": None,
    }
    yaml_texts = {}
    for name, config in [
        ("default", Config()),
        ("kitti", Config.from_yaml(kitti_yaml)),
        ("odd", Config().with_overrides(odd_overrides)),
        ("odd2", Config().with_overrides({
            "dataset.path": "a'b", "dataset.camera": "x: y", "output.root": "[list]",
            "output.run_name": "{braces}", "features.detector": "@at", "matcher.matcher": "it's",
            "odometry.ransac_method": "", "odometry.scale_source": "123",
            "runtime.log_level": "0x1F", "evaluation.alignment": "True",
            "dataset.sequence": "1:30", "output.plot_format": "a\tb",
        })),
    ]:
        buffer = io.StringIO()
        yaml.safe_dump(config.to_dict(), buffer, sort_keys=False, default_flow_style=False)
        yaml_texts[name] = {"yaml": buffer.getvalue(), "dict": _jsonable(config.to_dict())}

    coercion_inputs = [
        ("loop_closure.enabled", v) for v in
        [True, False, "true", "True", " yes ", "no", "on", "OFF", "1", "0", "maybe", 1, 0, 1.0,
         None, [True]]
    ] + [
        ("odometry.min_inliers", v) for v in
        [40, "40", " 40 ", "4_0", "+5", "-3", 3.7, -3.7, "3.7", True, False, None, [1], "many",
         "1e3", 1e3, "0x10", {"a": 1}]
    ] + [
        ("odometry.ransac_threshold_px", v) for v in
        [1, "1", "1e-4", "1E5", " 2.5 ", ".5", "5.", "inf", "-Infinity", "nan", True, "abc",
         None, "1_000.5", [1.0]]
    ] + [
        ("dataset.sequence", v) for v in [5, 1.5, True, [1, 2], {"a": 1}, "05", None, 1e16,
                                         [1, "x", None, 2.5]]
    ] + [
        ("dataset.max_frames", v) for v in [None, 7, "7", 7.9]
    ] + [
        ("matcher.max_distance", v) for v in [None, 64, "32.5"]
    ] + [
        ("evaluation.rpe_deltas", v) for v in [[1, 2, 3], [1, "2", 3.0], 5, "abc", [], [None]]
    ] + [
        ("evaluation.drift_segment_lengths_m", v) for v in [[100, 200.5], ["1e2"], [True]]
    ]
    coercion = []
    for dotted, value in coercion_inputs:
        section, key = dotted.split(".")
        try:
            config = Config.from_dict({section: {key: value}})
            out = {"ok": True, "result": _typed(getattr(getattr(config, section), key))}
        except ConfigError as exc:
            out = {"ok": False, "error": str(exc)}
        except (OverflowError, ValueError, TypeError) as exc:
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        coercion.append({"key": dotted, "input": _jsonable(value), "out": out})

    structural = []
    for data in [{"odometery": {}}, {"odometry": {"min_inlyers": 3, "zzz": 1}},
                 {"odometry": 5}, {"odometry": None}, {"odometry": [1]},
                 {"dataset": {"path": "x"}, "bogus": 1, "alpha": 2}]:
        try:
            Config.from_dict(data)
            structural.append({"input": data, "ok": True})
        except ConfigError as exc:
            structural.append({"input": data, "ok": False, "error": str(exc)})

    override_errors = []
    for overrides in [{"odometry.nope": 1}, {"nope.key": 1}, {"odometry.min_inliers.x": 1},
                      {"odometry": 1}, {"dataset.sequence": None}]:
        try:
            Config().with_overrides(overrides)
            override_errors.append({"input": overrides, "ok": True})
        except ConfigError as exc:
            override_errors.append({"input": overrides, "ok": False, "error": str(exc)})

    resolve_scalars = [
        "00", "0", "007", "08", "0o17", "0x1F", "0b101", "1_000", "1e-4", "1.0e-4", "1.0E+4", ".5",
        "5.", "-1", "+1", "-0", "~", "null", "Null", "NULL", "''", "yes", "No", "on", "OFF", "y",
        "n", "true", "True", "TRUE", "tRue", "1:30", "1:30.5", "-1:30", ".inf", "-.Inf", "+.INF",
        ".NaN", "'00'", '"0x1F"', "image_0", "data/kitti/dataset", "sequence_{sequence}",
        "1.2.3", "0.85", "1e3", "3.0e-3", "12_3.4_5", "-.5", "+.5e3", "0.", "--1", "1__0",
        "0x_1f", "0b", "0_", "_1", "190:20:30", "1:60", "!!str 123", "!!int '5'", "!!float 1",
        "[1, 2]", "{a: 1}", "2001-12-14",
    ]
    resolution = []
    for scalar in resolve_scalars:
        try:
            loaded = yaml.safe_load(f"v: {scalar}\n")["v"]
            kind = type(loaded).__name__
            if kind in ("date", "datetime"):
                resolution.append({"scalar": scalar, "type": kind, "value": str(loaded)})
            else:
                resolution.append({"scalar": scalar, "type": kind, "value": _jsonable(loaded)})
        except yaml.YAMLError as exc:
            resolution.append({"scalar": scalar, "type": "error", "value": str(exc)})

    parse_scalar_cases = []
    for text in ["true", "False", "yes", "off", "on", "no", "42", "-7", "3.5", "1e-3", "none",
                 "null", "", "magsac", "00", "05", "-05", "0", "-0", "007", "1_000", "inf",
                 "nan", "-", "--5", "1.5e3", " 3", "+5", "0x10", "NULL", "None"]:
        try:
            parse_scalar_cases.append({"text": text, "out": _typed(_parse_scalar(text))})
        except ValueError as exc:  # e.g. "--5": lstrip("-") passes isdigit, int() fails
            parse_scalar_cases.append({"text": text, "error": f"ValueError: {exc}"})

    write("config", {
        "configs": yaml_texts,
        "coercion": coercion,
        "structural": structural,
        "override_errors": override_errors,
        "yaml_resolution": resolution,
        "parse_scalar": parse_scalar_cases,
    })


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def export_transforms() -> None:
    from conftest import random_se3, small_se3
    from monocular_slam.geometry import transforms as tf

    rng = np.random.default_rng(42)
    poses = [random_se3(rng) for _ in range(25)] + [small_se3(rng, 40.0, 3.0) for _ in range(15)]
    poses.append(np.eye(4))
    poses.append(tf.se3_from_rt(Rotation.from_euler("y", 180, degrees=True).as_matrix(),
                                np.array([1.0, 2.0, 3.0])))
    poses.append(tf.se3_from_rt(Rotation.from_rotvec([0, 0, np.pi - 1e-7]).as_matrix(),
                                np.zeros(3)))
    poses.append(tf.se3_from_rt(Rotation.from_rotvec([1e-9, -2e-9, 3e-9]).as_matrix(),
                                np.zeros(3)))
    perturbed = []
    for T in poses[:15]:
        noisy = T.copy()
        noisy[:3, :3] += rng.normal(scale=1e-3, size=(3, 3))
        perturbed.append(noisy)
    perturbed.append(np.diag([1.0, 1.0, -1.0, 1.0]))
    xis = [rng.normal(size=6) * 0.4 for _ in range(20)] + [
        np.array([0.1, -0.2, 0.3, 1e-14, 0.0, 0.0]), np.array([1.0, 2.0, 3.0, 0, 0, 0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.0]),
    ]

    pose_cases = []
    for T in poses:
        pose_cases.append({
            "T": enc(T),
            "inverse": enc(tf.invert_se3(T)),
            "angle_rad": fhex(tf.rotation_angle_rad(T)),
            "angle_deg": fhex(tf.rotation_angle_deg(T[:3, :3])),
            "so3_log": enc(tf.so3_log(T[:3, :3])),
            "se3_log": enc(tf.se3_log(T)),
            "kitti_row": enc(tf.se3_to_kitti_row(T)),
            "is_valid": bool(tf.is_valid_se3(T)),
        })
    projection_cases = [{
        "M": enc(M),
        "so3": enc(tf.project_to_so3(M[:3, :3])),
        "se3": enc(tf.project_to_se3(M)),
        "is_rotation": bool(tf.is_rotation_matrix(M[:3, :3])),
    } for M in perturbed]
    exp_cases = [{"xi": enc(xi), "se3_exp": enc(tf.se3_exp(xi)),
                  "so3_exp": enc(tf.so3_exp(xi[3:]))} for xi in xis]
    pairs = [(poses[i], poses[i + 1]) for i in range(0, 20, 2)]
    relative_cases = [{"a": enc(a), "b": enc(b), "relative": enc(tf.relative_pose(a, b)),
                       "compose": enc(tf.compose(a, b, a))} for a, b in pairs]
    points = rng.normal(size=(30, 3)) * 5
    transform_points_case = {"T": enc(poses[3]), "points": enc(points),
                             "out": enc(tf.transform_points(poses[3], points))}

    euler_cases = []
    for seq, angles in [("y", 90.0), ("y", 1.5), ("y", -3.0), ("y", 180.0), ("y", 0.8),
                        ("x", 45.0), ("z", 90.0), ("y", 0.2), ("y", 3.0), ("y", 47.0),
                        ("y", 63.0), ("y", 5.0), ("y", 15.0), ("y", 30.0), ("y", 4.0),
                        ("y", 0.5), ("y", 2.0), ("y", 10.0), ("y", 2.4),
                        ("xyz", [20.0, -35.0, 60.0]), ("y", 360.0 / 120), ("y", 360.0 / 150),
                        ("y", 360.0 / 80), ("y", 360.0 / 60), ("y", 360.0 / 40),
                        ("y", 360.0 / 30), ("y", 360.0 / 20), ("y", 360.0 / 10),
                        ("y", 360.0 / 5)]:
        euler_cases.append({"seq": seq, "angles": angles,
                            "matrix": enc(Rotation.from_euler(seq, angles, degrees=True)
                                          .as_matrix())})
    rotvec_cases = [{"rotvec": enc(v), "matrix": enc(Rotation.from_rotvec(v).as_matrix())}
                    for v in [rng.normal(size=3) * s for s in (1e-9, 1e-4, 1e-3, 0.01, 0.5, 2.0,
                                                               3.1)]]
    write("transforms", {
        "poses": pose_cases,
        "projection": projection_cases,
        "exp": exp_cases,
        "relative": relative_cases,
        "transform_points": transform_points_case,
        "euler": euler_cases,
        "rotvec": rotvec_cases,
    })


def _noisy_trajectory(reference, rng, sigma_pos=0.3, yaw_bias_deg=0.05):
    from monocular_slam.geometry.pose import Trajectory
    from monocular_slam.geometry.transforms import invert_se3, se3_from_rt

    poses = [reference[0].copy()]
    R_bias = Rotation.from_euler("y", yaw_bias_deg, degrees=True).as_matrix()
    for i in range(len(reference) - 1):
        relative = invert_se3(reference[i]) @ reference[i + 1]
        step = se3_from_rt(R_bias @ relative[:3, :3], relative[:3, 3] * 1.02)
        poses.append(poses[-1] @ step)
    poses = np.stack(poses)
    poses[:, :3, 3] += rng.normal(scale=sigma_pos, size=(len(poses), 3))
    return Trajectory(poses, frame_ids=reference.frame_ids.copy())


def export_trajectory_and_metrics() -> None:
    from monocular_slam.evaluation import alignment, metrics
    from monocular_slam.geometry.pose import Trajectory
    from synthetic import straight_drive

    rng = np.random.default_rng(7)
    short = Trajectory(straight_drive(40, step=0.9, yaw_rate_deg=0.8))
    long = Trajectory(straight_drive(500, step=0.9, yaw_rate_deg=0.2))
    perturbed = short.poses.copy()
    perturbed[:, :3, :3] += rng.normal(scale=1e-3, size=(len(short), 3, 3))
    perturbed = Trajectory(perturbed)

    buffer = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    buffer.close()
    short.save_kitti(buffer.name)
    saved_text = Path(buffer.name).read_text()
    Path(buffer.name).unlink()

    T = short.poses[7].copy()
    trajectory_cases = {
        "short": enc(short.poses),
        "long": enc(long.poses),
        "step_lengths": enc(short.step_lengths()),
        "cumulative": enc(short.cumulative_distance()),
        "path_length": fhex(short.path_length()),
        "long_path_length": fhex(long.path_length()),
        "relative_1": enc(short.relative_poses(1)),
        "relative_3": enc(short.relative_poses(3)),
        "transform_T": enc(T),
        "transformed": enc(short.transformed(T, scale=2.5).poses),
        "relative_to_first": enc(short.transformed(T).relative_to_first().poses),
        "perturbed": enc(perturbed.poses),
        "orthonormalized": enc(perturbed.orthonormalized().poses),
        "kitti_text": saved_text,
    }

    estimate_short = _noisy_trajectory(short, rng, 0.2, 0.3)
    estimate_long = _noisy_trajectory(long, rng, 0.5, 0.05)
    scaled_long = estimate_long.transformed(np.eye(4), scale=0.37)
    cases = []
    for name, est, ref in [("short", estimate_short, short), ("long", estimate_long, long),
                           ("long_scaled", scaled_long, long)]:
        case = {"name": name, "estimated": enc(est.poses), "reference": enc(ref.poses)}
        for mode in ("sim3", "se3", "none"):
            aligned, result = alignment.align_trajectory(est, ref, mode=mode)
            ate = metrics.absolute_trajectory_error(est, ref, alignment=mode)
            case[f"align_{mode}"] = {
                "R": enc(result.R), "t": enc(result.t), "scale": fhex(result.scale),
                "aligned_positions": enc(aligned.positions[::25]), "describe": result.describe(),
                "ate": {k: (fhex(v) if isinstance(v, float) else v)
                        for k, v in ate.to_dict().items()},
                "ate_rmse": fhex(ate.rmse), "ate_errors": enc(ate.errors),
            }
        for delta in (1, 10, 100):
            if delta < len(est):
                rpe = metrics.relative_pose_error(est, ref, delta=delta)
                case[f"rpe_{delta}"] = {k: (fhex(v) if isinstance(v, float) else v)
                                        for k, v in rpe.__dict__.items()}
        drift = metrics.kitti_drift(est, ref)
        case["drift"] = {"translation_pct": fhex(drift.translation_pct),
                         "rotation_deg_per_m": fhex(drift.rotation_deg_per_m),
                         "n_segments": drift.n_segments}
        evaluation = metrics.evaluate_trajectory(est, ref, label=name, rpe_deltas=(1, 10, 100))
        case["evaluation_json"] = json.dumps(evaluation.to_dict(), indent=2)
        case["summary_line"] = evaluation.summary_line()
        cases.append(case)

    write("trajectory", trajectory_cases)
    write("metrics", {"cases": cases,
                      "ate_reduction": [[fhex(a), fhex(b), fhex(metrics.ate_reduction_pct(a, b))]
                                        for a, b in [(10.0, 4.0), (4.0, 5.0), (8.42, 2.7),
                                                     (11.88, 4.03), (16.21, 17.67)]]})


def export_pose_graph() -> None:
    from monocular_slam.geometry.pose import Trajectory
    from monocular_slam.geometry.transforms import invert_se3, se3_from_rt
    from monocular_slam.optimization.gtsam_backend import optimize_pose_graph
    from monocular_slam.optimization.pose_graph import build_pose_graph
    from test_pose_graph import FakeLoopClosure, add_yaw_drift, circular_trajectory

    cases = []
    inputs: dict = {}

    def run(name, trajectory, loops, **kwargs):
        graph = build_pose_graph(trajectory, loops)
        residuals = graph.edge_residuals()
        result = optimize_pose_graph(graph, **kwargs)
        key = hashlib.sha1(trajectory.poses.tobytes() + trajectory.frame_ids.tobytes()).hexdigest()
        inputs.setdefault(key[:16], {"poses": enc(trajectory.poses),
                                     "frame_ids": enc(trajectory.frame_ids)})
        cases.append({
            "name": name,
            "kwargs": kwargs,
            "trajectory": key[:16],
            "loops": [{"match_id": lc.match_id, "query_id": lc.query_id,
                       "T": enc(lc.T_match_query)} for lc in loops],
            "residuals_before": {k: enc(v) for k, v in residuals.items()},
            "optimized": enc(result.trajectory.poses),
            "initial_error": fhex(result.initial_error),
            "final_error": fhex(result.final_error),
            "iterations": result.iterations,
            "residuals_after": {k: enc(v) for k, v in result.residuals_after.items()},
            "to_dict_json": json.dumps(result.to_dict(), indent=2),
            "describe": graph.describe(),
        })

    reference = circular_trajectory(150)
    drifted = add_yaw_drift(reference, 0.2)
    loop = FakeLoopClosure(0, 149, invert_se3(reference[0]) @ reference[149])
    run("drift150_lm", drifted, [loop])
    run("drift150_lm_nonrobust", drifted, [loop], robust_loop_kernel=False)
    run("drift150_gn", drifted, [loop], optimizer="gauss_newton")
    reference120 = circular_trajectory(120)
    bogus = FakeLoopClosure(10, 60, se3_from_rt(np.eye(3), np.array([500.0, 0.0, 0.0])),
                            n_inliers=50)
    run("bogus_robust", reference120, [bogus], robust_loop_kernel=True)
    run("bogus_naive", reference120, [bogus], robust_loop_kernel=False)
    run("chain40", circular_trajectory(40), [])
    reference80 = circular_trajectory(80)
    drifted80 = add_yaw_drift(reference80, 0.3)
    two_loops = [FakeLoopClosure(0, 79, invert_se3(reference80[0]) @ reference80[79]),
                 FakeLoopClosure(20, 60, invert_se3(reference80[20]) @ reference80[60])]
    run("drift80_two_loops", drifted80, two_loops)
    frame_ids = Trajectory(circular_trajectory(20).poses, frame_ids=np.arange(0, 100, 5))
    run("chain20_frame_ids", frame_ids, [], max_iterations=5)
    write("pose_graph", {"inputs": inputs, "cases": cases})


_TRIANGULATION_CASES = ("forward_yaw1.5", "noise0.5_t1.5")


def export_epipolar() -> None:
    from monocular_slam.geometry import epipolar
    from synthetic import KITTI_K, forward_motion, make_scene, pure_rotation, \
        two_view_correspondences

    cases = []

    def add(name, motion, method="magsac", threshold=0.5, recover_kwargs=None, **corr_kwargs):
        pa, pb, cam1 = two_view_correspondences(motion, **corr_kwargs)
        cv2.setRNGSeed(0)
        em = epipolar.estimate_essential_matrix(pa, pb, KITTI_K, threshold_px=threshold,
                                                method=method)
        case = {
            "name": name, "method": method, "threshold": threshold,
            "motion": enc(motion), "points_a": enc(pa), "points_b": enc(pb),
            "points_cam1": enc(cam1) if name in _TRIANGULATION_CASES else None,
            "essential": {"ok": em.ok, "reason": em.reason, "n_points": em.n_points,
                          "E": None if em.E is None else enc(em.E),
                          "mask": enc(em.inlier_mask)},
            "flow": fhex(epipolar.median_flow_px(pa, pb)),
            "parallax_uncompensated": fhex(epipolar.median_parallax_deg(pa, pb, KITTI_K)),
        }
        if em.ok:
            pose = epipolar.recover_relative_pose(em.E, pa, pb, KITTI_K, em.inlier_mask,
                                                  **(recover_kwargs or {}))
            case["pose"] = {
                "ok": pose.ok, "reason": pose.reason,
                "T": None if pose.T is None else enc(pose.T),
                "mask": enc(pose.inlier_mask),
                "n_cheirality": pose.n_cheirality_inliers,
                "parallax": fhex(pose.median_parallax_deg),
                "rotation_deg": fhex(pose.rotation_deg),
                "recover_kwargs": recover_kwargs or {},
            }
            cv2.setRNGSeed(0)
            selection = epipolar.select_two_view_model(pa, pb, KITTI_K, em.E,
                                                       threshold_px=threshold)
            case["selection"] = {
                "score_h": fhex(selection.score_h), "score_e": fhex(selection.score_e),
                "ratio_h": fhex(selection.ratio_h), "degenerate": selection.is_degenerate,
                "H": None if selection.H is None else enc(selection.H),
            }
            if pose.T is not None and name in _TRIANGULATION_CASES:
                case["triangulated"] = enc(epipolar.triangulate_points(pose.T, pa, pb, KITTI_K))
        cases.append(case)

    add("forward", forward_motion(0.9))
    add("forward_yaw2", forward_motion(0.9, yaw_deg=2.0))
    add("forward_yaw-3_lat", forward_motion(0.9, yaw_deg=-3.0, lateral=0.1))
    add("forward_2.5_yaw5", forward_motion(2.5, yaw_deg=5.0))
    add("forward_yaw1.5", forward_motion(0.9, yaw_deg=1.5))
    add("forward_yaw1_ransac", forward_motion(0.9, yaw_deg=1.0), method="ransac")
    add("forward_yaw1_usac_accurate", forward_motion(0.9, yaw_deg=1.0), method="usac_accurate")
    add("forward_yaw1_lmeds", forward_motion(0.9, yaw_deg=1.0), method="lmeds")
    add("noise0.5_t1.5", forward_motion(0.9, yaw_deg=1.0), threshold=1.5, noise_px=0.5)
    add("noise0.7_yaw2", forward_motion(0.9, yaw_deg=2.0), threshold=1.5, noise_px=0.7)
    add("pure_rotation3", pure_rotation(3.0))
    add("pure_rotation0.5", pure_rotation(0.5))
    add("stationary", forward_motion(0.0))
    add("yaw10_rotguard", forward_motion(0.9, yaw_deg=10.0),
        recover_kwargs={"max_rotation_deg": 2.0})
    add("loop_like", forward_motion(3.0, yaw_deg=4.0), threshold=1.0,
        recover_kwargs={"min_inlier_ratio": 0.0, "max_rotation_deg": 180.0,
                        "min_inliers": 15}, noise_px=0.2)
    planar = make_scene(600, seed=2)
    planar[:, 2] = 20.0 + 0.5 * planar[:, 0]
    add("planar", forward_motion(0.9, yaw_deg=1.0), points_world=planar)
    write("epipolar", {"cases": cases, "K": enc(KITTI_K)})


# --------------------------------------------------------------------------- #
# Features, loop closure
# --------------------------------------------------------------------------- #


def _keypoints(frame):
    return enc(np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave,
                          kp.class_id] for kp in frame.keypoints], dtype=np.float64))


def export_features() -> None:
    from kitti_fixture import make_textured_image
    from monocular_slam.features.detector import FeatureDetector
    from monocular_slam.features.matcher import FeatureMatcher

    image = make_textured_image(0, width=480, height=200)
    shifted = np.zeros_like(image)
    shifted[:, :-10] = image[:, 10:]
    detections = {}
    for name, detector in [
        ("orb500", FeatureDetector("orb", max_features=500)),
        ("orb40", FeatureDetector("orb", max_features=40)),
        ("orb300_clahe", FeatureDetector("orb", max_features=300, clahe=True)),
        ("orb_fast_score", FeatureDetector("orb", max_features=500, score_type=1)),
        ("sift200", FeatureDetector("sift", max_features=200)),
    ]:
        for label, img in (("image", image), ("shifted", shifted)):
            if label == "shifted" and name != "orb500":
                continue
            frame = detector.detect(img)
            detections[f"{name}/{label}"] = {
                "keypoints": _keypoints(frame),
                "descriptors": None if frame.descriptors is None else enc(frame.descriptors),
            }
    orb = FeatureDetector("orb", max_features=500)
    f0, f1 = orb.detect(image, 0, 0), orb.detect(shifted, 1, 1)
    matches = {}
    for name, matcher in [
        ("default", FeatureMatcher()),
        ("no_ceiling", FeatureMatcher(max_distance=None)),
        ("ratio0.6_nocc", FeatureMatcher(use_ratio_test=True, ratio=0.6, cross_check=False)),
        ("loose", FeatureMatcher(use_ratio_test=False, cross_check=False)),
        ("cc_only", FeatureMatcher(use_ratio_test=False, cross_check=True)),
        ("ceiling30", FeatureMatcher(max_distance=30.0)),
    ]:
        result = matcher.match(f0, f1)
        matches[name] = {"indices_a": enc(result.indices_a), "indices_b": enc(result.indices_b),
                         "distances": enc(result.distances), "n_raw": result.n_raw}
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[:, :100] = 255
    masked = orb.detect(image, mask=mask)
    write("features", {
        "image_md5": hashlib.md5(image.tobytes()).hexdigest(),
        "detections": detections,
        "matches": matches,
        "masked": _keypoints(masked),
    })


def export_textured_images(requests) -> None:
    """Drawing parameters of kitti_fixture.make_textured_image per (seed, size)."""
    from kitti_fixture import make_textured_image

    params = {}
    renders = []
    for index, width, height, seed in requests:
        key = f"{seed}/{width}x{height}"
        if key not in params:
            rng = np.random.default_rng(seed)
            circles = []
            for _ in range(45):
                cx = int(rng.integers(10, width - 10))
                cy = int(rng.integers(10, height - 10))
                radius = int(rng.integers(3, 9))
                shade = int(rng.integers(0, 255))
                circles.append([cx, cy, radius, shade])
            rects = []
            for _ in range(12):
                x0 = int(rng.integers(0, width))
                y0 = int(rng.integers(0, height))
                shade = int(rng.integers(0, 255))
                rects.append([x0, y0, shade])
            params[key] = {"circles": circles, "rects": rects}
        image = make_textured_image(index, width=width, height=height, seed=seed)
        renders.append([index, width, height, seed, hashlib.md5(image.tobytes()).hexdigest()])
    # Fixture sequences render frames 0..39 at the default size with seed 0.
    for index in range(40):
        image = make_textured_image(index)
        renders.append([index, 320, 120, 0, hashlib.md5(image.tobytes()).hexdigest()])
    write("textured_images", {"params": params, "renders": renders})


def export_database() -> None:
    from monocular_slam.loop_closure.database import KeyframeDatabase, VisualVocabulary
    from test_loop_closure import clustered_descriptors, make_keyframe

    scene_a = clustered_descriptors(6, 40, seed=10)
    scene_b = clustered_descriptors(6, 40, seed=20)
    scene_c = clustered_descriptors(6, 40, seed=30)
    descriptor_sets = [scene_a, scene_a, scene_b, scene_c, scene_a, scene_a]
    vocab = VisualVocabulary.train(np.concatenate(descriptor_sets), vocabulary_size=24, seed=0)
    db = KeyframeDatabase(vocab)
    for i, descriptors in enumerate(descriptor_sets):
        db.add(make_keyframe(i, descriptors, distance=float(i) * 50.0))
    db.build_index()
    queries = []
    for query_id in range(len(db)):
        for sep, path, k, sim in [(2, 50.0, 3, 0.0), (1, 0.0, 5, 0.0), (3, 0.0, 5, 0.0),
                                  (2, 0.0, 5, 0.3)]:
            result = db.query(query_id, min_keyframe_separation=sep, min_path_separation_m=path,
                              top_k=k, min_similarity=sim)
            queries.append({"query": query_id, "args": [sep, path, k, sim],
                            "result": [[i, fhex(s)] for i, s in result]})
    float_rng = np.random.default_rng(0)
    float_desc = np.concatenate(
        [float_rng.normal(loc=c, scale=0.05, size=(40, 8)) for c in (-5.0, 0.0, 5.0)]
    ).astype(np.float32)
    float_vocab = VisualVocabulary.train(float_desc, vocabulary_size=3, seed=0)
    write("database", {
        "descriptor_sets": [enc(d) for d in descriptor_sets],
        "points": [enc(kf.points) for kf in db],
        "centroids": enc(vocab.centroids),
        "words": [enc(vocab.quantize(d)) for d in descriptor_sets],
        "histograms": [enc(vocab.histogram(d)) for d in descriptor_sets],
        "bow": enc(db.bow_matrix),
        "similarity": [enc(db.similarity_scores(i)) for i in range(len(db))],
        "neighbour_score": [[i, w, fhex(db.neighbour_score(i, window=w))]
                            for i in range(len(db)) for w in (1, 2, 5)],
        "queries": queries,
        "float_descriptors": enc(float_desc),
        "float_centroids": enc(float_vocab.centroids),
        "float_words": enc(float_vocab.quantize(float_desc)),
    })


def export_loop_detector() -> None:
    import test_loop_closure as tlc
    from kitti_fixture import make_textured_image
    from monocular_slam.features.detector import FeatureDetector
    from monocular_slam.features.matcher import FeatureMatcher
    from monocular_slam.loop_closure.database import KeyframeDatabase, VisualVocabulary
    from monocular_slam.loop_closure.detector import LoopCandidate, LoopClosureDetector
    from monocular_slam.geometry.transforms import se3_from_rt
    from synthetic import KITTI_K, forward_motion

    helper = tlc.TestGeometricVerification()
    matcher = FeatureMatcher(norm=cv2.NORM_HAMMING, min_matches=10)

    def detector(**overrides):
        kwargs = {"K": KITTI_K, "matcher": matcher, "min_matches": 20, "min_inliers": 15,
                  "min_inlier_ratio": 0.3}
        kwargs.update(overrides)
        return LoopClosureDetector(**kwargs)

    def database_payload(db):
        return [{"points": enc(kf.points), "descriptors": enc(kf.descriptors), "pose": enc(kf.pose),
                 "distance": fhex(kf.path_distance_m), "frame_index": kf.frame_index}
                for kf in db]

    cases = []

    def verify_case(name, db, query_id, match_id, attributes=None):
        loop_detector = detector()
        for key, value in (attributes or {}).items():
            setattr(loop_detector, key, value)
        cv2.setRNGSeed(0)
        closure = loop_detector.verify(db, LoopCandidate(query_id=query_id, match_id=match_id,
                                                         similarity=0.9))
        cases.append({
            "name": name, "database": database_payload(db), "query": query_id, "match": match_id,
            "attributes": attributes or {},
            "closure": None if closure is None else {
                "dict": closure.to_dict(), "T": enc(closure.T_match_query),
                "scale_m": fhex(closure.scale_m), "inlier_ratio": fhex(closure.inlier_ratio),
            },
            "stats": loop_detector.stats.to_dict(),
        })

    verify_case("pair_yaw4", helper.synthetic_pair(forward_motion(3.0, yaw_deg=4.0)), 1, 0)
    verify_case("pair_yaw2", helper.synthetic_pair(forward_motion(3.0, yaw_deg=2.0)), 1, 0)
    db, _ = helper.three_view_database(true_separation_m=2.0, drift_m=15.0)
    verify_case("three_view_2m", db, 2, 0)
    db, _ = helper.three_view_database(true_separation_m=3.0, drift_m=20.0)
    verify_case("three_view_3m", db, 2, 0)
    db, _ = helper.three_view_database(true_separation_m=6.0, drift_m=5.0)
    verify_case("three_view_6m_gate2", db, 2, 0, {"max_loop_scale_m": 2.0})
    verify_case("pair_unmeasured_kept", helper.synthetic_pair(forward_motion(3.0, yaw_deg=2.0)),
                1, 0, {"min_scale_points": 10_000, "require_measured_scale": False})

    # Planted revisit with a vocabulary fixed from Python, so the whole detect()
    # cascade can be compared exactly.
    orb = FeatureDetector("orb", max_features=800)
    images = ([make_textured_image(i, seed=1) for i in range(4)]
              + [make_textured_image(0, seed=200 + i) for i in range(16)]
              + [make_textured_image(i, seed=1) for i in range(4)])
    planted = KeyframeDatabase()
    for i, image in enumerate(images):
        frame = orb.detect(image, i, i)
        planted.add(tlc.make_keyframe(i, frame.descriptors, points=frame.points,
                                      pose=se3_from_rt(np.eye(3), np.array([0.0, 0.0, 5.0 * i])),
                                      distance=5.0 * i))
    sample = np.concatenate([kf.descriptors for kf in planted])
    vocab = VisualVocabulary.train(sample, vocabulary_size=64, seed=0)
    planted.set_vocabulary(vocab)
    loop_detector = LoopClosureDetector(
        K=KITTI_K, matcher=matcher, min_keyframe_separation=10, min_path_separation_m=30.0,
        min_similarity=0.1, similarity_ratio=0.3, min_matches=20, min_inliers=10,
        min_inlier_ratio=0.2, cooldown_keyframes=1)
    cv2.setRNGSeed(0)
    closures = loop_detector.detect(planted)
    write("loop_detector", {
        "verify": cases,
        "planted": {
            "centroids": enc(vocab.centroids),
            "closures": [{"dict": c.to_dict(), "T": enc(c.T_match_query)} for c in closures],
            "stats": loop_detector.stats.to_dict(),
            "bow": enc(planted.bow_matrix),
        },
    })


# --------------------------------------------------------------------------- #
# End-to-end pipeline on the synthetic KITTI fixture
# --------------------------------------------------------------------------- #


def export_pipeline_synthetic() -> None:
    from kitti_fixture import write_kitti_sequence
    from monocular_slam.pipeline import SlamPipeline, build_metrics, write_diagnostics_csv

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = write_kitti_sequence(Path(tmp) / "dataset", "00", n_frames=40)
        base = {
            "dataset.path": str(root), "dataset.sequence": "00",
            "output.root": str(Path(tmp) / "outputs"), "runtime.log_level": "ERROR",
            "runtime.progress_interval": 0, "features.max_features": 400,
            "matcher.min_matches": 8, "odometry.min_inliers": 6,
            "odometry.min_inlier_ratio": 0.05, "loop_closure.enabled": False,
            "output.plot_3d": False, "evaluation.rpe_deltas": [1, 5],
        }
        for name, extra in [("vo_gt_scale", {}), ("vo_unit_scale", {"odometry.scale_source": "none"}),
                            ("vo_constant", {"odometry.scale_source": "constant"}),
                            ("vo_step2", {"dataset.frame_step": 2})]:
            config = Config().with_overrides({**base, **extra})
            result = SlamPipeline(config).run()
            csv_path = Path(tmp) / f"{name}.csv"
            write_diagnostics_csv(result, csv_path)
            traj_path = result.raw_trajectory.save_kitti(Path(tmp) / f"{name}.txt")
            metrics = build_metrics(result)
            for volatile in ("runtime_fps", "timing", "environment"):
                metrics.pop(volatile, None)
            metrics["dataset"]["root"] = "<root>"  # a temporary directory
            # Per-frame timings vary run to run; blank them so the fixture is
            # stable (the C++ test skips these columns).
            rows = list(csv.DictReader(io.StringIO(csv_path.read_text(), newline="")))
            for row in rows:
                for column in ("detect_ms", "match_ms", "pose_ms"):
                    row[column] = ""
            scrubbed = io.StringIO(newline="")
            writer = csv.DictWriter(scrubbed, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            results[name] = {
                "overrides": {**base, **extra},
                "diagnostics_csv": scrubbed.getvalue(),
                "trajectory_txt": traj_path.read_text(),
                "trajectory": enc(result.raw_trajectory.poses),
                "metrics_json": json.dumps(metrics, indent=2),
            }
    for payload in results.values():
        payload["overrides"].pop("dataset.path")
        payload["overrides"].pop("output.root")
    write("pipeline_synthetic", results)


# --------------------------------------------------------------------------- #
# KITTI references
# --------------------------------------------------------------------------- #


def export_kitti(reference_dir: Path) -> None:
    out = GOLDEN / "kitti"
    out.mkdir(parents=True, exist_ok=True)
    copies = {
        "sequence_00_300": ["metrics.json", "diagnostics.csv", "trajectory_raw.txt",
                            "config.resolved.yaml"],
        "sequence_05_300": ["metrics.json", "diagnostics.csv", "trajectory_raw.txt",
                            "config.resolved.yaml"],
        "sequence_00": ["metrics.json"],
        "sequence_01": ["metrics.json"],
        "sequence_05": ["metrics.json"],
    }
    # Full-sequence trajectories are stored as camera positions only: the
    # 12-column files would push the fixtures past their size budget, and the
    # end-to-end comparison is on positions (ATE) anyway.
    full_positions = {"sequence_00": ["trajectory_raw.txt", "trajectory_optimized.txt"],
                      "sequence_01": ["trajectory_raw.txt"],
                      "sequence_05": ["trajectory_raw.txt"]}
    for run, names in copies.items():
        source = reference_dir / run
        if not source.is_dir():
            print(f"  (skipping {run}: {source} not found)")
            continue
        (out / run).mkdir(exist_ok=True)
        for name in names:
            if (source / name).is_file():
                shutil.copy2(source / name, out / run / name)
        for name in full_positions.get(run, []):
            if (source / name).is_file():
                rows = np.loadtxt(source / name, dtype=np.float64)
                xyz = rows[:, [3, 7, 11]]
                np.savetxt(out / run / name.replace(".txt", "_xyz.txt"), xyz, fmt="%.9e")
        print(f"  copied {run}: {', '.join(names + full_positions.get(run, []))}")


def export_metadata() -> None:
    import gtsam  # noqa: F401
    import scipy

    def pip_version(package: str) -> str:
        try:
            from importlib.metadata import version

            return version(package)
        except Exception:  # pragma: no cover - metadata lookup is best effort
            return "unknown"

    write("metadata", {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cv2": cv2.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "gtsam": pip_version("gtsam"),
        "pyyaml": yaml.__version__,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                     cwd=REPO).stdout.strip(),
    })


def record_test_data() -> list:
    """Run the Python suite under the recorder plugin."""
    out = GOLDEN / "testdata"
    if out.exists():
        shutil.rmtree(out)
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p",
               "golden_recorder", "--golden-out", str(out), str(REPO / "tests")]
    env = {**__import__("os").environ, "PYTHONPATH": str(REPO / "tools")}
    completed = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    print("  " + completed.stdout.strip().splitlines()[-1])
    if completed.returncode != 0:
        print(completed.stdout[-4000:])
        raise SystemExit("Python test suite failed while recording golden data")
    return json.loads((out / "textured_image_requests.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kitti-ref", type=Path, default=None,
                        help="Directory with Python run outputs (sequence_00_300, sequence_00, ...)")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Do not re-record per-test random data")
    args = parser.parse_args()

    GOLDEN.mkdir(parents=True, exist_ok=True)
    print("Recording per-test data")
    if args.skip_tests:
        requests = json.loads((GOLDEN / "testdata" / "textured_image_requests.json").read_text())
    else:
        requests = record_test_data()
    print("Exporting module goldens")
    export_metadata()
    export_compat()
    export_config()
    export_transforms()
    export_trajectory_and_metrics()
    export_pose_graph()
    export_epipolar()
    export_features()
    export_textured_images(requests)
    export_database()
    export_loop_detector()
    export_pipeline_synthetic()
    if args.kitti_ref is not None:
        print("Copying KITTI references")
        export_kitti(args.kitti_ref)
    total = sum(p.stat().st_size for p in GOLDEN.rglob("*") if p.is_file())
    print(f"Total golden size: {total / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
