"""pytest plugin that records the random test data every Python test consumes.

The C++ port cannot reproduce NumPy's PCG64 streams, so instead of re-deriving
random inputs it *replays* them: this plugin wraps every ``np.random.Generator``
that test code creates (the ``rng`` fixture, ``np.random.default_rng(seed)``
calls inside tests and helpers) plus the seeded helper functions, and writes
what they returned, in call order, per pytest node id.

Library code (``src/monocular_slam``) is deliberately *not* recorded: its own
RNG use (vocabulary seeding, descriptor subsampling) is ported algorithmically
with ``std::mt19937_64`` and validated statistically instead.

Usage (normally via tools/export_golden.py)::

    pytest -p golden_recorder --golden-out tests/golden/testdata

Output: one ``<test_module>.json`` per test module with

* ``streams``: ``{node_id: [record, ...]}`` where a record is
  ``{"fn": name, "value": encoded}``. Primitive generator draws are named after
  the method (``normal``, ``uniform``, ``integers``, ``random``); recorded
  helpers by their function name. Draws made *inside* a recorded helper are
  not recorded separately, so each stream holds exactly what the test itself
  saw.
* ``shared``: large helper outputs (``make_scene``, ``clustered_descriptors``)
  stored once, keyed by their arguments, and referenced from the streams as
  ``{"ref": key}``.

Arrays are encoded as ``{"dtype", "shape", "b64"}`` (little-endian raw bytes),
which is exact and roughly half the size of decimal text.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TESTS_DIR = str(REPO / "tests")

_DTYPES = {
    np.dtype(np.float64): "f8",
    np.dtype(np.float32): "f4",
    np.dtype(np.int64): "i8",
    np.dtype(np.int32): "i4",
    np.dtype(np.uint8): "u1",
    np.dtype(np.bool_): "b1",
}

_REAL_DEFAULT_RNG = np.random.default_rng

_state = {
    "current": None,  # list of records for the running test, or None
    "depth": 0,  # >0 while inside a recorded/silent helper
    "shared": {},  # module -> {key: encoded}
    "streams": {},  # module -> {nodeid: [records]}
    "module": None,
    "textured_images": set(),  # (index, width, height, seed) render requests
}


#: Arrays whose base64 payload exceeds this many characters are stored once in
#: the module's content-addressed ``shared`` table: fixtures hand the same data
#: to every test in a class, and storing it per test would multiply the size.
_SHARE_THRESHOLD = 512


def encode(value, share: bool = True):
    """Encode a NumPy value (or nested tuple of them) for JSON."""
    if isinstance(value, tuple):
        return {"tuple": [encode(v, share) for v in value]}
    if isinstance(value, np.generic):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        if arr.dtype not in _DTYPES:
            raise TypeError(f"golden_recorder cannot encode dtype {arr.dtype}")
        if arr.ndim == 0:
            # JSON numbers written with repr() round-trip float64 exactly.
            return {"dtype": _DTYPES[arr.dtype], "shape": [], "v": arr.item()}
        little = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
        encoded = {
            "dtype": _DTYPES[arr.dtype],
            "shape": list(arr.shape),
            "b64": base64.b64encode(little.tobytes()).decode("ascii"),
        }
        if share and len(encoded["b64"]) > _SHARE_THRESHOLD and _state["module"] is not None:
            key = "sha1:" + hashlib.sha1(json.dumps(encoded, sort_keys=True).encode()).hexdigest()[:20]
            _state["shared"].setdefault(_state["module"], {})[key] = encoded
            return {"ref": key}
        return encoded
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    raise TypeError(f"golden_recorder cannot encode {type(value).__name__}")


def _record(fn: str, value) -> None:
    current = _state["current"]
    if current is None or _state["depth"] > 0:
        return
    current.append({"fn": fn, "value": encode(value)})


def _caller_is_test_code(depth: int = 2) -> bool:
    filename = sys._getframe(depth).f_code.co_filename
    return filename.startswith(TESTS_DIR)


class RecordingGenerator:
    """Proxy around ``np.random.Generator`` that records every draw."""

    def __init__(self, generator: np.random.Generator) -> None:
        self._generator = generator

    def __getattr__(self, name: str):
        attr = getattr(self._generator, name)
        if not callable(attr):
            return attr

        @functools.wraps(attr)
        def wrapper(*args, **kwargs):
            result = attr(*args, **kwargs)
            _record(name, result)
            return result

        return wrapper


def _patched_default_rng(*args, **kwargs):
    generator = _REAL_DEFAULT_RNG(*args, **kwargs)
    if _caller_is_test_code():
        return RecordingGenerator(generator)
    return generator


def _recorded_helper(module_name: str, func, *, shared: bool = False, silent: bool = False):
    """Wrap a helper so its *output* is recorded and its internals are not."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _state["depth"] += 1
        try:
            result = func(*args, **kwargs)
        finally:
            _state["depth"] -= 1
        if silent:
            if func.__name__ == "make_textured_image":
                bound = _bind(func, args, kwargs)
                _state["textured_images"].add(
                    (bound["index"], bound["width"], bound["height"], bound["seed"])
                )
            return result
        if shared:
            key = f"{func.__name__}{_args_key(func, args, kwargs)}"
            table = _state["shared"].setdefault(_state["module"], {})
            if key not in table:
                table[key] = encode(result, share=False)
            current = _state["current"]
            if current is not None and _state["depth"] == 0:
                current.append({"fn": func.__name__, "value": {"ref": key}})
        else:
            _record(func.__name__, result)
        return result

    wrapper.__golden_wrapped__ = True
    return wrapper


def _bind(func, args, kwargs) -> dict:
    import inspect

    bound = inspect.signature(func).bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _args_key(func, args, kwargs) -> str:
    arguments = _bind(func, args, kwargs)
    parts = []
    for name, value in arguments.items():
        if isinstance(value, np.ndarray):
            value = f"array{list(value.shape)}"
        parts.append(f"{name}={value!r}")
    return "(" + ",".join(parts) + ")"


#: helper name -> (shared, silent)
HELPERS = {
    "random_se3": (False, False),
    "small_se3": (False, False),
    "make_scene": (True, False),
    "clustered_descriptors": (True, False),
    "make_textured_image": (False, True),
}


def _patch_module(module) -> None:
    for name, (shared, silent) in HELPERS.items():
        func = getattr(module, name, None)
        if func is None or getattr(func, "__golden_wrapped__", False):
            continue
        setattr(module, name, _recorded_helper(module.__name__, func, shared=shared, silent=silent))


# --------------------------------------------------------------------------- #
# pytest hooks
# --------------------------------------------------------------------------- #


def pytest_addoption(parser):
    parser.addoption("--golden-out", default=None, help="directory for recorded test data")


def pytest_configure(config):
    np.random.default_rng = _patched_default_rng


def pytest_collection_finish(session):
    # Test modules bind helpers at import (``from conftest import random_se3``),
    # so every module that has them gets patched after collection. Helpers
    # calling helpers (two_view_correspondences -> make_scene) resolve through
    # their defining module's globals, which are patched too.
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None) or ""
        if filename.startswith(TESTS_DIR):
            _patch_module(module)


def pytest_runtest_protocol(item, nextitem):
    module = Path(item.fspath).stem
    _state["module"] = module
    records: list = []
    _state["current"] = records
    _state["streams"].setdefault(module, {})[_node_key(item)] = records
    return None  # let pytest run the protocol as usual


def pytest_runtest_logfinish(nodeid, location):
    _state["current"] = None


def _node_key(item) -> str:
    # "test_transforms.py::TestAngles::test_rotation_angle_recovers_input[30.0]"
    return item.nodeid.split("/", 1)[-1] if item.nodeid.startswith("tests/") else item.nodeid


def pytest_sessionfinish(session, exitstatus):
    out = session.config.getoption("--golden-out")
    if not out:
        return
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for module, streams in sorted(_state["streams"].items()):
        payload = {
            "module": module,
            "streams": {k: v for k, v in streams.items() if v},
            "shared": _state["shared"].get(module, {}),
        }
        (out_dir / f"{module}.json").write_text(
            json.dumps(payload, indent=None, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
    requests = sorted(_state["textured_images"])
    (out_dir / "textured_image_requests.json").write_text(
        json.dumps([list(r) for r in requests]) + "\n", encoding="utf-8"
    )
