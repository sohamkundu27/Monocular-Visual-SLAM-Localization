# Python reference implementation

This is the original Python implementation of the monocular visual SLAM pipeline. The C++17
port at the repository root replaced it as the main build. It is kept, unchanged apart from
paths, as the numerical oracle the port is checked against: `tools/export_golden.py` runs it to
produce the fixtures in `tests/golden/`.

See the [top-level README](../README.md) for the algorithms, results and the C++ build.

```bash
# from the repository root
python3 -m venv .venv
.venv/bin/pip install -r legacy_python/requirements.txt
.venv/bin/pip install -e legacy_python

# tests (synthetic data only, no KITTI download needed)
cd legacy_python && ../.venv/bin/python -m pytest -q && cd ..

# run from the repository root so configs/kitti.yaml and outputs/ resolve
.venv/bin/python legacy_python/scripts/run_slam.py -s 00 --dataset-path /path/to/KITTI/dataset
```

The configuration (`configs/kitti.yaml`) is shared with the C++ build and stays at the
repository root.
