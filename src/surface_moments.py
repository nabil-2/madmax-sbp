"""Reusable projections of measured-face height/tilt powers.

These are frequency-independent spatial integrals, not fitted stack responses.
The retained Taylor terms are checked against direct sampled-grid projection.
"""

from pathlib import Path
import hashlib
import json
import math
import os
import shutil

import numpy as np

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "data/3d/surface_moments"
HEIGHT_ORDER = 10
TOTAL_ORDER = 30


def powers(total_order=TOTAL_ORDER, height_order=HEIGHT_ORDER):
    return np.asarray([
        (p, n - r - p, r)
        for n in range(1, total_order + 1)
        for r in range(1, min(n, height_order) + 1)
        for p in range(n - r + 1)
    ], dtype=int)


def cache_spec(model, surfaces):
    return dict(
        version=1, total_order=TOTAL_ORDER, height_order=HEIGHT_ORDER,
        surface_sha256=surfaces.fingerprint, radius_m=model.radius,
        arrays={key: model.meta["arrays"][key]["sha256"]
                for key in ("basis_fields", "gram", "x_m", "y_m")},
    )


def cache_path(model, surfaces, directory=DEFAULT_CACHE):
    spec = cache_spec(model, surfaces)
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return Path(directory) / key, spec


def build_surface_moments(model, surfaces, *, directory=DEFAULT_CACHE, backend="numpy"):
    """Prepare on CPU, or use optional PyTorch/CUDA for the fixed integrals."""
    path, spec = cache_path(model, surfaces, directory)
    if (path / "manifest.json").is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    work = path.with_name(path.name + f".building-{os.getpid()}")
    work.mkdir(exist_ok=True)
    pqrs = powers()
    B = model.Binside
    dual = np.linalg.solve(model.gram, B.conj().T)
    uniform = model.uniform[model.aperture]
    xy = np.column_stack([model.inside_x, model.inside_y])
    xp = np.array([(xy[:, 0] / model.radius)**p / float(math.factorial(p))
                   for p in range(TOTAL_ORDER + 1)])
    yp = np.array([(xy[:, 1] / model.radius)**p / float(math.factorial(p))
                   for p in range(TOTAL_ORDER + 1)])
    if backend == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA surface-moment preparation needs an available GPU")
        ddual = torch.as_tensor(dual, dtype=torch.complex128, device="cuda")
        dB = torch.as_tensor(B, dtype=torch.complex128, device="cuda")
        du = torch.as_tensor(uniform, device="cuda")
        chunk = 48
        print("Surface projection GPU:", torch.cuda.get_device_name(0), flush=True)
    else:
        chunk = 16
    for face in range(6):
        h = surfaces.evaluate(face // 2, face % 2, xy) / model.radius
        hp = np.array([h**r / float(math.factorial(r)) for r in range(HEIGHT_ORDER + 1)])
        matrices = np.lib.format.open_memmap(
            work / f"face_{face}_matrices.npy", mode="w+", dtype=complex,
            shape=(len(pqrs), model.n, model.n),
        )
        sources = np.lib.format.open_memmap(
            work / f"face_{face}_sources.npy", mode="w+", dtype=complex,
            shape=(len(pqrs), model.n),
        )
        for lo in range(0, len(pqrs), chunk):
            hi = min(lo + chunk, len(pqrs))
            p, q, r = pqrs[lo:hi].T
            w = xp[p] * yp[q] * hp[r]
            if backend == "cuda":
                dw = torch.as_tensor(w, device="cuda")
                matrices[lo:hi] = ((ddual[None] * dw[:, None]) @ dB).cpu().numpy()
                sources[lo:hi] = ((dw * du).to(dB.dtype) @ dB.conj()).cpu().numpy()
            else:
                matrices[lo:hi] = (dual[None] * w[:, None]) @ B
                sources[lo:hi] = (w * uniform) @ B.conj()
        matrices.flush()
        sources.flush()
        del matrices, sources
        print(f"Prepared measured face {face + 1}/6: {len(pqrs)} moments", flush=True)
    np.save(work / "powers.npy", pqrs)
    (work / "manifest.json").write_text(json.dumps(spec, indent=2) + "\n")
    if path.exists():
        shutil.rmtree(work)
    else:
        work.rename(path)
    return path


def load_surface_moments(model, surfaces, directory=DEFAULT_CACHE):
    path, spec = cache_path(model, surfaces, directory)
    if not (path / "manifest.json").is_file():
        build_surface_moments(model, surfaces, directory=directory)
    if json.loads((path / "manifest.json").read_text()) != spec:
        raise ValueError("Measured-face projection cache has incompatible metadata")
    return (
        np.load(path / "powers.npy"),
        [np.load(path / f"face_{face}_matrices.npy", mmap_mode="r") for face in range(6)],
        [np.load(path / f"face_{face}_sources.npy", mmap_mode="r") for face in range(6)],
    )
