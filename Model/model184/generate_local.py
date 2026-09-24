#!/usr/bin/env python3
"""Generate the "biohub_synthetic" TIME SEQUENCES locally with the author's own generator code
(model184/generator_src.py, CC0, Jose Freitas, notebook josefreitasalvesneto/biohub-synthetic-dataset)
instead of downloading the 18.5 GB dataset.

Output (exact published format, np.savez, not compressed):
    <out>/seq_XXXX.npz   volumes (T,64,64,64) uint16 | nodes (M,5) float32 [t,z,y,x,track_id]
                         edges (E,2) int32 | divisions (D,) int32 | voxel_um_pooled (3,) float32
    <out>/../manifest.json, metadata.json, calibration.json

Usage:
    python model184/generate_local.py --n-seq 400 --out C:\\biohub_data\\work\\model184\\synthetic\\sequences --seed 0 --workers 4
Resumable: existing seq_XXXX.npz (and seq_XXXX.skip markers) are skipped; files are written to a temp
name and renamed, so a killed run never leaves a truncated npz behind. CPU only.
"""
from __future__ import annotations

import os

# one compute thread per worker process (numpy/BLAS must not oversubscribe the 4-process budget)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# the published build's explicit generator knobs (notebook cell 0); both equal the code defaults
os.environ.setdefault("SYNTH_PLACE_MODE", "shell")
os.environ.setdefault("SYNTH_EDGE_FRAC", "0.35")

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generator_src as gen  # noqa: E402

_STATE: dict = {}


def _jsonable_shell(shell):
    if shell is None:
        return None
    return {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in shell.items()}


def _shell_from_json(d):
    if d is None:
        return None
    return {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in d.items()}


def load_or_run_calibration(path: Path, train_dir: Path | None, recalibrate: bool):
    if path.exists() and not recalibrate:
        c = json.loads(path.read_text())
        print(f"calibration: reusing {path}", flush=True)
        return c["videos"], np.array(c["counts"], float), _shell_from_json(c["shell"]), c
    t0 = time.time()
    td = gen.find_train_dir() if train_dir is None else train_dir
    assert td, "train directory not found"
    videos, counts, shell = gen.calibrate(td)
    c = {"train_dir": str(td), "videos": videos, "frames_per_video": [0, 1],
         "counts": counts.tolist(), "shell": _jsonable_shell(shell), "seconds": time.time() - t0}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c, indent=2, default=float) + "\n")
    # json round-trip of float64 is exact (repr), so a resumed run sees bit-identical calibration
    return videos, counts, _shell_from_json(c["shell"]), c


def _init(counts, shell, out, T, div_p, seed):
    _STATE.update(counts=np.asarray(counts, float), shell=shell, out=Path(out), T=T, div_p=div_p, seed=seed)


def _work(j: int):
    t0 = time.time()
    out: Path = _STATE["out"]
    fp = out / f"seq_{j:04d}.npz"
    seq = gen.generate_sequence(j, _STATE["counts"], _STATE["shell"], T_SEQ=_STATE["T"],
                                div_p=_STATE["div_p"], seed=_STATE["seed"])
    if seq is None:                                   # the notebook skips this index
        (out / f"seq_{j:04d}.skip").write_text("len(pos) < 10\n")
        return j, None, time.time() - t0
    tmp = out / f"seq_{j:04d}.npz.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, **seq)
    os.replace(tmp, fp)
    return j, _entry(fp, seq), time.time() - t0


def _entry(fp: Path, seq=None):
    if seq is None:                                   # existing file: read the small tables only (lazy npz)
        with np.load(fp) as d:
            seq = {"nodes": d["nodes"], "edges": d["edges"], "divisions": d["divisions"]}
        T = int(round(float(seq["nodes"][:, 0].max()))) + 1 if len(seq["nodes"]) else 0
    else:
        T = int(seq["volumes"].shape[0])
    return dict(file=f"sequences/{fp.name}", T=T, n_nodes=int(len(seq["nodes"])),
                n_edges=int(len(seq["edges"])), n_divisions=int(len(seq["divisions"])),
                bytes=int(fp.stat().st_size))


def write_manifest(root: Path, out: Path, args, calib: dict, counts, shell):
    entries = [_entry(fp) for fp in sorted(out.glob("seq_*.npz"))]
    nd = sum(m["n_divisions"] for m in entries); nn = sum(m["n_nodes"] for m in entries)
    total = sum(m["bytes"] for m in entries)
    (root / "manifest.json").write_text(json.dumps({"static": [], "sequences": entries}, indent=2) + "\n")
    meta = dict(
        n_static=0, n_sequences=len(entries), total_gb=total / 1024**3, seq_len=args.seq_len,
        total_divisions=nd, total_nodes=nn, division_rate=nd / max(nn, 1),
        voxel_native_um=[1.625, 0.40625, 0.40625],
        pooling="stride 4x in XY (vol[:, ::4, ::4]) -- identical to the official evaluator",
        real_count_range=[float(counts.min()), float(counts.max())],
        shell_kind=(shell or {}).get("kind"),
        motion_calibration=dict(median_step_um_per_frame=gen.STEP_MED, lag1_persistence=gen.PERSIST,
                                sister_separation_um=gen.SISTER),
        note_division_rate="Divisions are deliberately over-sampled; the real ground-truth rate is "
                           "about 0.26% of nodes. Re-weight your loss if you need calibrated priors.",
        config={k: v for k, v in os.environ.items() if k.startswith("SYNTH_")},
        local_generation=dict(generator="model184/generator_src.py (josefreitasalvesneto/biohub-synthetic-dataset, CC0)",
                              seed=args.seed, div_rate=args.div_rate, calibration_videos=calib["videos"],
                              skipped=sorted(p.stem for p in out.glob("seq_*.skip"))),
    )
    (root / "metadata.json").write_text(json.dumps(meta, indent=2, default=float) + "\n")
    return len(entries), total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seq", type=int, required=True, help="generate sequence indices start..start+N-1")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(r"C:\biohub_data\work\model184\synthetic\sequences"))
    ap.add_argument("--seed", type=int, default=0,
                    help="0 = the notebook's own per-sequence seeds (500000+j / 900000+j*97+t); "
                         "any other value gives an independent stream default_rng([seed, base])")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=int(os.environ.get("DS_SEQ_LEN", "6")))
    ap.add_argument("--div-rate", type=float, default=float(os.environ.get("DS_DIV_RATE", "0.05")))
    ap.add_argument("--train-dir", type=Path, default=None)
    ap.add_argument("--recalibrate", action="store_true")
    args = ap.parse_args()
    if args.workers > 4:
        ap.error("at most 4 worker processes on this machine")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    root = out.parent
    for stale in out.glob("seq_*.npz.tmp*"):
        stale.unlink()
    videos, counts, shell, calib = load_or_run_calibration(root / "calibration.json", args.train_dir,
                                                           args.recalibrate)
    print(f"=== LOCAL SEQUENCES | calib {videos[0]}..{videos[-1]} ({len(videos)} movies) | real density "
          f"{counts.min():.0f}-{counts.max():.0f} (n={len(counts)}) | shell {shell['kind'] if shell else '?'} "
          f"| T={args.seq_len} div={args.div_rate} seed={args.seed} ===", flush=True)

    want = list(range(args.start, args.start + args.n_seq))
    todo = [j for j in want if not (out / f"seq_{j:04d}.npz").exists() and not (out / f"seq_{j:04d}.skip").exists()]
    print(f"{len(want) - len(todo)} of {len(want)} already present, {len(todo)} to generate, "
          f"{args.workers} workers", flush=True)
    t0 = time.time(); done = 0; cpu = 0.0
    if todo:
        init = (counts, shell, str(out), args.seq_len, args.div_rate, args.seed)
        if args.workers <= 1:
            _init(*init); it = map(_work, todo)
        else:
            pool = Pool(args.workers, initializer=_init, initargs=init)
            it = pool.imap_unordered(_work, todo)
        for j, entry, secs in it:
            done += 1; cpu += secs
            wall = time.time() - t0
            eta = wall / done * (len(todo) - done)
            msg = "SKIPPED (<10 cells)" if entry is None else (
                f"{entry['n_nodes']} nodes {entry['n_divisions']} div {entry['bytes']} B")
            print(f"[{done}/{len(todo)}] seq_{j:04d}: {msg} | {secs:.1f}s worker | wall {wall:.0f}s "
                  f"({wall / done:.1f} s/seq) | eta {eta / 60:.1f} min", flush=True)
            if done % 25 == 0:
                write_manifest(root, out, args, calib, counts, shell)
        if args.workers > 1:
            pool.close(); pool.join()
    n, total = write_manifest(root, out, args, calib, counts, shell)
    wall = time.time() - t0
    print(f"DONE: {done} generated in {wall:.0f}s wall ({wall / max(done, 1):.1f} s/seq wall, "
          f"{cpu / max(done, 1):.1f} s/seq per worker) | {n} sequences on disk, {total / 2**20:.1f} MiB "
          f"({total / max(n, 1):.0f} B/seq) -> {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
