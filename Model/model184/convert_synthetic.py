#!/usr/bin/env python3
"""Convert the CC0 "biohub_synthetic" sequences into the competition's <stem>.zarr + <stem>.geff
layout so the UNCHANGED support-pack trainer (scripts/train_unet_transformer.py) can pretrain on them.

Source format (see model184/readme.txt for the notebook evidence), one file per sequence:
    sequences/seq_XXXX.npz
        volumes          (T=6, 64, 64, 64) uint16   already stride-pooled: native[:, ::4, ::4]
        nodes            (M, 5) float32             [t, z, y, x, track_id]; z,y,x in NATIVE voxels
                                                    (64 x 256 x 256 grid), sub-voxel floats
        edges            (E, 2) int32               row indices into `nodes`, parent(t) -> child(t+1)
        divisions        (D,)   int32               row indices of nodes with two outgoing edges
        voxel_um_pooled  (3,)   float32             [1.625, 1.625, 1.625]

Target format (identical to data/raw/train/<movie>.zarr|.geff):
    <stem>.zarr   zarr v3 group; array "0" (T, 64, 256, 256) uint16, chunks (1, 64, 256, 256),
                  blosc/zstd/clevel1/bitshuffle; attrs: OME multiscales (scale 1, 1.625, .40625, .40625)
                  and image_statistics.quantiles {"0.0","0.001","0.01","0.1","0.9","0.99","0.999","1.0"}
    <stem>.geff   GEFF 1.x, directed, node props t/z/y/x int64 in NATIVE voxels, no edge props,
                  axes t/z/y/x with scale + min/max, extra.estimated_number_of_nodes

Resolution decision: the trainer reads image[t:t+W, ::1, ::4, ::4].  The synthetic volumes are already
that strided read, so each pooled voxel is repeated 4x4 in (Y, X) (nearest neighbour).  Then
upsampled[:, :, ::4, ::4] == pooled EXACTLY (asserted), coordinates stay in native voxels exactly like
the real GEFFs, and downsample=(1,4,4) / voxel_size / ds_scale in the trainer are all unchanged.

Usage:
    python model184/convert_synthetic.py --src <dir with seq_*.npz> --dst C:\\biohub_data\\work\\model184\\synth --limit 500 --seed 0
    python model184/convert_synthetic.py --self-test
Keep --dst SHORT and outside OneDrive (zarr/geff temp files break the 260-char Windows path limit).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT.parent / "Data" / "public_checkpoints" / "support-pack" / "repo"

NATIVE_SCALE = (1.625, 0.40625, 0.40625)      # um per native voxel (z, y, x)
POOL = 4                                       # XY stride used by the generator and by the trainer
NATIVE_SHAPE = (64, 256, 256)
QUANTILE_KEYS = ("0.0", "0.001", "0.01", "0.1", "0.9", "0.99", "0.999", "1.0")
NODE_ID_STRIDE = 1_000_000                     # real GEFF ids look like (t+1)*1e6 + k


# --------------------------------------------------------------------------------------------------
# reading + validating one synthetic sequence
# --------------------------------------------------------------------------------------------------
def load_sequence(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as d:
        missing = {"volumes", "nodes", "edges"} - set(d.files)
        if missing:
            raise ValueError(f"{npz_path.name}: missing arrays {sorted(missing)} (has {d.files})")
        seq = {k: d[k] for k in d.files}
    vols, nodes = seq["volumes"], np.atleast_2d(seq["nodes"])
    edges = seq["edges"].reshape(-1, 2).astype(np.int64)
    if vols.ndim != 4 or vols.dtype != np.uint16:
        raise ValueError(f"{npz_path.name}: volumes must be 4-D uint16, got {vols.shape} {vols.dtype}")
    if nodes.ndim != 2 or nodes.shape[1] != 5:
        raise ValueError(f"{npz_path.name}: nodes must be (M, 5) [t,z,y,x,track_id], got {nodes.shape}")
    if len(edges) and (edges.min() < 0 or edges.max() >= len(nodes)):
        raise ValueError(f"{npz_path.name}: edge index out of range")
    seq["nodes"], seq["edges"] = nodes, edges
    seq["divisions"] = seq.get("divisions", np.zeros(0, np.int32)).astype(np.int64).ravel()
    return seq


def resolve_layout(vol_shape: tuple[int, ...]) -> tuple[int, int]:
    """Return (y_repeat, x_repeat) needed to reach the native 64 x 256 x 256 grid."""
    _, z, y, x = vol_shape
    if z != NATIVE_SHAPE[0]:
        raise ValueError(f"unexpected Z={z}; the generator always writes Z=64")
    if (y, x) == (NATIVE_SHAPE[1] // POOL, NATIVE_SHAPE[2] // POOL):
        return POOL, POOL                       # pre-pooled (the published dataset)
    if (y, x) == NATIVE_SHAPE[1:]:
        return 1, 1                             # someone regenerated at native resolution
    raise ValueError(f"unexpected volume shape {vol_shape}")


def build_tables(seq: dict[str, np.ndarray], max_lineages: int | None, rng: np.random.Generator,
                 name: str) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    """Integer native-voxel node table + (source_id, target_id) edge table, real-GEFF id scheme."""
    nodes, edges = seq["nodes"], seq["edges"]
    T = seq["volumes"].shape[0]
    t = np.rint(nodes[:, 0]).astype(np.int64)
    if t.min() < 0 or t.max() >= T:
        raise ValueError(f"{name}: node t outside [0, {T})")
    # Real GEFFs store integer NATIVE voxel coordinates; the trainer computes coord/4 then .long().
    # Round the sub-voxel float to the nearest native voxel and clip into the volume.
    zyx = np.rint(nodes[:, 1:4]).astype(np.int64)
    zyx = np.clip(zyx, 0, np.array(NATIVE_SHAPE) - 1)
    tid = np.rint(nodes[:, 4]).astype(np.int64)  # lineage (root-cell) id; daughters inherit it

    if len(edges):
        dt = t[edges[:, 1]] - t[edges[:, 0]]
        if not np.all(dt == 1):
            raise ValueError(f"{name}: {int((dt != 1).sum())} edges do not link t -> t+1")
        if len(np.unique(edges[:, 1])) != len(edges):
            raise ValueError(f"{name}: a node has two parents (merge)")
    out_deg = np.bincount(edges[:, 0], minlength=len(nodes)) if len(edges) else np.zeros(len(nodes), int)
    if out_deg.max(initial=0) > 2:
        raise ValueError(f"{name}: out-degree > 2")
    div_from_graph = np.flatnonzero(out_deg == 2)
    if len(seq["divisions"]) and not np.array_equal(np.sort(seq["divisions"]), div_from_graph):
        raise ValueError(f"{name}: `divisions` array disagrees with out-degree-2 nodes")

    keep = np.ones(len(nodes), bool)
    if max_lineages is not None:
        roots = np.unique(tid)
        if len(roots) > max_lineages:
            keep = np.isin(tid, rng.choice(roots, size=max_lineages, replace=False))
    if len(edges):
        ekeep = keep[edges[:, 0]] & keep[edges[:, 1]]
        if max_lineages is not None and np.any(keep[edges[:, 0]] != keep[edges[:, 1]]):
            raise ValueError(f"{name}: an edge crosses lineages; track_id is not a lineage id")
        edges = edges[ekeep]

    # node ids: (t+1)*1e6 + 1-based rank inside the frame, like the organisers' files
    node_id = np.zeros(len(nodes), np.int64)
    for ti in np.unique(t[keep]):
        idx = np.flatnonzero(keep & (t == ti))
        if len(idx) >= NODE_ID_STRIDE:
            raise ValueError("too many nodes in one frame for the id scheme")
        node_id[idx] = (ti + 1) * NODE_ID_STRIDE + np.arange(1, len(idx) + 1)

    table = {"node_id": node_id[keep], "t": t[keep], "z": zyx[keep, 0], "y": zyx[keep, 1],
             "x": zyx[keep, 2], "track_id": tid[keep]}
    edge_ids = np.stack([node_id[edges[:, 0]], node_id[edges[:, 1]]], 1) if len(edges) else np.zeros((0, 2), np.int64)
    kept_div = int((np.bincount(edges[:, 0], minlength=len(nodes)) == 2).sum()) if len(edges) else 0
    stats = {"n_nodes": int(keep.sum()), "n_edges": int(len(edge_ids)), "n_divisions": kept_div,
             "n_nodes_total": int(len(nodes)),
             "max_nodes_per_frame": int(np.bincount(t[keep]).max())}
    return table, edge_ids, stats


# --------------------------------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------------------------------
def write_zarr(path: Path, volumes: np.ndarray, rep: tuple[int, int]) -> dict[str, float]:
    import zarr
    from zarr.codecs import BloscCodec, BytesCodec

    if path.exists():
        shutil.rmtree(path)
    up = volumes
    if rep != (1, 1):
        up = np.repeat(np.repeat(volumes, rep[0], axis=2), rep[1], axis=3)
        # the exactness guarantee the whole approach rests on
        if not np.array_equal(up[:, :, ::rep[0], ::rep[1]], volumes):
            raise AssertionError("strided read of the upsampled volume does not reproduce the source")
    T = up.shape[0]
    if up.shape[1:] != NATIVE_SHAPE:
        raise AssertionError(f"upsampled shape {up.shape}")

    # Quantiles over the ORIGINAL voxels (the only ones the trainer ever reads); identical to the
    # quantiles of the repeated array. Same keys as the organisers' image_statistics block.
    qv = np.quantile(volumes.astype(np.float32).ravel(), [float(k) for k in QUANTILE_KEYS])
    quantiles = {k: float(v) for k, v in zip(QUANTILE_KEYS, qv)}
    if quantiles["0.999"] <= quantiles["0.001"]:
        raise ValueError(f"{path.name}: degenerate intensity quantiles {quantiles}")

    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    arr = group.create_array(
        "0", shape=up.shape, dtype="uint16", chunks=(1, *NATIVE_SHAPE), fill_value=0,
        serializer=BytesCodec(endian="little"),
        compressors=[BloscCodec(cname="zstd", clevel=1, shuffle="bitshuffle", typesize=2, blocksize=0)],
    )
    arr[:] = up
    axes = [{"name": "T", "type": "time", "unit": "second"}] + [
        {"name": n, "type": "space", "unit": "micrometer"} for n in "ZYX"]
    group.attrs.update({
        "multiscales": [{
            "version": "0.5", "axes": axes, "name": "0",
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, *NATIVE_SCALE]}]}],
        }],
        "image_statistics": {"quantiles": quantiles},
        "synthetic": {"source": "josefreitasalvesneto/biohub-synthetic-dataset (CC0)",
                      "xy_nearest_neighbour_repeat": list(rep), "frames": int(T)},
    })
    return quantiles


def write_geff(path: Path, table: dict[str, np.ndarray], edge_ids: np.ndarray, n_frames: int,
               with_track_id: bool) -> None:
    import geff
    import polars as pl
    import tracksdata as td
    from geff_spec import Axis, PropMetadata

    if path.exists():
        shutil.rmtree(path)
    graph = td.graph.IndexedRXGraph()
    cols = ["t", "z", "y", "x"] + (["track_id"] if with_track_id else [])
    for c in cols:
        if c != "t":                              # "t" exists by default
            graph.add_node_attr_key(c, pl.Int64, default_value=-1)
    rows = [dict(zip(cols, vals)) for vals in zip(*(table[c].tolist() for c in cols))]
    graph.bulk_add_nodes(rows, indices=table["node_id"].tolist())
    if len(edge_ids):
        graph.bulk_add_edges([{"source_id": int(s), "target_id": int(t)} for s, t in edge_ids])

    scale = {"t": 1.0, "z": NATIVE_SCALE[0], "y": NATIVE_SCALE[1], "x": NATIVE_SCALE[2]}
    axes = [Axis(name=a, type="time" if a == "t" else "space",
                 min=float(table[a].min()), max=float(table[a].max()), scale=scale[a])
            for a in ("t", "z", "y", "x")]
    meta = geff.GeffMetadata(
        directed=True, axes=axes,
        node_props_metadata={c: PropMetadata(identifier=c, dtype="int64", varlength=False) for c in cols},
        edge_props_metadata={},
        # Labels are COMPLETE here, so the organisers' "estimated" count is simply the node count.
        extra={"estimated_number_of_nodes": int(len(table["node_id"])), "synthetic_frames": int(n_frames)},
    )
    graph.to_geff(path, geff_metadata=meta, overwrite=True)   # same call as biohub_tracking.io.save_graph

    # tracksdata hard-codes t as Int32, so the array lands on disk as int32 while the organisers'
    # files (and our metadata) say int64.  Rewrite that one array so the file is dtype-identical.
    import zarr
    root = zarr.open_group(str(path), mode="r+")
    props = root["nodes/props/t"]
    t_vals = np.asarray(props["values"][:]).astype(np.int64)
    if props["values"].dtype != np.int64:
        del props["values"]
        props.create_array("values", data=t_vals, chunks=(max(1, len(t_vals)),))
    geff_attrs = dict(root.attrs["geff"])             # write_arrays restates the dtype it actually wrote
    geff_attrs["node_props_metadata"]["t"]["dtype"] = "int64"
    root.attrs["geff"] = geff_attrs


def convert_one(npz_path: Path, dst: Path, stem: str, max_lineages: int | None,
                rng: np.random.Generator, with_track_id: bool) -> dict[str, object]:
    seq = load_sequence(npz_path)
    rep = resolve_layout(seq["volumes"].shape)
    table, edge_ids, stats = build_tables(seq, max_lineages, rng, npz_path.name)
    quantiles = write_zarr(dst / f"{stem}.zarr", seq["volumes"], rep)
    write_geff(dst / f"{stem}.geff", table, edge_ids, seq["volumes"].shape[0], with_track_id)
    return {"stem": stem, "source": npz_path.name, "frames": int(seq["volumes"].shape[0]),
            "q0.001": quantiles["0.001"], "q0.999": quantiles["0.999"], **stats}


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def convert_many(src: Path, dst: Path, limit: int | None, seed: int, max_lineages: int | None,
                 with_track_id: bool, n_test: int, overwrite: bool) -> dict[str, object]:
    files = sorted(src.glob("seq_*.npz")) or sorted((src / "sequences").glob("seq_*.npz"))
    if not files:
        raise FileNotFoundError(f"no seq_*.npz under {src} (or {src / 'sequences'})")
    rng = np.random.default_rng(seed)
    if limit is not None and limit < len(files):
        files = [files[i] for i in sorted(rng.choice(len(files), size=limit, replace=False))]
    dst.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, f in enumerate(files):
        stem = f"syn_{f.stem.split('_')[-1]}"          # "syn" plays the role of the embryo prefix
        if (dst / f"{stem}.geff").exists() and (dst / f"{stem}.zarr").exists() and not overwrite:
            print(f"[{i + 1}/{len(files)}] {stem} exists, skipping (use --overwrite)", flush=True)
            continue
        row = convert_one(f, dst, stem, max_lineages, rng, with_track_id)
        rows.append(row)
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(files):
            print(f"[{i + 1}/{len(files)}] {stem}: {row['n_nodes']} nodes, {row['n_edges']} edges, "
                  f"{row['n_divisions']} div, max/frame {row['max_nodes_per_frame']} | "
                  f"{time.time() - t0:.0f}s", flush=True)
    stems = sorted(p.stem for p in dst.glob("syn_*.zarr") if (dst / f"{p.stem}.geff").exists())
    # Trainer-format splits file: a list of folds, each {"train": [...], "test": [...]}.
    # The `test` list only feeds the trainer's mandatory internal loader; with evaluate() patched to a
    # no-op (model156 policy) it never influences the checkpoint, so it is taken FROM the train list.
    splits = [{"train": stems, "test": stems[:max(1, n_test)]}]
    (dst / "synthetic_splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    summary = {"converted_now": len(rows), "movies_in_dst": len(stems), "seed": seed,
               "max_lineages": max_lineages, "bytes_on_disk": dir_bytes(dst),
               "max_nodes_per_frame": max((r["max_nodes_per_frame"] for r in rows), default=0),
               "divisions": sum(r["n_divisions"] for r in rows), "rows": rows}
    (dst / "conversion_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# --------------------------------------------------------------------------------------------------
# self-test: fabricate one fake seq_0000.npz in the generator's exact format, convert, load it back
# --------------------------------------------------------------------------------------------------
def fabricate_npz(path: Path, seed: int = 0, T: int = 6, n_cells: int = 24, div_p: float = 0.15) -> dict:
    """Mimics run_dataset_build() of the generator notebook: same arrays, dtypes, units, id logic."""
    rng = np.random.default_rng(seed)
    lo, hi = np.array([4, 12, 12], float), np.array([59, 243, 243], float)
    tracks = [dict(pos=rng.uniform(lo, hi), tid=k, parent=None, is_div=False) for k in range(n_cells)]
    nodes, edges, divs, prev_ids, nid = [], [], [], {}, 0
    vols = np.zeros((T, 64, 64, 64), np.uint16)
    zz, yy, xx = np.meshgrid(np.arange(64), np.arange(0, 256, 4), np.arange(0, 256, 4), indexing="ij")
    for t in range(T):
        v = np.clip(rng.normal(0.04, 0.01, (64, 64, 64)), 0, 1)                 # dark medium
        for tr in tracks:                                                        # blobs on the NATIVE grid,
            p = tr["pos"]                                                        # sampled at the stride
            r2 = ((zz - p[0]) * 1.625) ** 2 + ((yy - p[1]) * 0.40625) ** 2 + ((xx - p[2]) * 0.40625) ** 2
            v = np.maximum(v, 0.8 * np.exp(-(r2 / 4.5 ** 2) ** 2))
        v = np.clip(rng.poisson(v * 300) / 300 + rng.normal(0, 0.006, v.shape), 0, 1)
        vols[t] = np.clip(v * 65535.0, 0, 65535).astype(np.uint16)
        cur = {}
        for k, tr in enumerate(tracks):
            nodes.append([t, *tr["pos"], tr["tid"]]); cur[k] = nid; nid += 1
        for k, tr in enumerate(tracks):
            if tr["parent"] is not None and tr["parent"] in prev_ids:
                edges.append([prev_ids[tr["parent"]], cur[k]])
                if tr["is_div"]:
                    divs.append(prev_ids[tr["parent"]])
        prev_ids = cur
        if t == T - 1:
            break
        nxt = []
        for k, tr in enumerate(tracks):
            step = rng.normal(0, 1.3, 3) / np.array([1.625, 0.40625, 0.40625])
            if rng.random() < div_p:
                d = rng.normal(0, 1, 3); d /= np.linalg.norm(d) + 1e-9
                half = 0.5 * 7.24 * d / np.array([1.625, 0.40625, 0.40625])
                for s in (1, -1):
                    nxt.append(dict(pos=np.clip(tr["pos"] + step + s * half, lo, hi), tid=tr["tid"], parent=k, is_div=True))
            else:
                nxt.append(dict(pos=np.clip(tr["pos"] + step, lo, hi), tid=tr["tid"], parent=k, is_div=False))
        tracks = nxt
    np.savez(path, volumes=vols, nodes=np.array(nodes, np.float32), edges=np.array(edges, np.int32),
             divisions=np.array(sorted(set(divs)), np.int32),
             voxel_um_pooled=np.array([1.625, 1.625, 1.625], np.float32))
    return {"n_nodes": len(nodes), "n_edges": len(edges), "n_div": len(set(divs))}


def self_test(repo: Path, tmp_root: Path | None) -> int:
    t0 = time.time()
    if tmp_root is None:
        short = Path(r"C:\biohub_data\work")
        tmp_root = short if short.is_dir() else None   # short path outside OneDrive when available
    tmp = Path(tempfile.mkdtemp(prefix="m184_", dir=str(tmp_root) if tmp_root else None))
    try:
        src, dst = tmp / "sequences", tmp / "out"
        src.mkdir()
        fab = fabricate_npz(src / "seq_0000.npz")
        print(f"fabricated seq_0000.npz: {fab}", flush=True)
        summary = convert_many(src, dst, limit=1, seed=0, max_lineages=None, with_track_id=False,
                               n_test=1, overwrite=True)
        stem = dst / "syn_0000"
        seq = load_sequence(src / "seq_0000.npz")

        sys.path.insert(0, str(repo / "src")); sys.path.insert(0, str(repo / "scripts"))
        from biohub_tracking.io import open_dataset

        # (1) exactly how the trainer opens a movie
        ds = open_dataset(stem, normalize=False, require_tracks=True, load_image=False, downsample=(1, 4, 4))
        assert ds.image_shape == (6, 64, 64, 64), ds.image_shape
        assert tuple(ds.scale) == NATIVE_SCALE, ds.scale
        assert "0.001" in ds.quantiles and "0.999" in ds.quantiles, ds.quantiles
        na = ds.tracks.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        ea = ds.tracks.edge_attrs(attr_keys=["source_id", "target_id"])
        assert len(na) == fab["n_nodes"] and len(ea) == fab["n_edges"], (len(na), len(ea))
        assert all(str(na.schema[c]) == "Int64" for c in ("t", "z", "y", "x")), na.schema
        print(f"open_dataset(load_image=False): shape={ds.image_shape} scale={ds.scale} "
              f"nodes={len(na)} edges={len(ea)} q0.001={ds.quantiles['0.001']:.1f} q0.999={ds.quantiles['0.999']:.1f}")

        # (2) full image load; the trainer's strided read must return the synthetic voxels bit-exactly
        full = open_dataset(stem, normalize=False, require_tracks=True, load_image=True)
        img = np.asarray(full.image)
        assert img.shape == (6, 64, 256, 256) and img.dtype == np.uint16, (img.shape, img.dtype)
        assert np.array_equal(img[:, ::1, ::4, ::4], seq["volumes"]), "strided read != source volumes"
        print("open_dataset(load_image=True): (6, 64, 256, 256) uint16; image[:, ::1, ::4, ::4] == source volumes EXACTLY")

        # (3) graph content round-trip: coordinates, divisions
        src_zyx = np.rint(seq["nodes"][:, 1:4]).astype(np.int64)
        got = na.sort("node_id").select(["z", "y", "x"]).to_numpy()
        assert np.array_equal(got, src_zyx), "node coordinates changed"
        n_div = int((ea.group_by("source_id").len()["len"] == 2).sum())
        assert n_div == fab["n_div"], (n_div, fab["n_div"])
        gm = json.loads((dst / "syn_0000.geff" / "zarr.json").read_text())["attributes"]["geff"]
        assert [a["name"] for a in gm["axes"]] == ["t", "z", "y", "x"]
        assert [a["scale"] for a in gm["axes"]] == [1.0, *NATIVE_SCALE]
        assert set(gm["node_props_metadata"]) == {"t", "z", "y", "x"}
        assert gm["extra"]["estimated_number_of_nodes"] == fab["n_nodes"]
        for c in ("t", "z", "y", "x"):
            aj = json.loads((dst / "syn_0000.geff" / "nodes" / "props" / c / "values" / "zarr.json").read_text())
            assert aj["data_type"] == "int64", (c, aj["data_type"])
        print(f"geff: axes/scale/node props match the real files; divisions={n_div}; "
              f"estimated_number_of_nodes={gm['extra']['estimated_number_of_nodes']}")

        # (4) the trainer's own window loader + Dataset.__getitem__ (needs torch; optional)
        try:
            import train_unet_transformer as trainer
            vm, windows = trainer.load_dataset_windows(stem, window_size=2, downsample=(1, 4, 4))
            fwd = trainer.FrameWindowDataset([(vm, windows)])
            item = fwd[0]
            want = (seq["volumes"][0:2].astype(np.float32) - vm.q_low) / (vm.q_high - vm.q_low + 1e-6)
            assert np.allclose(item["imgs"].float().numpy(), np.clip(want, 0, None).astype(np.float16), atol=1e-3)
            assert len(windows) == 5 and tuple(item["imgs"].shape) == (2, 64, 64, 64)
            print(f"trainer.load_dataset_windows: {len(windows)} windows, max_nodes={fwd.max_nodes}, "
                  f"voxel_size={vm.voxel_size}, imgs={tuple(item['imgs'].shape)}, "
                  f"targets={tuple(item['targets'].shape)}, target edges in window0={int(item['targets'].sum())}")
        except ImportError as exc:                     # torch/models not importable -> loader proof (1-3) still stands
            print(f"trainer import skipped: {exc}")

        zb, gb = dir_bytes(dst / "syn_0000.zarr"), dir_bytes(dst / "syn_0000.geff")
        print(f"disk: zarr={zb / 2**20:.2f} MiB geff={gb / 2**20:.3f} MiB (noise-dominated fake volume, 6 frames)")
        print(f"SELF-TEST PASSED in {time.time() - t0:.1f}s  ({summary['movies_in_dst']} movie)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, help="dir containing seq_*.npz (or its parent biohub_synthetic/)")
    ap.add_argument("--dst", type=Path, help="output dir (short path, outside OneDrive)")
    ap.add_argument("--limit", type=int, default=None, help="convert a seeded random subset of N sequences")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-lineages", type=int, default=None,
                    help="keep only N random lineages (root cells) per sequence; bounds the trainer's "
                         "max_nodes x max_nodes target padding (RAM). Default: keep every node.")
    ap.add_argument("--with-track-id", action="store_true", help="also store track_id as an extra node prop")
    ap.add_argument("--n-test", type=int, default=1, help="movies listed in the dummy `test` split")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="support-pack repo (self-test only)")
    ap.add_argument("--tmp-root", type=Path, default=None, help="self-test temp root (short path)")
    args = ap.parse_args()
    if args.self_test:
        return self_test(args.repo, args.tmp_root)
    if args.src is None or args.dst is None:
        ap.error("--src and --dst are required unless --self-test")
    s = convert_many(args.src, args.dst, args.limit, args.seed, args.max_lineages, args.with_track_id,
                     args.n_test, args.overwrite)
    print(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
