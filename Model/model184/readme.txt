MODEL184 - SYNTHETIC PRETRAINING FOR THE PUBLIC TemporalUNet3D + NODE TRANSFORMER

RESULT (2026-09-20): REJECTED on the held-out embryo.
  Recipe: pretrain 40 epochs on 400 locally generated synthetic sequences (max 150 lineages each),
  then the model182 fold0 schedule (80 x 125, batch 2) on 128 x 6bba movies, strict full-model warm
  start (136 tensors); exact 0.947 predictor + post-processing on the same 24 held-out 44b6 movies.
    from scratch (model182)          proxy 0.79657   edge TP/FP/FN 10925/1259/1604  missed GT nodes  799
    synthetic pretrain + fine-tune   proxy 0.75994   edge TP/FP/FN 10240/1174/2289  missed GT nodes 1503
  The pretrained model loses on 20 of 24 movies; the loss is detection recall (almost twice as many
  annotated nodes undetected), not linking. Likely causes: the synthetic volumes saturate the 16-bit
  range (normalised intensities cap near 1.0 vs ~2.7 in real movies) and every cell is labelled, so
  the detector learns a brightness/contrast prior that fine-tuning on sparse labels does not undo.
  Single seed per arm, so the exact size is uncertain, but the direction is consistent across movies.
  The forum claim (+0.012 for a different, weaker pipeline) does not transfer to this setup.

Status 2026-09-19: converter written and self-tested. NOTHING was downloaded, NO training was run,
no score exists. Everything below about the real synthetic files is read from the generator's source
code, not from the files themselves (see RISKS).

Files
  model184\convert_synthetic.py   seq_*.npz -> <stem>.zarr + <stem>.geff (+ synthetic_splits.json)
  model184\readme.txt             this file


1. THE SYNTHETIC FORMAT (evidence = notebook cells)

Generator : C:\biohub_data\public\josefreitasalvesneto__biohub-synthetic-dataset\biohub-synthetic-dataset.ipynb
            code cell 15, function run_dataset_build()  (markdown cell 14 is its docstring)
Explorer  : C:\biohub_data\public\josefreitasalvesneto__synthetic-3d-microscopy-data-for-cell-tracking\
            synthetic-3d-microscopy-data-for-cell-tracking.ipynb, markdown cell 4 (format table),
            code cells 3, 5, 15, 16 (how it is read back)
Licence   : CC0 (explorer cell 17).

biohub_synthetic/sequences/seq_XXXX.npz   (np.savez, NOT compressed; XXXX = 4-digit index j)
  volumes          (T, 64, 64, 64) uint16
                   gen: `vols = np.zeros((T_SEQ, 64, 64, 64), np.uint16)  # pooled, to fit the budget`
                        `vols[t] = _to_u16(pool_xy(nat))`
                   pool_xy(f) = f[::1, ::4, ::4] (stride, NOT block mean) of a native (64,256,256)
                   float volume in [0,1]; _to_u16 = clip(v*65535, 0, 65535).  So volumes ARE ALREADY
                   XY-STRIDED BY 4 and are exactly what the trainer's image[:, ::1, ::4, ::4] would
                   return from the native movie. Intensity uses the full 0..65535 range (real movies
                   top out around 4000).
  T                T_SEQ = int(os.environ.get("DS_SEQ_LEN", "6")) -> 6 frames per sequence by default.
                   Consistent with the published totals (explorer cell 0): 4,056,226 nodes /
                   2,174 sequences = 1,866 nodes per sequence ~ 6 frames x ~310 cells, and
                   165,267 / 4,056,226 = 4.07% division nodes ~ DS_DIV_RATE 0.05 x 5/6.
                   manifest.json records T per sequence; metadata.json records seq_len. CHECK THEM.
  nodes            (M, 5) float32, columns [t, z, y, x, track_id]
                   gen: `nodes.append([t, tr["pos"][0], tr["pos"][1], tr["pos"][2], tr["tid"]])`
                   ORDER is z, y, x. UNITS are NATIVE VOXELS of the 64 x 256 x 256 grid (positions
                   come from sample_shell_positions(n0, (64, 256, 256), ...) and are clipped to
                   [4,12,12]..[59,243,243]); they are sub-voxel floats, NOT pooled coordinates and NOT
                   micrometres. (The explorer divides static centroids by [1,4,4] to overlay them on
                   pooled images; the sequence plot in cell 15 just plots them raw.)
                   Rows are ordered frame by frame; row index == node index used by edges/divisions.
                   track_id = tr["tid"] is the ROOT-CELL (lineage) id: both daughters inherit the
                   mother's tid, so it is a lineage id, not a tracklet id. float32 holds it exactly.
  edges            (E, 2) int32, [parent_row, child_row] into `nodes`, always t -> t+1, every cell of
                   frame t+1 has exactly one parent (no appearances, no disappearances, no merges).
                   gen: `edges.append([prev_ids[tr["parent_slot"]], cur_ids[k]])`
  divisions        (D,) int32, sorted unique row indices of mother nodes (out-degree 2).
                   gen: `divisions=np.array(sorted(set(divs)), np.int32)`
  voxel_um_pooled  (3,) float32 = [1.625, 1.625, 1.625]
  Cells per sequence: n0 = max(20, choice(real DoG counts) * 1.25), i.e. roughly 30..520 at t=0,
  growing ~5% per frame from divisions. Per-frame division probability 0.05 (real GT ~0.26% of nodes).
  Motion: OU step, median 1.86 um/frame, persistence 0.30, sister separation ~7.24 um, plus a drift.

biohub_synthetic/static/vol_XXXXX.npz     volume (64,256,256) uint16 NATIVE, centroids (N,3) float32
                   native voxels (z,y,x), voxel_um [1.625,0.40625,0.40625]. No time axis, no edges ->
                   useless for this trainer (it needs frame pairs). Not converted.
manifest.json      {"static": [{file, n_cells, shape, bytes}...],
                    "sequences": [{file:"sequences/seq_0000.npz", T, n_nodes, n_edges, n_divisions, bytes}...]}
metadata.json      n_static, n_sequences, total_gb, seq_len, total_divisions, total_nodes, division_rate,
                   voxel_native_um, pooling, real_count_range, shell_kind, motion_calibration{...},
                   note_division_rate, config{SYNTH_* env}
Size: one seq npz = 6*64^3*2 B = 3.0 MiB + tables; 2,174 sequences ~ 6.5 GB of the 18.5 GB dataset.
      Download only sequences/ (kaggle datasets download -f per file, or unzip selectively).


2. WHAT THE UNCHANGED LOADER / TRAINER REQUIRE (read from the code)

biohub_tracking.io.open_dataset(<dir>/<stem>):
  - <stem>.zarr : zarr group with array "0" shaped (T,Z,Y,X); attrs["multiscales"][0]["datasets"][0]
    ["coordinateTransformations"][0] = {"type":"scale","scale":[1,1.625,0.40625,0.40625]} (last 3 used);
    attrs["image_statistics"]["quantiles"] dict with string keys.
  - <stem>.geff : read with td.graph.IndexedRXGraph.from_geff (geff structure validation is ON, so the
    stated and actual dtypes must agree).
train_unet_transformer.load_dataset_windows():
  - raises unless quantiles has "0.001" and "0.999"; normalises (raw - q0.001)/(q0.999 - q0.001), clamp(min 0)
  - needs node attrs node_id,t,z,y,x and edge attrs source_id,target_id; coords = [z,y,x]/downsample,
    then .long() (floor) for feature indexing and the detection target
  - image read: zarr["0"][t:t+W, ::1, ::4, ::4]; voxel_size = scale*downsample = 1.625 isotropic;
    transformer coords are multiplied back by downsample (native-voxel units)
  - a window is dropped if any of its frames has zero nodes
  - extra.estimated_number_of_nodes is NOT read by the loader/trainer (only scripts/evaluate.py reads it)
Real reference (data\raw\train\6bba_05b6850b): array (100,64,256,256) uint16, chunks (1,64,256,256),
  bytes + blosc(zstd, clevel 1, bitshuffle, typesize 2); quantile keys 0.0,0.001,0.01,0.1,0.9,0.99,0.999,1.0;
  geff 1.1, directed, axes t/z/y/x with scale 1/1.625/0.40625/0.40625 and min/max, node props t,z,y,x all
  int64 (integer native voxels), no edge props, extra = {"estimated_number_of_nodes": 6362},
  node ids like (t+1)*1_000_000 + k.


3. CONVERSION DECISIONS (convert_synthetic.py)

a) Resolution: volumes are already the strided read, so each voxel is repeated 4x4 in (Y,X):
   up = repeat(repeat(v, 4, axis=2), 4, axis=3) -> (T,64,256,256). up[:, :, ::4, ::4] == v bit-exactly
   (asserted for every sequence at conversion time and again in the self-test through open_dataset and
   through the trainer's own FrameWindowDataset.__getitem__). Trainer args stay downsample=(1,4,4).
   REJECTED alternative: store the 64^3 volumes as-is with zarr scale 1.625 iso and train stage 1 with
   downsample=(1,1,1). Images and voxel_size would be identical, but the trainer feeds the transformer
   `coords * downsample`, so Y/X distances would be 4x smaller than in stage 2 / inference and
   config.json would record the wrong downsample. Not equivalent -> not used.
b) Coordinates: kept in NATIVE voxels (as generated), rounded to the nearest integer and stored int64,
   exactly like the organisers' GEFFs. The trainer then floors y/4, x/4 - the same convention (and
   the same sub-voxel bias) it applies to the real annotations. Sub-voxel precision is discarded
   because the loader only ever uses integer voxels.
c) GEFF written via tracksdata graph.to_geff(path, geff_metadata=...) (same call as io.save_graph) with
   explicit axes (scale, min, max), node props t,z,y,x int64, no edge props,
   extra.estimated_number_of_nodes = node count (labels are complete). tracksdata hard-codes t as Int32,
   so the converter rewrites nodes/props/t/values as int64 and fixes the stated dtype, making the file
   dtype-identical to the real ones. Differences that remain and are harmless: geff_version 1.3 instead
   of 1.1, extra also has "tracksdata": {} and "synthetic_frames". track_id is dropped by default
   (--with-track-id keeps it as an extra int64 node prop; the trainer ignores it).
d) Node ids follow the real scheme (t+1)*1e6 + rank. Edges are validated: t -> t+1 only, no merges,
   out-degree <= 2, `divisions` must equal the out-degree-2 nodes.
e) Zarr: v3, array "0", chunks (1,64,256,256), bytes + blosc zstd clevel 1 bitshuffle - same codecs as
   the real movies. Quantiles are computed exactly (np.quantile over all T*64^3 source voxels, which has
   the same distribution as the repeated array) and written under the same 8 keys.
f) Stems are syn_XXXX ("syn" sits where the embryo id sits in real names). The converter also writes
   <dst>\synthetic_splits.json = [{"train": all stems, "test": [first stem]}] and conversion_manifest.json.
g) --max-lineages N keeps N random root lineages per sequence (whole lineages, so divisions stay intact).
   Reason: FrameWindowDataset pre-pads a (W-1, M, M) float32 target for EVERY window, with M = the largest
   per-frame node count in the whole split. Dense synthetic frames reach ~650 nodes -> 1.7 MB per window;
   500 sequences x 5 windows ~ 4.4 GB padded + ~1.5 GB unpadded copies held in RAM, and
   detect_and_match() runs a Python/GPU-sync loop per matched detection. With --max-lineages 150,
   M ~ 200 and the target tables shrink to ~0.5 GB. Cost: labels become sparse again (like the real
   data, where det_neg_weight=0.01 already tolerates unlabelled nuclei). Default keeps everything.
h) --limit N picks a seeded random subset of the available seq files (sorted, rng.choice, --seed).

Disk estimate, 500 sequences after conversion
  Raw (T,64,256,256) uint16 is 50 MB per sequence, but the 4x4 repetition compresses away: the self-test
  volume (same noise model as the generator: Poisson 300 pe + read noise 0.006, so close to worst case)
  is 4.54 MiB as zarr vs 3.0 MiB for the raw pooled voxels, geff < 0.1 MiB.
  => ~4.5-6 MiB per sequence => ~2.3-3 GB for 500 sequences (budget 4 GB). Source npz for those 500:
  1.5 GB (can be deleted after conversion). All 2,174 sequences: ~10-13 GB. Conversion ~4-5 s each
  (~40 min for 500). Put --dst on C:\biohub_data\work\model184\synth (short path, outside OneDrive).


4. SELF-TEST (no downloads, ~13 s)
  .venv-gpu\Scripts\python.exe model184\convert_synthetic.py --self-test
  Fabricates one seq_0000.npz in the exact generator format (6x64x64x64 uint16, float native coords,
  lineage track ids, int32 edges/divisions), converts it, then:
  open_dataset(load_image=False, require_tracks=True, downsample=(1,4,4)) as the trainer does;
  open_dataset(load_image=True) and checks image[:, ::1, ::4, ::4] == source volumes; checks coordinates,
  division count, geff metadata and on-disk dtypes; finally runs the trainer's load_dataset_windows and
  FrameWindowDataset[0]. Last run:
    fabricated seq_0000.npz: {'n_nodes': 197, 'n_edges': 173, 'n_div': 22}
    open_dataset(load_image=False): shape=(6, 64, 64, 64) scale=(1.625, 0.40625, 0.40625) nodes=197 edges=173 q0.001=0.0 q0.999=50457.0
    open_dataset(load_image=True): (6, 64, 256, 256) uint16; image[:, ::1, ::4, ::4] == source volumes EXACTLY
    geff: axes/scale/node props match the real files; divisions=22; estimated_number_of_nodes=197
    trainer.load_dataset_windows: 5 windows, max_nodes=46, voxel_size=(1.625, 1.625, 1.625), imgs=(2, 64, 64, 64), targets=(1, 46, 46), target edges in window0=25
    disk: zarr=4.54 MiB geff=0.008 MiB (noise-dominated fake volume, 6 frames)
    SELF-TEST PASSED in 12.7s  (1 movie)


5. TWO-STAGE TRAINING RECIPE (not run)

Both stages go through a wrapper shaped like model182\train_fold_windows.py (plan-only default,
--execute + BIOHUB_RUNS_RESUMED=1, watchdog thread, sdpa_kernel(SDPBackend.MATH) around trainer.train -
the fused SDPA kernel crashes on the RTX 4050, see model156\readme.txt). The vendored trainer saves
edge_predictor_best.pth whenever test acc*recall >= best, i.e. it SELECTS THE CHECKPOINT ON ITS `test`
SPLIT. Neutralise it exactly as model156/model182 do:
    trainer.evaluate = lambda *a, **k: (0.0, 0.0, 0.0)   # score 0 >= best 0 every epoch -> final epoch kept
    trainer.WEIGHTS_PATH = <output root>                 # checkpoint -> <root>\<method>\split_<fold>\
and give `test` one movie that is already in `train` (it only feeds the mandatory shape/max_nodes loader).
Windows: num_workers=0 (spawned DataLoader workers die with EOFError in this trainer).

Stage 0 - convert
  .venv-gpu\Scripts\python.exe model184\convert_synthetic.py --src <...>\biohub_synthetic\sequences
      --dst C:\biohub_data\work\model184\synth --limit 500 --seed 0 [--max-lineages 150]

Stage 1 - pretrain on ~500 synthetic sequences (2,500 two-frame windows), random init
  trainer.train(
      data_dir=Path(r"C:\biohub_data\work\model184\synth"), fold=0,
      splits_file=Path(r"C:\biohub_data\work\model184\synth\synthetic_splits.json"),
      method="model184_pretrain_syn500_seed20260919",
      n_epochs=40, max_iters=125,            # 5,000 iters x batch 2 = 10,000 windows = 4 passes
      lr=1e-4, batch_size=2, num_workers=0,
      unet_out_channels=32, unet_layers=[32, 64, 128], unet_weights=None,
      downsample=(1, 4, 4), det_loss_weight=1.0, det_neg_weight=0.01,
      seed=20260919, window_size=2, pool_kernel_um=5.0, data_parallel=False)
  Same architecture/loss settings as model182 so stage 2 and the existing predictor load it unchanged.
  det_neg_weight: keep 0.01 if --max-lineages was used; with complete labels 0.05-0.1 is defensible
  (every unlabelled voxel really is background) but changes the detection-logit calibration that the
  0.965 deployment threshold was tuned on, so 0.01 is the safe default.
  Expected wall time: model182 measured ~0.9 s/iter on real data; dense synthetic frames (hundreds of
  matched detections per frame) will be slower - pilot 1 epoch x 8 iters first, then set the watchdog.

Stage 2 - fine-tune on real movies
  IMPORTANT: train(unet_weights=...) does `unet.load_state_dict(state, strict=False)` on the BARE UNet.
  Passing the stage-1 edge_predictor_best.pth there loads NOTHING (all keys are prefixed "unet." /
  "transformer." / "detect_head.") and only prints "N missing, M unexpected". Keep unet_weights=None and
  warm-start the whole model from the wrapper, without touching the trainer file:
      _Base = trainer.UNetNodeTransformer
      class _WarmStart(_Base):
          def __init__(self, *a, **k):
              super().__init__(*a, **k)
              state = torch.load(PRETRAINED, map_location="cpu", weights_only=True)
              self.load_state_dict(state, strict=True)      # strict: fail loudly on any mismatch
      trainer.UNetNodeTransformer = _WarmStart
  (UNet-only warm start alternative: save {k[5:]: v for k, v in state.items() if k.startswith("unet.")}
  to a file and pass that as unet_weights; verify the printed "0 missing, 0 unexpected".)

  2a. Validation first (train-proxy bias lesson: never judge this on train movies). Reuse the frozen
      embryo folds model156\outer_splits.json and the model182 schedule so the ONLY change vs the
      from-scratch baseline (fold0 official 0.7502 on 71 held-out 44b6 movies) is the initialisation:
  trainer.train(
      data_dir=DEFAULT_DATA, fold=<0|1>, splits_file=<internal split: train=fold train, test=[train[0]]>,
      method="model184_ft_fold<k>_seed20260914",
      n_epochs=80, max_iters=125, lr=1e-4, batch_size=2, num_workers=0,
      unet_out_channels=32, unet_layers=[32, 64, 128], unet_weights=None,   # warm start via _WarmStart
      downsample=(1, 4, 4), det_loss_weight=1.0, det_neg_weight=0.01,
      seed=20260914, window_size=2, pool_kernel_um=5.0, data_parallel=False)
      Score the held-out embryo with the frozen export + organiser evaluator (model156\evaluate_fold).
      If 80 epochs erase the benefit, try the precommitted shorter/lower variant n_epochs=40, lr=5e-5.
  2b. Only if 2a beats the from-scratch fold: fine-tune on all 199 movies for submission
      (train = 199 stems, test = [train[0]], same arguments, final epoch kept).


6. RISKS / UNKNOWNS
  - The real npz files were never opened (download forbidden). Format facts come from the generator
    source + the author's format table; the published files could have been built with other env
    values (DS_SEQ_LEN, DS_DIV_RATE). First thing after download: read metadata.json (seq_len) and
    np.load one file. The converter validates shapes/dtypes/edges and handles T != 6 and native
    (64,256,256) volumes, and fails loudly otherwise.
  - Intensity statistics differ: synthetic volumes span 0..65535 with heavy saturation (the author
    targets ~26% saturated cells) so after q0.001/q0.999 normalisation they top out near 1.0, whereas
    real movies reach ~2.7 (max 3994 vs q0.999 1478). Dark background also sits at a different
    normalised level. Fine-tuning has to absorb this; it may make pretraining less useful.
  - Only 6 frames and 5% divisions per frame (15x the real rate), no cell appearance/disappearance, no
    out-of-view exits: the transformer can learn an inflated division prior and "every node has a
    successor". Stage 2 must be long enough to undo that; watch division FP on the held-out fold.
  - Generator calibration used the first 10 movies of the first embryo (sorted ids -> 44b6). So fold 0
    (train 6bba, held-out 44b6) has mild indirect exposure of the held-out embryo's density/shell
    statistics through the synthetic data; fold 1 (held-out 6bba) is the cleaner test.
  - RAM/speed with dense labels (section 3g) on this shared 6 GB laptop; use --max-lineages or fewer
    sequences if loading 500 dense sequences does not fit.
  - Forum reports credit synthetic pretraining, but there is no local evidence yet that THIS dataset
    helps THIS model; decide on the embryo-held-out folds, not on train-movie proxies.


7. LOCAL GENERATION INSTEAD OF DOWNLOAD (2026-09-19)

Files
  model184\generator_src.py   the author's generator code, assembled MECHANICALLY from line ranges of the
                              notebook's code cells (cell 2 constants, cell 4 pool_xy + DoG detector, cell 6
                              read_frame/norm, cell 13 native generator incl. fit_shell /
                              sample_shell_positions / gen_volume_native, cell 15 _to_u16/_real_stats).
                              Provenance: Jose Freitas (josefreitasalvesneto), notebook
                              josefreitasalvesneto/biohub-synthetic-dataset, CC0. Rebuild script:
                              C:\biohub_data\work\model184\nbcells\build_src.py <dst>.
  model184\generate_local.py  --n-seq N --start 0 --out <dir> --seed S --workers 4 [--recalibrate]
Real-data inputs the generator needs (reproduced exactly as run_dataset_build/_real_stats do):
  only the IMAGES (no .geff) of the first 10 sorted movies of the first sorted embryo, frames t=0 and t=1:
    44b6_0113de3b 44b6_0b24845f 44b6_0c582fdc 44b6_0db75fae 44b6_12dfb391
    44b6_144b256d 44b6_1574802b 44b6_18ced818 44b6_1d530831 44b6_24264f12
  each frame: norm (p1/p99.5) -> stride pool [::1,::4,::4] -> dog_detect_precise. Gives (a) 20 DoG counts
  (140..417) from which n0 = max(20, choice(counts)*1.25) is drawn, (b) fit_shell on the first 8 frames'
  detections -> kind "plane". Cached in C:\biohub_data\work\model184\synthetic\calibration.json (reused on
  resume so every run sees bit-identical calibration). Motion constants (1.86 um, 0.30, 7.24 um) are
  hard-coded in the notebook; the .geff files are NOT read.
Deviations from the notebook (none touch physics or sampling):
  - find_train_dir -> data\raw\train (or $BIOHUB_TRAIN_DIR).
  - the body of the part-B while loop became generate_sequence(j, ...), so indices run in parallel and
    are resumable. RNG seeds unchanged with --seed 0 (500000+j motion, 900000+j*97+t rendering), so
    seq_j should be the published seq_j IF Kaggle's train set/library versions give the same calibration
    (not verifiable without downloading). --seed S != 0 uses default_rng([S, base]) = new data.
  - no static volumes, no size/time budget loop (fixed N), files written via temp name + rename;
    manifest.json has "static": [] and metadata.json has an extra "local_generation" block.
  - 1 BLAS thread per worker; CPU only.
Smoke test (4 sequences, 4 workers): 8.3-9.5 s per sequence per worker, 2.6 s/seq wall; 3.18-3.23 MB each
  (mean 3,196,195 B); 177..464 cells at t=0, 1211..3187 nodes, 47..147 divisions per sequence.
  convert_synthetic.py -> C:\biohub_data\work\model184\converted_smoke: 4 movies, 19.4 MB (4.9 MB/seq).
  Loader check (C:\biohub_data\work\model184\nbcells\check_loader.py): open_dataset(load_image False/True),
  strided read == source volumes bit-exactly, node/edge counts match, trainer.load_dataset_windows gives
  5 windows each and FrameWindowDataset[0] works. q0.001 ~350-520, q0.999 ~65000-65535.
  NOTE max nodes/frame reached 611 (seq_0001) -> see 3g (--max-lineages) before loading hundreds of these.
Full run launched 2026-09-19 18:30 (400 sequences, indices 0..399, ~17 min, ~1.3 GB):
  Start-Process .venv-gpu\Scripts\python.exe -ArgumentList "model184\generate_local.py --n-seq 400 --out
    C:\biohub_data\work\model184\synthetic\sequences --seed 0 --workers 4" -RedirectStandardOutput
    C:\biohub_data\work\model184\generate.log -RedirectStandardError C:\biohub_data\work\model184\generate.err
Resume / extend: rerun the same command (existing seq_XXXX.npz are skipped, stale *.tmp* removed, manifest
  rebuilt from the files on disk). More data: raise --n-seq (e.g. 1000) or add --start 400.
  Then: convert_synthetic.py --src C:\biohub_data\work\model184\synthetic\sequences --dst C:\biohub_data\work\model184\synth
  RESULT: finished 18:45, 918 s wall (2.3 s/seq), 400 sequences, 1220 MiB, no skipped indices, empty generate.err.
  Totals are in C:\biohub_data\work\model184\synthetic\metadata.json / manifest.json.
