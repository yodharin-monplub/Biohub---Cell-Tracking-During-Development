#!/usr/bin/env python3
"""Add an optional THIRD seed to the patched 0.947 predictor (idempotent text patch).

When BIOHUB_TERTIARY_WEIGHTS is set, the pipeline's "secondary model" becomes an ensemble of the
public secondary seed and the tertiary checkpoint:
  - encode(): UNet feature maps are concatenated on the channel axis; detection logits are averaged
    after per-model standardisation (zero mean / unit std per frame) so neither seed dominates;
  - _index_features(): unchanged generic gather on the concatenated map;
  - predict_edges(): features are split back per model, each model scores edges with its own
    transformer, logits are averaged.
Everything downstream (mean/std alignment to the primary detector, retention guard, D4 feature
TTA, low_margin_consensus fusion, ILP, post-processing) is untouched. With the env var unset the
patched script behaves exactly as before.

Usage: python model187/patch_tertiary.py <repo_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = "        secondary_model, secondary_window_size, secondary_downsample = load_model(Path(secondary_weights_text), device,)\n"
MARK = "# --- model187 tertiary seed ---"
INSERT = ANCHOR + f'''        {MARK}
        _tertiary_text = os.environ.get("BIOHUB_TERTIARY_WEIGHTS", "").strip()
        if _tertiary_text:
            _tertiary_model, _tw, _td = load_model(Path(_tertiary_text), device,)
            if _tw != secondary_window_size or _td != secondary_downsample:
                raise ValueError("Tertiary model has an incompatible inference grid")

            class _SeedEnsemble(torch.nn.Module):
                def __init__(self, members):
                    super().__init__()
                    self.members = torch.nn.ModuleList(members)

                def encode(self, imgs):
                    outs = [m.encode(imgs) for m in self.members]
                    unet_out = torch.cat([o[0] for o in outs], dim = 2)
                    n_frames = len(outs[0][1])
                    det = []
                    for f in range(n_frames):
                        parts = []
                        for _u, d in outs:
                            x = d[f].float()
                            parts.append((x - x.mean()) / x.std(unbiased = False).clamp_min(1e-4))
                        det.append(torch.stack(parts, dim = 0).mean(dim = 0).to(outs[0][1][f].dtype))
                    return unet_out, det

                def _index_features(self, feat_maps, coords, mask):
                    return self.members[0]._index_features(feat_maps, coords, mask)

                def predict_edges(self, feat_src, feat_tgt, coords_src, coords_tgt, pos_src, pos_tgt, mask_src, mask_tgt):
                    n = len(self.members)
                    c = feat_src.shape[-1] // n
                    logits = [m.predict_edges(feat_src[..., i * c:(i + 1) * c], feat_tgt[..., i * c:(i + 1) * c],
                                              coords_src, coords_tgt, pos_src, pos_tgt, mask_src, mask_tgt)
                              for i, m in enumerate(self.members)]
                    return torch.stack(logits, dim = 0).mean(dim = 0)

            secondary_model = _SeedEnsemble([secondary_model, _tertiary_model]).to(device).eval()
            print(f"TERTIARY_SEED_ACTIVE: {{_tertiary_text}}", flush = True)
'''


def main() -> None:
    repo = Path(sys.argv[1])
    path = repo / "scripts" / "predict_unet_transformer.py"
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        print("already patched")
        return
    if text.count(ANCHOR) != 1:
        raise SystemExit(f"anchor found {text.count(ANCHOR)} times; refusing to patch")
    path.write_text(text.replace(ANCHOR, INSERT), encoding="utf-8")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print("patched", path)


if __name__ == "__main__":
    main()
