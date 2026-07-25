"""
Load a QueryEncoder from a "slim" checkpoint (proj.pt + config.json only,
no re-uploaded copy of the frozen sentence-transformer base weights).

The base model is identical across every training run in this project since
it is always frozen (--unfreeze_lm was never passed) - only the small
projection MLP (proj.pt, <1MB) differs. Re-downloading the ~90MB base model
from HuggingFace once is much cheaper than storing it redundantly per
checkpoint in git.

Usage:
    from scripts.load_slim_encoder import load_slim_encoder
    encoder = load_slim_encoder("local_checkpoints/slim/query-encoder-logit")
"""
import json
from pathlib import Path

import torch

from router.query_encoder import QueryEncoder

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_slim_encoder(slim_dir: str | Path, device: str | None = None) -> QueryEncoder:
    slim_dir = Path(slim_dir)
    config = json.load(open(slim_dir / "config.json"))
    proj_dim = config.get("proj_dim", 256)

    encoder = QueryEncoder(BASE_MODEL, device=device, proj_dim=proj_dim)
    encoder.proj.load_state_dict(torch.load(slim_dir / "proj.pt", map_location=encoder.device))
    encoder.eval()
    return encoder


if __name__ == "__main__":
    import sys
    enc = load_slim_encoder(sys.argv[1] if len(sys.argv) > 1 else "local_checkpoints/slim/query-encoder-logit")
    vec = enc.encode("sanity check prompt")
    print(f"Loaded OK. Output embedding shape: {vec.shape}, norm: {(vec**2).sum()**0.5:.4f}")
