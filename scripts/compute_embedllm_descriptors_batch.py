"""Batch-compute logit + perplexity descriptors for a pool of EmbedLLM models.

Unlike MixInstruct/RouterBench, the EmbedLLM dataset (RZ412/EmbedLLM) only
stores a binary correctness label per (prompt, model) pair -- no generated
response text -- so perplexity descriptors can't be computed "for free" from
already-stored text the way they were for MixInstruct (see PROGRESS.md
section 14). Instead, this script captures the model's own greedy
continuation during the same generate() call used for the logit descriptor,
and immediately scores that continuation with the shared GPT2 judge. This
avoids downloading/running each model twice (once for logit, once for
perplexity) -- one model load produces both descriptors.

For each model in --pool: resolve its hf_id from the registry, optionally
pre-flight-check that there's enough free disk for its weights, download +
load it (4-bit by default), compute both descriptors, save them, then delete
its HF cache before moving to the next model so peak disk usage stays
bounded to roughly one model at a time. Failures (gated repo, OOM, missing
tokenizer, etc.) are logged and skipped rather than aborting the whole batch,
and completed models are skipped on re-run so the script is safe to re-launch
after a crash or timeout.
"""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from router.descriptors import _get_shared_vocab_topk
from router.registry import REGISTRY
from router.utils import load_model_and_tokenizer
from scripts.compute_descriptors_perplexity import perplexity_fingerprint


def log_line(log_path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    if log_path:
        with open(log_path, "a") as f:
            f.write(line + "\n")


def free_model_cache(hf_id: str) -> None:
    """Delete this model's local HF cache snapshot (same approach used in
    colab/repro_mixinstruct.py) so disk usage doesn't grow unbounded across
    the whole pool."""
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    hashes = [
        rev.commit_hash
        for repo in cache.repos
        if repo.repo_id == hf_id
        for rev in repo.revisions
    ]
    if hashes:
        cache.delete_revisions(*hashes).execute()


def estimate_repo_download_gb(hf_id: str) -> float | None:
    """Best-effort size lookup via the HF Hub API. Returns None (rather than
    raising) on any failure so a flaky API call degrades to "skip the
    pre-flight check" instead of aborting the batch."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(hf_id, files_metadata=True)
        weight_exts = (".bin", ".safetensors", ".pt", ".gguf", ".h5")
        total = sum(
            (f.size or 0) for f in info.siblings if f.rfilename.endswith(weight_exts)
        )
        return total / 1e9 if total else None
    except Exception:
        return None


def _existing_ancestor(path: str) -> Path:
    p = Path(path).expanduser()
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def check_disk_headroom(hf_id: str, headroom_gb: float) -> tuple[bool, str]:
    import os

    cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    free_gb = shutil.disk_usage(_existing_ancestor(cache_dir)).free / 1e9
    est_gb = estimate_repo_download_gb(hf_id)
    if est_gb is None:
        return True, f"size lookup failed, skipping check (free={free_gb:.1f}GB)"
    needed = est_gb + headroom_gb
    ok = free_gb >= needed
    return ok, f"est_download={est_gb:.1f}GB free={free_gb:.1f}GB need>={needed:.1f}GB"


def compute_logit_and_perplexity(
    model,
    tokenizer,
    probes: list[str],
    top_k: int,
    n_tokens: int,
    batch_size: int,
) -> tuple[np.ndarray, list[int], np.ndarray, int]:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generate() with causal LMs
    top_token_ids = _get_shared_vocab_topk(tokenizer, probes, k=top_k)

    token_to_idx = {tid: i for i, tid in enumerate(top_token_ids)}
    probs_accum = np.zeros(len(top_token_ids), dtype=np.float32)
    ppl_accum: list[float] = []
    n_total = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for start in tqdm(range(0, len(probes), batch_size), desc="logit+ppl"):
            batch_prompts = probes[start : start + batch_size]
            enc = tokenizer(
                batch_prompts, return_tensors="pt", padding=len(batch_prompts) > 1
            ).to(device)
            prompt_len = enc["input_ids"].shape[1]
            gen = model.generate(
                **enc,
                max_new_tokens=n_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            logits = torch.stack(gen.scores, dim=1)  # (B, T, vocab)
            probs = torch.softmax(logits, dim=-1)

            for b in range(probs.size(0)):
                for tid, idx in token_to_idx.items():
                    probs_accum[idx] += probs[b, :, tid].mean().item()
                n_total += 1

            # Left-padding puts every prompt flush against `prompt_len`, so
            # the freshly generated continuation for row b is always
            # sequences[b, prompt_len:] regardless of that row's own
            # (unpadded) prompt length.
            continuations = gen.sequences[:, prompt_len:].cpu()
            texts = tokenizer.batch_decode(continuations, skip_special_tokens=True)
            for text in texts:
                ppl_accum.append(perplexity_fingerprint(text))

            # Same fragmentation issue as compute_logit_descriptor() at
            # larger N (see descriptors.py comment) -- free eagerly each
            # iteration rather than waiting for the loop to end.
            del enc, gen, logits, probs
            torch.cuda.empty_cache()

    logit_descriptor = probs_accum / max(n_total, 1)
    logit_descriptor = logit_descriptor / (np.linalg.norm(logit_descriptor) + 1e-12)

    ppl_arr = np.asarray(ppl_accum, dtype=np.float32)
    # Same empty/too-short-response fix as compute_descriptors_perplexity.py
    # (commit dbdd200): zero-fill unscoreable entries instead of letting one
    # inf poison the whole vector's L2 norm.
    n_bad = int(np.isinf(ppl_arr).sum())
    if n_bad:
        ppl_arr[np.isinf(ppl_arr)] = 0.0
    ppl_descriptor = ppl_arr / (np.linalg.norm(ppl_arr) + 1e-12)

    return (
        logit_descriptor.astype(np.float32),
        top_token_ids,
        ppl_descriptor.astype(np.float32),
        n_bad,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="JSON list of registry labels (org__repo)")
    ap.add_argument("--probes", required=True, help="Probes JSON: list of {prompt, prompt_id}")
    ap.add_argument("--out_logit_dir", required=True)
    ap.add_argument("--out_perplexity_dir", required=True)
    ap.add_argument("--topk", type=int, default=192)
    ap.add_argument("--n_tokens", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--four_bit", dest="four_bit", action="store_true", default=True)
    ap.add_argument("--no_four_bit", dest="four_bit", action="store_false")
    ap.add_argument("--progress_log", default=None)
    ap.add_argument("--min_disk_headroom_gb", type=float, default=15.0)
    ap.add_argument("--skip_disk_check", action="store_true")
    args = ap.parse_args()

    pool = json.load(open(args.pool))
    probes_raw = json.load(open(args.probes))
    probes = [p["prompt"] for p in probes_raw]

    out_logit_dir = Path(args.out_logit_dir)
    out_logit_dir.mkdir(parents=True, exist_ok=True)
    out_ppl_dir = Path(args.out_perplexity_dir)
    out_ppl_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.progress_log
    succeeded, failed, skipped = [], [], []

    log_line(log_path, f"Starting batch over {len(pool)} models "
                        f"(topk={args.topk}, n_tokens={args.n_tokens}, "
                        f"batch_size={args.batch_size}, four_bit={args.four_bit})")

    for label in pool:
        if label not in REGISTRY:
            log_line(log_path, f"[skip] {label}: not found in experts/registry.json")
            skipped.append(label)
            continue
        hf_id = REGISTRY[label]["hf_id"]

        logit_path = out_logit_dir / f"{label}.npy"
        ppl_path = out_ppl_dir / f"{label}.npy"
        if logit_path.exists() and ppl_path.exists():
            log_line(log_path, f"[skip] {label}: both descriptors already exist")
            continue

        if not args.skip_disk_check:
            ok, detail = check_disk_headroom(hf_id, args.min_disk_headroom_gb)
            log_line(log_path, f"[preflight] {label}: {detail}")
            if not ok:
                log_line(log_path, f"[FAILED] {label} ({hf_id}): insufficient disk headroom")
                failed.append((label, "insufficient_disk"))
                continue

        log_line(log_path, f"[start] {label} ({hf_id}) ...")
        model = tok = None
        success = False
        try:
            model, tok = load_model_and_tokenizer(hf_id, four_bit=args.four_bit)
            logit_desc, token_ids, ppl_desc, n_bad_ppl = compute_logit_and_perplexity(
                model, tok, probes,
                top_k=args.topk, n_tokens=args.n_tokens, batch_size=args.batch_size,
            )
            np.save(logit_path, logit_desc)
            with open(logit_path.with_suffix(".tokens.json"), "w") as f:
                json.dump(token_ids, f)
            np.save(ppl_path, ppl_desc)
            log_line(
                log_path,
                f"[done] {label}: logit dim={logit_desc.size}, "
                f"ppl dim={ppl_desc.size} ({n_bad_ppl} zero-filled)",
            )
            success = True
        except Exception as e:  # noqa: BLE001 - intentionally broad: one bad
            # model must not kill the rest of a long unattended batch.
            msg = str(e)
            hint = ""
            lowered = msg.lower()
            if "gated" in lowered or "403" in msg or "401" in msg:
                hint = " [hint: gated repo -- has this HF account accepted the license, and is HF_TOKEN set?]"
            elif "out of memory" in lowered or "cuda oom" in lowered:
                hint = " [hint: OOM -- this model may need more cards than currently available]"
            log_line(log_path, f"[FAILED] {label} ({hf_id}): {msg}{hint}")
            failed.append((label, msg))
        finally:
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
            if success:
                free_model_cache(hf_id)
                succeeded.append(label)
            else:
                log_line(log_path, f"  (leaving {hf_id} HF cache on disk in case of retry)")

    log_line(
        log_path,
        f"Batch finished. succeeded={len(succeeded)} failed={len(failed)} skipped={len(skipped)}",
    )
    if failed:
        log_line(log_path, f"Failed models: {[name for name, _ in failed]}")


if __name__ == "__main__":
    main()
