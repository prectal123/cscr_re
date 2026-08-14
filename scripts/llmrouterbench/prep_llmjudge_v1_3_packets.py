"""Prepare V1.3 judging packets for a HUMAN-IN-THE-LOOP judge (Claude, via the
chat session) instead of the paid Batch API path. For each of the 66 probes,
writes an anonymized packet (query + rubric info + 20 shuffled, unlabeled
responses) that the judge reads and scores, plus a separate answer key
(anon_id -> real model name) that is NOT shown to the judge during grading.

Outputs:
  v1_3_judge_input/{dataset}__{p_rank}.json   (packet -- read this to judge)
  v1_3_judge_keys/{dataset}__{p_rank}.json    (anon_id -> real model name)

Usage: python prep_llmjudge_v1_3_packets.py
"""
import json
import random
from pathlib import Path

import common_lite20 as common

DATA_DIR = Path("local_descriptors/llmrouterbench_lite20")
PACKET_DIR = DATA_DIR / "v1_3_judge_input"
KEY_DIR = DATA_DIR / "v1_3_judge_keys"
RESULTS_DIR = DATA_DIR / "v1_3_judge_results"
OPEN_ENDED_RESPONSE_CAP = 3000  # arenahard family: full response text needed
REASONING_TAIL_CAP = 500  # ground-truth categories: just the concluding chunk of raw_output

OPEN_ENDED = {"arenahard", "arenahard_coding", "arenahard_creative_writing", "arenahard_math"}

CATEGORY_GUIDANCE = {
    "arenahard": (
        "General instruction-following. Weight: correctness of any factual/technical claims, "
        "completeness relative to what was asked, and helpfulness."
    ),
    "arenahard_coding": (
        "Open-ended coding task. Read the code and verify its logic by reasoning through it (you "
        "cannot execute it) -- check correctness, that it addresses the stated requirements, and "
        "whether edge cases are handled."
    ),
    "arenahard_creative_writing": (
        "Creative writing task. Weight: craft quality (imagery, flow, voice), and strict adherence "
        "to any explicit constraints in the prompt (requested language, rhyme, form, topic, length)."
    ),
    "arenahard_math": (
        "Open-ended math problem with no provided reference answer. Independently verify the final "
        "answer and the reasoning yourself before scoring."
    ),
}

RUBRIC_NOTE = {
    "ground_truth": (
        "Each response gives 'prediction' (the model's final extracted answer) and 'reasoning_tail' "
        "(the last ~500 chars of its full reasoning, for partial-credit context only -- not the full "
        "chain of thought). Score 1-10 on whether 'prediction' is equivalent to 'ground_truth' "
        "(allow different valid formats/notation for the same value): 9-10 prediction is correct "
        "(equivalent to ground_truth); 6-8 prediction is correct but reasoning_tail shows a flawed "
        "step, or prediction is a superset/imprecise-but-equivalent form; 3-5 reasoning_tail shows "
        "the right approach but prediction is wrong; 1-2 prediction is wrong and reasoning_tail is "
        "irrelevant, missing, or nonsensical."
    ),
    "open_ended": (
        "Score 1-10 on absolute quality (no reference answer exists): 9-10 fully addresses the "
        "prompt, high quality, meets every explicit constraint; 6-8 minor gaps/errors; 3-5 notable "
        "gaps/errors/ignored constraints; 1-2 off-topic or broken."
    ),
}


def load_probes():
    with open(DATA_DIR / "v1_3_probe_selection.json", encoding="utf-8") as f:
        return json.load(f)["selected"]


def load_setA():
    import pickle

    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        return pickle.load(f)["setA"]


def build_ground_truth_lookup(datasets_needed):
    """dataset -> list[dict(ground_truth=..., record=...)], record kept per-model for prediction."""
    lookup = {}
    for ds in datasets_needed:
        lookup[ds] = common.load_dataset_records(ds)  # model -> list[record]
    return lookup


def main():
    selected = load_probes()
    setA = load_setA()

    gt_datasets = [c["dataset"] for c in selected if c["dataset"] not in OPEN_ENDED]
    gt_lookup = build_ground_truth_lookup(gt_datasets)

    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(0)
    n_packets = 0
    for cat in selected:
        ds = cat["dataset"]
        ds_idx_map = setA[ds]["idx"]
        for p_rank, probe in enumerate(cat["probes"]):
            local_i = probe["local_idx_in_setA"]
            query = probe["query"]

            models_shuffled = common.MODELS_20[:]
            rng.shuffle(models_shuffled)
            anon_ids = [f"R{k+1:02d}" for k in range(len(models_shuffled))]

            responses = []
            key = {}

            if ds in OPEN_ENDED:
                rubric_type = "open_ended"
                ground_truth = None
                guidance = CATEGORY_GUIDANCE[ds]
                for anon_id, m in zip(anon_ids, models_shuffled):
                    text = (setA[ds]["raw_outputs"][m][local_i] or "")[:OPEN_ENDED_RESPONSE_CAP]
                    responses.append({"anon_id": anon_id, "response": text})
                    key[anon_id] = m
            else:
                rubric_type = "ground_truth"
                orig_idx = ds_idx_map[local_i]
                recs_by_model = gt_lookup[ds]
                ground_truth = recs_by_model[models_shuffled[0]][orig_idx].get("ground_truth")
                if ground_truth is None:
                    ground_truth = "(no reference recorded -- judge on general correctness)"
                for anon_id, m in zip(anon_ids, models_shuffled):
                    rec = recs_by_model[m][orig_idx]
                    prediction = rec.get("prediction")
                    raw_output = rec.get("raw_output") or ""
                    responses.append({
                        "anon_id": anon_id,
                        "prediction": prediction if prediction is not None else "(no prediction extracted)",
                        "reasoning_tail": ("..." + raw_output[-REASONING_TAIL_CAP:]) if len(raw_output) > REASONING_TAIL_CAP else raw_output,
                    })
                    key[anon_id] = m

            packet = {
                "dataset": ds,
                "p_rank": p_rank,
                "rubric_type": rubric_type,
                "rubric": RUBRIC_NOTE[rubric_type],
                "category_guidance": guidance,
                "query": query,
                "ground_truth": ground_truth,
                "responses": responses,
            }

            fname = f"{ds}__{p_rank}.json"
            with open(PACKET_DIR / fname, "w", encoding="utf-8") as f:
                json.dump(packet, f, ensure_ascii=False, indent=1)
            with open(KEY_DIR / fname, "w", encoding="utf-8") as f:
                json.dump(key, f, ensure_ascii=False, indent=1)
            n_packets += 1

    print(f"Wrote {n_packets} judging packets -> {PACKET_DIR}")
    print(f"Answer keys (do not read while judging) -> {KEY_DIR}")
    print(f"Write judged results (one file per packet, {{anon_id: score}}) -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
