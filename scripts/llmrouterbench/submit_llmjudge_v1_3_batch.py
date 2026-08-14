"""Submit the V1.3 LLM-judge scoring batch: a single unified judge (Claude
Sonnet 5) re-scores the same 66 probes (3/category x 22 categories,
v1_3_probe_selection.json) already used to generate the 20-model raw_outputs
in setA_setB_split.pkl. This replaces the 22 categories' heterogeneous
original scoring (exact-match, unit tests, an external judge of unknown
identity for the arenahard family) with one consistent 1-10 rubric-graded
axis -- the "universal, subgroup-unbiased scoring" direction noted in
PROGRESS.md section 17.

Two rubric branches:
  - GROUND_TRUTH categories (18 of 22): probe + reference answer + candidate
    response -> does the candidate reach an equivalent correct conclusion
    (partial credit for right approach / wrong final answer, etc.)
  - OPEN_ENDED categories (4: the arenahard family, ground_truth is None in
    the source records): probe + candidate response only -> absolute quality
    rubric (correctness where verifiable, completeness, clarity, constraint
    adherence)

All 1320 (probe, model) pairs go in ONE Batch API request -- 50% cheaper than
sync calls, no concurrency/rate-limit handling needed, and each request is
independently stateless so there's no risk of the judge inferring patterns
across models the way there would be in a shared-context chat.

Usage: python submit_llmjudge_v1_3_batch.py
Writes local_descriptors/llmrouterbench_lite20/v1_3_batch_id.json with the
batch id; run build_llmjudge_fp_v1_3_from_batch.py once it's done.
"""
import json
import os
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

import common_lite20 as common

DATA_DIR = Path("local_descriptors/llmrouterbench_lite20")
JUDGE_MODEL = "claude-sonnet-5"
API_KEY_FILE = Path(r"C:\Users\user\anthropic_key.txt")


def _load_api_key_env():
    """Read the API key from a local file at runtime (never printed/logged) if
    ANTHROPIC_API_KEY isn't already set in the environment."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if API_KEY_FILE.exists():
        os.environ["ANTHROPIC_API_KEY"] = API_KEY_FILE.read_text(encoding="utf-8").strip()
RESPONSE_CHAR_CAP = 8000  # ~2000 tokens; caps rare very-long CoT outputs

OPEN_ENDED = {"arenahard", "arenahard_coding", "arenahard_creative_writing", "arenahard_math"}

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief (2-4 sentence) justification, written before deciding the score.",
        },
        "score": {
            "type": "integer",
            "description": "Quality score from 1 (fails entirely) to 10 (excellent, fully correct/complete).",
        },
    },
    "required": ["reasoning", "score"],
    "additionalProperties": False,
}

GROUND_TRUTH_RUBRIC = """You are grading a single AI-generated response against a reference answer. \
You do not know, and must not try to guess, which model produced this response -- grade the \
content only.

Question:
{query}

Reference answer:
{ground_truth}

Candidate response:
{response}

Score the candidate response from 1 to 10 on whether it reaches a conclusion equivalent to the \
reference answer:
- 9-10: Final answer is correct (equivalent to the reference, even if expressed differently), \
reasoning (if any) is sound.
- 6-8: Final answer is correct but reasoning has a flaw, or the answer is correct but incomplete or \
imprecisely stated.
- 3-5: Reasoning shows partial understanding or the right approach, but the final answer is wrong.
- 1-2: Final answer is wrong and reasoning is absent, irrelevant, or fundamentally mistaken.

Write your reasoning first, then the score."""

OPEN_ENDED_RUBRIC = """You are grading a single AI-generated response to an open-ended prompt. There \
is no fixed reference answer -- evaluate the response on its own merits. You do not know, and must \
not try to guess, which model produced this response -- grade the content only.

Category: {category}
{category_guidance}

Prompt:
{query}

Candidate response:
{response}

Score the candidate response from 1 to 10:
- 9-10: Fully addresses the prompt, high quality, no errors, meets every explicit constraint in the \
prompt (format, language, style, length, etc.).
- 6-8: Addresses the prompt well with minor gaps, small errors, or a missed minor constraint.
- 3-5: Partially addresses the prompt; notable gaps, errors, or ignored constraints.
- 1-2: Fails to address the prompt, off-topic, or fundamentally broken.

Write your reasoning first, then the score."""

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


def load_probes():
    with open(DATA_DIR / "v1_3_probe_selection.json", encoding="utf-8") as f:
        return json.load(f)["selected"]


def load_setA():
    import pickle

    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        return pickle.load(f)["setA"]


def build_ground_truth_lookup(datasets_needed):
    """dataset -> list[str|None] ground_truth, aligned to setA local index via idx mapping."""
    lookup = {}
    for ds in datasets_needed:
        recs_by_model = common.load_dataset_records(ds)
        recs = recs_by_model[common.MODELS_20[0]]  # ground_truth is identical across models
        lookup[ds] = [r.get("ground_truth") for r in recs]
    return lookup


def main():
    selected = load_probes()
    setA = load_setA()

    gt_datasets = [c["dataset"] for c in selected if c["dataset"] not in OPEN_ENDED]
    gt_lookup = build_ground_truth_lookup(gt_datasets)

    requests = []
    meta = []  # parallel: (dataset, probe_rank, model) in custom_id order
    for cat in selected:
        ds = cat["dataset"]
        ds_idx_map = setA[ds]["idx"]  # local_idx_in_setA -> original record index
        for p_rank, probe in enumerate(cat["probes"]):
            local_i = probe["local_idx_in_setA"]
            query = probe["query"]
            if ds in OPEN_ENDED:
                guidance = CATEGORY_GUIDANCE[ds]
            else:
                orig_idx = ds_idx_map[local_i]
                ground_truth = gt_lookup[ds][orig_idx]
                if ground_truth is None:
                    ground_truth = "(no reference recorded -- judge on general correctness)"

            for m in common.MODELS_20:
                response = (setA[ds]["raw_outputs"][m][local_i] or "")[:RESPONSE_CHAR_CAP]
                custom_id = f"{ds}__{p_rank}__{common.NAME_TO_SAFE[m]}".replace(".", "-")

                if ds in OPEN_ENDED:
                    prompt = OPEN_ENDED_RUBRIC.format(
                        category=ds, category_guidance=guidance, query=query, response=response,
                    )
                else:
                    prompt = GROUND_TRUTH_RUBRIC.format(
                        query=query, ground_truth=ground_truth, response=response,
                    )

                requests.append(
                    Request(
                        custom_id=custom_id,
                        params=MessageCreateParamsNonStreaming(
                            model=JUDGE_MODEL,
                            max_tokens=1024,
                            thinking={"type": "disabled"},
                            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                            messages=[{"role": "user", "content": prompt}],
                        ),
                    )
                )
                meta.append((ds, p_rank, m))

    print(f"Built {len(requests)} judge requests (66 probes x 20 models).")

    _load_api_key_env()
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    print(f"Submitted batch: {batch.id}  status={batch.processing_status}")

    with open(DATA_DIR / "v1_3_batch_id.json", "w", encoding="utf-8") as f:
        json.dump({"batch_id": batch.id, "custom_id_order": [f"{d}__{r}__{common.NAME_TO_SAFE[m]}".replace(".", "-") for d, r, m in meta]}, f)
    print(f"Saved batch id -> {DATA_DIR / 'v1_3_batch_id.json'}")
    print("Batches usually finish within an hour (max 24h). Run "
          "build_llmjudge_fp_v1_3_from_batch.py once processing_status == 'ended'.")


if __name__ == "__main__":
    main()
