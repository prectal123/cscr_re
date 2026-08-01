"""Render the seed=0 full_loo_with_rank_results.json (Ceiling + Perplexity,
oracle_match_rate + AUC) as a table image."""
import json
import matplotlib.pyplot as plt

IN_PATH = "local_descriptors/routerbench-analysis/full_loo_with_rank_results.json"
OUT_PATH = "local_descriptors/routerbench-analysis/full_loo_with_rank_table_seed0.png"

with open(IN_PATH) as f:
    data = json.load(f)

ceiling_by_name = {r["held_out"]: r for r in data["Ceiling"]}
perp_by_name = {r["held_out"]: r for r in data["Perplexity"]}
order = [r["held_out"] for r in data["Ceiling"]]

DISPLAY_NAME = {
    "meta__code-llama-instruct-34b-chat": "code-llama-34b",
    "meta__llama-2-70b-chat": "llama-2-70b",
    "mistralai__mistral-7b-chat": "mistral-7b",
    "mistralai__mixtral-8x7b-chat": "mixtral-8x7b",
    "zero-one-ai__Yi-34B-Chat": "Yi-34B-Chat",
    "WizardLM__WizardLM-13B-V1.2": "WizardLM",
}

col_labels = ["model (held out)", "n_oracle_is_M", "Ceiling\noracle_match_rate", "Ceiling AUC (p)",
              "Perplexity\noracle_match_rate", "Perplexity AUC (p)",
              "Δ oracle_match_rate\n(Ceiling - Perp)", "Δ AUC\n(Ceiling - Perp)"]

def fmt_auc(auc, p, keep_p=False):
    if keep_p:
        return f"{auc:.3f} (p={p:.4f})"
    return f"{auc:.3f}"

cell_text = []
for name in order:
    c, pxr = ceiling_by_name[name], perp_by_name[name]
    disp = DISPLAY_NAME.get(name, name)
    # keep the p-value only on the one AUC cell that ISN'T colored (not significant):
    # claude-v2's Perplexity AUC (p=0.1174) -- the only white/uncolored AUC box in
    # the whole table, so its p-value is the only one that actually needs spelling out.
    keep_p_here = (name == "claude-v2")
    delta_rate = c["oracle_match_rate"] - pxr["oracle_match_rate"]
    delta_auc = c["auc_heldout_correctness"] - pxr["auc_heldout_correctness"]
    cell_text.append([
        disp,
        f"{c['n_oracle_is_M']}",
        f"{c['oracle_match_rate']*100:.1f}%",
        fmt_auc(c["auc_heldout_correctness"], c["point_biserial_p_heldout"]),
        f"{pxr['oracle_match_rate']*100:.1f}%",
        fmt_auc(pxr["auc_heldout_correctness"], pxr["point_biserial_p_heldout"], keep_p=keep_p_here),
        f"{delta_rate*100:+.1f}%p",
        f"{delta_auc:+.3f}",
    ])

# pooled row
c_pooled_num = sum(ceiling_by_name[n]["n_oracle_is_M"] * ceiling_by_name[n]["oracle_match_rate"] for n in order)
c_pooled_den = sum(ceiling_by_name[n]["n_oracle_is_M"] for n in order)
p_pooled_num = sum(perp_by_name[n]["n_oracle_is_M"] * perp_by_name[n]["oracle_match_rate"] for n in order)
p_pooled_den = sum(perp_by_name[n]["n_oracle_is_M"] for n in order)
c_pooled_rate = c_pooled_num / c_pooled_den
p_pooled_rate = p_pooled_num / p_pooled_den
c_auc_mean = sum(ceiling_by_name[n]["auc_heldout_correctness"] for n in order) / len(order)
p_auc_mean = sum(perp_by_name[n]["auc_heldout_correctness"] for n in order) / len(order)
cell_text.append(["POOLED (n-weighted)", f"{c_pooled_den}",
                   f"{c_pooled_rate*100:.1f}%", "-",
                   f"{p_pooled_rate*100:.1f}%", "-",
                   f"{(c_pooled_rate - p_pooled_rate)*100:+.1f}%p",
                   f"{(c_auc_mean - p_auc_mean):+.3f} (mean AUC)"])

fig, ax = plt.subplots(figsize=(18.5, 6.2))
ax.axis("off")
fig.suptitle("RouterBench full 11-fold LOO (seed=0, beta=1.0 load-balancing)\n"
             "oracle_match_rate (strict argmax hit) vs AUC (rank-quality over all 7,300 Set B prompts)",
             fontsize=13.5, fontweight="bold", y=1.04)

table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 1.85)

header_color = "#1a2744"
pooled_color = "#dde3ee"
n_rows = len(cell_text)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#333333")
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
    elif r == n_rows:  # pooled row
        cell.set_facecolor(pooled_color)
        cell.set_text_props(fontweight="bold")
    else:
        name = order[r - 1]
        c_auc, c_p = ceiling_by_name[name]["auc_heldout_correctness"], ceiling_by_name[name]["point_biserial_p_heldout"]
        px_auc, px_p = perp_by_name[name]["auc_heldout_correctness"], perp_by_name[name]["point_biserial_p_heldout"]
        if c == 3:  # Ceiling AUC col
            if c_p < 0.05 and c_auc > 0.5:
                cell.set_facecolor("#e8f5e9")
                cell.set_text_props(fontweight="bold", color="#1b5e20")
            elif c_p < 0.05 and c_auc < 0.5:
                cell.set_facecolor("#ffebee")
                cell.set_text_props(fontweight="bold", color="#b71c1c")
        if c == 5:  # Perplexity AUC col
            if px_p < 0.05 and px_auc > 0.5:
                cell.set_facecolor("#e8f5e9")
                cell.set_text_props(fontweight="bold", color="#1b5e20")
            elif px_p < 0.05 and px_auc < 0.5:
                cell.set_facecolor("#ffebee")
                cell.set_text_props(fontweight="bold", color="#b71c1c")
        if c in (6, 7):  # delta columns: green = Ceiling better, red = Perplexity better
            delta_rate = ceiling_by_name[name]["oracle_match_rate"] - perp_by_name[name]["oracle_match_rate"]
            delta_auc = c_auc - px_auc
            delta_val = delta_rate if c == 6 else delta_auc
            if delta_val > 0:
                cell.set_facecolor("#e8f5e9")
                cell.set_text_props(fontweight="bold", color="#1b5e20")
            elif delta_val < 0:
                cell.set_facecolor("#ffebee")
                cell.set_text_props(fontweight="bold", color="#b71c1c")

fig.text(0.5, -0.03,
          "AUC: green = significantly above chance (0.5, p<0.05), red = significantly below chance, "
          "white = not significant (p shown)   |   single seed (seed=0) -- multi-seed stability not yet verified",
          ha="center", fontsize=9.5, style="italic", color="#444444")

plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
