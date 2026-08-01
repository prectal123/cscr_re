"""Render the seed-ensemble pilot result as a table image."""
import json
import matplotlib.pyplot as plt

IN_PATH = "local_descriptors/routerbench-analysis/seed_ensemble_pilot_results.json"
OUT_PATH = "local_descriptors/routerbench-analysis/seed_ensemble_pilot_table.png"

DISPLAY_NAME = {
    "gpt-4-1106-preview": "gpt-4 (stable/strong)",
    "claude-v2": "claude-v2 (unstable)",
    "mistralai__mistral-7b-chat": "mistral-7b (consistently weak)",
    "WizardLM__WizardLM-13B-V1.2": "WizardLM (unstable)",
}
ORDER = ["gpt-4-1106-preview", "claude-v2", "mistralai__mistral-7b-chat", "WizardLM__WizardLM-13B-V1.2"]

data = json.load(open(IN_PATH))

col_labels = ["model", "seed 0/1/2 AUC", "mean-of-\nindividual AUC", "SCORE-LEVEL\nensemble AUC",
              "mean-of-\nindividual rate", "SCORE-LEVEL\nensemble rate"]
cell_text = []
for name in ORDER:
    d = data[name]
    seed_aucs = ", ".join(f"{m['auc']:.3f}" for m in d["per_seed_metrics"])
    cell_text.append([
        DISPLAY_NAME[name], seed_aucs,
        f"{d['mean_of_individual_auc']:.4f}", f"{d['ensembled_metrics']['auc']:.4f}",
        f"{d['mean_of_individual_rate']*100:.1f}%", f"{d['ensembled_metrics']['oracle_match_rate']*100:.1f}%",
    ])

fig, ax = plt.subplots(figsize=(16.5, 4.2))
ax.axis("off")
fig.suptitle("Seed-ensemble pilot: does averaging SIMILARITY SCORES across 3 seeds beat averaging final AUC numbers?\n"
             "(Ceiling FP, beta=1.0, 4 models covering stable/unstable/null patterns)",
             fontsize=12.5, fontweight="bold", y=1.08)

table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center",
                  colWidths=[0.24, 0.22, 0.14, 0.14, 0.14, 0.14])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.1)

header_color = "#1a2744"
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#333333")
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        name = ORDER[r - 1]
        d = data[name]
        mean_auc = d["mean_of_individual_auc"]
        ens_auc = d["ensembled_metrics"]["auc"]
        if c == 3:  # ensemble AUC column
            if ens_auc > mean_auc + 0.005:
                cell.set_facecolor("#e8f5e9")
                cell.set_text_props(fontweight="bold", color="#1b5e20")
            elif ens_auc < mean_auc - 0.005:
                cell.set_facecolor("#ffebee")
                cell.set_text_props(fontweight="bold", color="#b71c1c")

fig.text(0.5, -0.05,
          "green = ensemble beats mean-of-individual by >0.005, red = ensemble worse by >0.005   |   "
          "result: no reliable improvement -- flat for the already-stable model, worse for 2/3 unstable/null models",
          ha="center", fontsize=9.5, style="italic", color="#444444")

plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
