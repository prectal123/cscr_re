"""Render the 3-seed-averaged Ceiling vs Perplexity AUC/oracle_match_rate
comparison as a table image, sorted by AUC delta descending."""
import json
import matplotlib.pyplot as plt

IN_PATH = "local_descriptors/routerbench-analysis/seed_avg_ceiling_vs_perp.json"
OUT_PATH = "local_descriptors/routerbench-analysis/seed_avg_ceiling_vs_perp_table.png"

DISPLAY_NAME = {
    "meta__code-llama-instruct-34b-chat": "code-llama-34b",
    "meta__llama-2-70b-chat": "llama-2-70b",
    "mistralai__mistral-7b-chat": "mistral-7b",
    "mistralai__mixtral-8x7b-chat": "mixtral-8x7b",
    "zero-one-ai__Yi-34B-Chat": "Yi-34B-Chat",
    "WizardLM__WizardLM-13B-V1.2": "WizardLM",
}

rows = json.load(open(IN_PATH))
# rows: [name, c_auc, p_auc, d_auc, c_rate, p_rate, d_rate, n_c]

col_labels = ["model", "Ceiling AUC\n(3-seed mean)", "Perplexity AUC\n(3-seed mean)", "Delta AUC",
              "Ceiling rate\n(3-seed mean)", "Perplexity rate\n(3-seed mean)", "Delta rate"]
cell_text = []
for n, ca, pa, da, cr, pr, dr, nc in rows:
    cell_text.append([DISPLAY_NAME.get(n, n), f"{ca:.4f}", f"{pa:.4f}", f"{da:+.4f}",
                       f"{cr*100:.1f}%", f"{pr*100:.1f}%", f"{dr*100:+.1f}%p"])

mean_d_auc = sum(r[3] for r in rows) / len(rows)
mean_d_rate = sum(r[6] for r in rows) / len(rows)
cell_text.append(["MEAN (11 models)", "-", "-", f"{mean_d_auc:+.4f}", "-", "-", f"{mean_d_rate*100:+.1f}%p"])

fig, ax = plt.subplots(figsize=(17, 6.2))
ax.axis("off")
fig.suptitle("RouterBench LOO: 3-seed-averaged Ceiling vs Perplexity (beta=1.0)\n"
             "sorted by AUC delta (Ceiling - Perplexity), descending",
             fontsize=13, fontweight="bold", y=1.04)

table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 1.9)

header_color = "#1a2744"
mean_color = "#dde3ee"
n_rows = len(cell_text)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#333333")
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
    elif r == n_rows:
        cell.set_facecolor(mean_color)
        cell.set_text_props(fontweight="bold")
    else:
        d_auc = rows[r - 1][3]
        d_rate = rows[r - 1][6]
        if c == 3:
            cell.set_facecolor("#e8f5e9" if d_auc > 0 else "#ffebee")
            cell.set_text_props(fontweight="bold", color="#1b5e20" if d_auc > 0 else "#b71c1c")
        if c == 6:
            cell.set_facecolor("#e8f5e9" if d_rate > 0 else "#ffebee")
            cell.set_text_props(fontweight="bold", color="#1b5e20" if d_rate > 0 else "#b71c1c")

fig.text(0.5, -0.03,
          "green = Ceiling wins on 3-seed average, red = Perplexity wins   |   values are simple means across seed 0/1/2 (not n-weighted pooling)",
          ha="center", fontsize=9.5, style="italic", color="#444444")

plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
