"""EmbedLLM Unseen deferral curves by model pool size -- fixed-probe design
(probe selection isolated from which models get FP vectors)."""
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

IN_PATH = "local_descriptors/embedllm-analysis/modelcount_sweep_unseen_withcurves_results.json"
OUT_PATH = "local_descriptors/embedllm-analysis/deferral_modelcount_unseen_FINAL.png"
CSCR_UNSEEN = 0.4848

d = json.load(open(IN_PATH, encoding="utf-8"))
ORDER = ["15", "30", "50", "75", "111"]

fig, ax = plt.subplots(figsize=(9, 6))
colors = cm.viridis(np.linspace(0, 0.85, len(ORDER)))

for label, color in zip(ORDER, colors):
    r = d[label]
    c, a = r["costs_mean_curve"], r["accs_mean_curve"]
    disp = f"{label} models (full pool)" if label == "111" else f"{label} models"
    style = dict(marker="s", markersize=5, linewidth=2.6, linestyle="--", color="#c0392b") if label == "111" \
        else dict(marker="o", markersize=3.5, linewidth=1.6, color=color)
    ax.plot(c, a, label=disp, **style)

ax.axhline(CSCR_UNSEEN, color="#999999", linestyle=":", linewidth=1.2, label=f"CSCR unseen (paper) = {CSCR_UNSEEN}")

ax.set_xlabel("Cost", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("EmbedLLM Unseen Deferral Curves, by Model Pool Size (3-seed mean)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.25)

plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
