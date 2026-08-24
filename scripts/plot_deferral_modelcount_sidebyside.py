"""EmbedLLM model-pool-size deferral curves, All-seen vs Unseen side by
side -- the key asymmetry: All-seen saturates/declines past 75 models,
Unseen keeps climbing through 111. Same model-count points (15/30/50/75/111)
for a direct visual pairing."""
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ALLSEEN_PATH = "local_descriptors/embedllm-analysis/modelcount_sweep_allseen_withcurves_results.json"
UNSEEN_PATH = "local_descriptors/embedllm-analysis/modelcount_sweep_unseen_withcurves_results.json"
OUT_PATH = "local_descriptors/embedllm-analysis/deferral_modelcount_sidebyside_FINAL.png"
CSCR_ALLSEEN = 0.541
CSCR_UNSEEN = 0.4848

d_as = json.load(open(ALLSEEN_PATH, encoding="utf-8"))
d_un = json.load(open(UNSEEN_PATH, encoding="utf-8"))
ORDER = ["15", "30", "50", "75", "111"]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
colors = cm.viridis(np.linspace(0, 0.85, len(ORDER)))

for ax, d, title, cscr_ref, cscr_label in [
    (axes[0], d_as, "All-seen (peaks at 75, dips at 111)", CSCR_ALLSEEN, "CSCR all-seen"),
    (axes[1], d_un, "Unseen (keeps rising through 111)", CSCR_UNSEEN, "CSCR unseen"),
]:
    for label, color in zip(ORDER, colors):
        r = d[label]
        c, a = r["costs_mean_curve"], r["accs_mean_curve"]
        if label == "111":
            ax.plot(c, a, marker="s", markersize=5, linewidth=2.6, linestyle="--", color="#c0392b",
                    label=f"{label} models (full pool)")
        else:
            ax.plot(c, a, marker="o", markersize=3.5, linewidth=1.8, color=color, label=f"{label} models")
    ax.axhline(cscr_ref, color="#999999", linestyle=":", linewidth=1.2, label=f"{cscr_label} (paper) = {cscr_ref}")
    ax.set_xlabel("Cost", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

fig.suptitle("EmbedLLM Model-Pool-Size Sweep: All-seen vs Unseen (3-seed mean)", fontsize=14.5, fontweight="bold", y=1.02)

plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
