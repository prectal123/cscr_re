"""EmbedLLM Unseen deferral curves by probe count -- official uncompressed
+ TAR-loss pipeline, freshly regenerated (previous file had lost most
points to an earlier overwrite bug)."""
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

IN_PATH = "local_descriptors/embedllm-analysis/probe_scale_sweep_unseen_multiseed_withcurves_results.json"
OUT_PATH = "local_descriptors/embedllm-analysis/deferral_probecount_unseen_FINAL.png"
CSCR_UNSEEN = 0.4848

d = json.load(open(IN_PATH, encoding="utf-8"))

ORDER = ["uncompressed-96", "uncompressed-192", "uncompressed-300", "uncompressed-1800",
         "uncompressed-4000", "uncompressed-8000", "uncompressed-15000", "uncompressed-25000"]
DISPLAY = {
    "uncompressed-96": "96 probes", "uncompressed-192": "192 probes (CSCR budget)",
    "uncompressed-300": "300 probes (outlier, see notes)", "uncompressed-1800": "1800 probes (headline)",
    "uncompressed-4000": "4000 probes", "uncompressed-8000": "8000 probes",
    "uncompressed-15000": "15000 probes", "uncompressed-25000": "25000 probes",
}

fig, ax = plt.subplots(figsize=(9, 6))
colors = cm.viridis(np.linspace(0, 0.85, len(ORDER)))

for label, color in zip(ORDER, colors):
    r = d[label]
    c, a = r["costs_mean_curve"], r["accs_mean_curve"]
    ax.plot(c, a, marker="o", markersize=3.5, linewidth=1.6, color=color, label=DISPLAY[label])

r = d["V2-full-uncompressed"]
ax.plot(r["costs_mean_curve"], r["accs_mean_curve"], marker="s", markersize=5, linewidth=2.6,
        color="#c0392b", linestyle="--", label="V2 full data (29673 probes)")

ax.axhline(CSCR_UNSEEN, color="#999999", linestyle=":", linewidth=1.2, label=f"CSCR unseen (paper) = {CSCR_UNSEEN}")

ax.set_xlabel("Cost", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("EmbedLLM Unseen Deferral Curves, by Probe Count (3-seed mean)", fontsize=13, fontweight="bold")
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.25)

plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
