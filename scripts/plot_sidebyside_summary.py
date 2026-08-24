"""Side-by-side COMPAR (blue) vs CSCR (red) AUDC bars across every
condition tested -- shows actual values with error bars (std, where a
multi-seed number exists) instead of a single pre-computed margin, so
variance (e.g. LLMRouterBench Unseen) is visible rather than hidden."""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_PATH = "local_descriptors/embedllm-analysis/sidebyside_summary.png"

# (label, compar_mean, compar_std, cscr_mean, cscr_std_or_None)
data = [
    ("EmbedLLM All-seen (1800, headline)", 0.5787, 0.0014, 0.541, None),
    ("EmbedLLM Unseen (1800, headline)", 0.5232, 0.0053, 0.4848, None),
    ("EmbedLLM All-seen (Full)", 0.5867, 0.0030, 0.541, None),
    ("EmbedLLM Unseen (Full)", 0.5299, 0.0080, 0.4848, None),
    ("EmbedLLM All-seen (192, fairness)", 0.5883, 0.0049, 0.541, None),
    ("EmbedLLM Unseen (192, fairness)", 0.5251, 0.0095, 0.4848, None),
    ("RouterBench All-seen (Full)", 0.7205, 0.0013, 0.711, None),
    ("RouterBench All-seen (192, fairness)", 0.7747, 0.0026, 0.711, None),
    ("LLMRouterBench All-seen (Full)", 0.7349, 0.0061, 0.6712, 0.0123),
    ("LLMRouterBench Unseen (Full)", 0.6719, 0.0357, 0.5319, 0.0669),
]

labels = [d[0] for d in data]
compar_m = [d[1] for d in data]
compar_s = [d[2] for d in data]
cscr_m = [d[3] for d in data]
cscr_s = [d[4] if d[4] is not None else 0 for d in data]

y = np.arange(len(data))
h = 0.35

COMPAR_COLOR = "#1a2744"
CSCR_COLOR = "#c0392b"

fig, ax = plt.subplots(figsize=(9.5, 6))
ax.barh(y + h / 2, compar_m, xerr=compar_s, height=h, color=COMPAR_COLOR, label="COMPAR",
        error_kw=dict(elinewidth=1.2, capsize=3))
ax.barh(y - h / 2, cscr_m, xerr=cscr_s, height=h, color=CSCR_COLOR, label="CSCR",
        error_kw=dict(elinewidth=1.2, capsize=3))

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("AUDC", fontsize=11)
ax.set_title("COMPAR vs CSCR, every condition tested", fontsize=14, fontweight="bold", pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", fontsize=10.5, frameon=False)
ax.tick_params(axis="x", labelsize=9.5)

fig.text(0.5, -0.03,
          "Error bars = std across 3 seeds where available; CSCR paper-reported points (no variance published) shown without error bars. "
          "LLMRouterBench CSCR values are our own reproduction (900-probe, no official paper baseline).",
          ha="center", fontsize=8.5, style="italic", color="#555555")

plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
