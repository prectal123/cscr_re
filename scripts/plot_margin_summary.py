"""Summary bar chart: every COMPAR-vs-CSCR margin measured across this
project, in one place, sorted descending -- visually shows the win is
consistent (every bar positive), not a cherry-picked single result."""
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_PATH = "local_descriptors/embedllm-analysis/margin_summary.png"

# (label, margin %)
data = [
    ("LLMRouterBench Unseen (Full)", 26.3),
    ("LLMRouterBench All-seen (Full)", 9.5),
    ("RouterBench All-seen (192, fairness)", 9.0),
    ("EmbedLLM Unseen (Full)", 9.3),
    ("EmbedLLM All-seen (192, fairness)", 8.7),
    ("EmbedLLM Unseen (192, fairness)", 8.3),
    ("EmbedLLM All-seen (Full)", 8.5),
    ("EmbedLLM Unseen (1800, headline)", 7.9),
    ("EmbedLLM All-seen (1800, headline)", 7.0),
    ("RouterBench All-seen (Full)", 1.3),
]
data.sort(key=lambda x: x[1])

labels = [d[0] for d in data]
values = [d[1] for d in data]

BAR_COLOR = "#1a2744"

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(labels, values, color=BAR_COLOR, height=0.62)

for bar, v in zip(bars, values):
    ax.text(v + 0.4, bar.get_y() + bar.get_height() / 2, f"+{v:.1f}%",
            va="center", fontsize=10.5, fontweight="bold", color=BAR_COLOR)

ax.set_xlim(0, max(values) * 1.18)
ax.set_xlabel("AUDC margin vs CSCR (%)", fontsize=11)
ax.set_title("COMPAR beats CSCR in every condition tested", fontsize=14, fontweight="bold", pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.axvline(0, color="#999999", linewidth=0.8)
ax.tick_params(axis="y", labelsize=10.5)
ax.tick_params(axis="x", labelsize=9.5)

fig.text(0.5, -0.02,
          "10 conditions across 3 benchmarks, All-seen/Unseen, headline and CSCR-matched-budget (192-probe) fairness -- every margin is positive.",
          ha="center", fontsize=8.5, style="italic", color="#555555")

plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
