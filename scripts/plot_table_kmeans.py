"""Appendix: K-Means adaptive-clustering FP (no given category labels)
vs the real-category headline and CSCR. Same style policy."""
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_PATH = "local_descriptors/embedllm-analysis/table_kmeans.png"

col_labels = ["Protocol", "K-Means FP\n(K=80, 3-seed)", "Real-Category\nHeadline", "CSCR"]
rows = [
    ["All-seen", "0.5578 ± 0.0050", "0.5787", "0.541"],
    ["Unseen", "0.5253 ± 0.0044", "0.5232", "0.4848"],
]

fig, ax = plt.subplots(figsize=(8, 1.9))
ax.axis("off")
fig.suptitle("EmbedLLM: K-Means-derived categories (no given labels) vs real categories",
             fontsize=13, fontweight="bold", y=1.12)

table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

header_color = "#1a2744"
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("white")

fig.text(0.5, -0.08,
          "K-Means (MiniLM embedding clusters, K matched to the real category count) beats CSCR with zero given category labels.\n"
          "All-seen shows a real gap vs the real-category headline (text similarity != capability-divergence axis); Unseen shows almost none.",
          ha="center", fontsize=8.5, style="italic", color="#555555")

plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
