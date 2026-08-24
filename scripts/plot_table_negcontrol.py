"""Appendix: negative control cascade (random probe selection, noise FP,
shuffled-category FP) -- isolates what signal actually drives All-seen vs
Unseen performance. Same style policy."""
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_PATH = "local_descriptors/embedllm-analysis/table_negcontrol.png"

col_labels = ["Test", "Protocol", "AUDC", "vs Real FP"]
rows = [
    ["Random probe selection", "All-seen", "0.5909", "Higher than top-var"],
    ["Pure noise FP", "All-seen", "0.5769 ± 0.0049", "~Same (0.5787)"],
    ["Pure noise FP", "Unseen", "0.3721 ± 0.0329", "Collapses, below CSCR"],
    ["Shuffled-category FP", "All-seen", "0.5730", "~Same (0.5787)"],
    ["Shuffled-category FP", "Unseen", "0.5261 ± 0.0008", "~Same (0.5232)"],
]

fig, ax = plt.subplots(figsize=(9.5, 2.9))
ax.axis("off")
fig.suptitle("Negative control cascade (EmbedLLM)", fontsize=13, fontweight="bold", y=1.1)

table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 2.1)

header_color = "#1a2744"
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("white")

fig.text(0.5, -0.06,
          "All-seen is largely insensitive to real FP content (classification collapse); Unseen needs real signal but not fine category alignment.",
          ha="center", fontsize=8.5, style="italic", color="#555555")

plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
