"""Appendix: full 4-way collapse diagnostic table (top3_share, models used,
rho, mislanding rate, outlier-drag corr) -- combines the original collapse
ablation (section 27.2) with today's mechanism-level diagnostic (section 31).
Same style policy: dark header only, plain white body."""
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_PATH = "local_descriptors/embedllm-analysis/table_collapse.png"

col_labels = ["Variant", "top3_share", "Models Used\n(of 35)", "rho\n(select. vs acc.)",
              "Mislanding\nRate", "Outlier-Drag\nCorr"]
rows = [
    ["Vanilla", "0.686", "23", "0.447", "0.668", "-0.149"],
    ["Catfilter only", "0.816", "15", "0.515", "0.532", "-0.093"],
    ["Min(0.3,3) only", "0.760", "13", "0.663", "0.541", "+0.013 (n.s.)"],
    ["Combined (TAR)", "0.767", "9", "0.669", "0.544", "-0.040"],
]

fig, ax = plt.subplots(figsize=(11, 2.6))
ax.axis("off")
fig.suptitle("Collapse mechanism diagnostic (seed=0, uncompressed headline FP)", fontsize=13, fontweight="bold", y=1.1)

table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 2.3)

header_color = "#1a2744"
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("white")

fig.text(0.5, -0.06,
          "Catfilter and Min(0.3,3) independently cut mislanding by a similar amount; combining them adds no further benefit -- "
          "consistent with both being the same top-min(30%,3) selection operator applied with different ranking signals.",
          ha="center", fontsize=8.5, style="italic", color="#555555")

plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
