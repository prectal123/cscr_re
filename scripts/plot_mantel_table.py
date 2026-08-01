"""Render the (circular, Set-A-vs-Set-A) Mantel/RSA test result as a table
image, with the circularity caveat spelled out directly in the image."""
import matplotlib.pyplot as plt

OUT_PATH = "local_descriptors/routerbench-analysis/mantel_rsa_setA_circular_table.png"

rows = [
    ("Ceiling", "+0.7144", "0.00031", "significant, but CIRCULAR -- both sides built from Set A accuracy"),
    ("Perplexity", "+0.0637", "0.76057", "not significant"),
]

col_labels = ["FP type", "Spearman rho\n(FP-space vs true-capability structure)", "Mantel p\n(100,000 permutations)", "caveat"]
cell_text = [list(r) for r in rows]

fig, ax = plt.subplots(figsize=(15, 3.4))
ax.axis("off")
fig.suptitle("Mantel/RSA structure test -- Set A vs Set A (CIRCULAR, not a held-out test)\n"
             "Ceiling FP is built directly from Set A accuracy, so this largely tests self-consistency of FP construction, not predictive validity",
             fontsize=12, fontweight="bold", y=1.08)

table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center",
                  colWidths=[0.15, 0.28, 0.22, 0.45])
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 2.2)

header_color = "#1a2744"
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#333333")
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
    elif r == 1:
        cell.set_facecolor("#fff3cd")  # yellow/caution, not green -- flagged as circular

fig.text(0.5, -0.05,
          "NOT used as evidence for the capability-alignment hypothesis -- kept only as a record that this specific test design was flawed "
          "(caught by user 2026-07-31). A corrected Set-A-vs-Set-B version is the planned follow-up.",
          ha="center", fontsize=9.5, style="italic", color="#444444")

plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
