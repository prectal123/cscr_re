"""Render the Ceiling FP top-k neighbor sweep (topk_knn_sweep.py) as a table
image, matching the style of routerbench_knn_summary_table.png."""
import matplotlib.pyplot as plt

OUT_PATH = "local_descriptors/routerbench-analysis/ceiling_topk_sweep_table.png"

# from topk_knn_sweep.py output (Ceiling FP section)
rows = [
    # k, FP-topk rho, random-k rho, delta_vs_random (p), delta_vs_uniform10 (p)
    (1, 0.4369, 0.2118, 0.2251, 0.0007, 0.0444, 0.3938),
    (2, 0.4521, 0.2877, 0.1644, 0.0003, 0.0596, 0.1239),
    (3, 0.4655, 0.3204, 0.1451, 0.0010, 0.0730, 0.0812),
    (5, 0.4848, 0.3620, 0.1227, 0.0031, 0.0923, 0.0280),
    (7, 0.4906, 0.3825, 0.1081, 0.0073, 0.0982, 0.0205),
    (10, 0.4906, 0.3924, 0.0982, 0.0205, 0.0982, 0.0205),
]
UNIFORM10_RHO = 0.3924

col_labels = ["k (neighbors)", "FP-topk rho", "random-k rho\n(size-matched control)",
              "delta vs random-k\n(p-value)", "delta vs uniform-10\n(p-value)"]

cell_text = []
for k, fp_rho, rand_rho, d_rand, p_rand, d_uni, p_uni in rows:
    cell_text.append([
        f"{k}",
        f"{fp_rho:.4f}",
        f"{rand_rho:.4f}",
        f"+{d_rand:.4f} (p={p_rand:.4f}{'*' if p_rand < 0.05 else ''})",
        f"+{d_uni:.4f} (p={p_uni:.4f}{'*' if p_uni < 0.05 else ''})",
    ])

fig, ax = plt.subplots(figsize=(12.5, 3.4))
ax.axis("off")
fig.suptitle("Ceiling FP -- top-k neighbor sweep\n"
             "(FP-selected top-k neighbors vs size-matched random-k neighbors, RouterBench)",
             fontsize=14, fontweight="bold", y=1.03)

table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

header_color = "#1a2744"
sig_color = "#e8f5e9"
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#333333")
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        k, p_rand, p_uni = rows[r - 1][0], rows[r - 1][4], rows[r - 1][6]
        if p_rand < 0.05:
            cell.set_facecolor(sig_color)
        if c in (3, 4):
            cell.set_text_props(fontweight="bold", color="#1b5e20")

fig.text(0.5, -0.02,
          f"[reference] full-10 uniform baseline mean rho = {UNIFORM10_RHO:.4f}   |   "
          f"* p < 0.05 (paired t-test, n=11 folds)   |   random-k = mean over 30 random draws per fold",
          ha="center", fontsize=9.5, style="italic", color="#444444")

plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
