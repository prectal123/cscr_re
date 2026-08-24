"""Fairness (CSCR's own 192-probe budget) results table -- same style
policy as plot_table_headline.py: dark header only, plain white body,
no highlighting."""
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_PATH = "local_descriptors/embedllm-analysis/table_fairness.png"

col_labels = ["Method", "Benchmark", "Protocol", "Probes", "AUDC (3-seed)", "vs CSCR paper"]
rows = [
    ["COMPAR", "EmbedLLM", "All-seen", "192", "0.5883 ± 0.0049", "+8.7%"],
    ["COMPAR", "EmbedLLM", "Unseen", "192", "0.5251 ± 0.0095", "+8.3%"],
    ["COMPAR", "RouterBench", "All-seen", "192", "0.7747 ± 0.0026", "+9.0%"],
    ["CSCR", "RouterBench", "All-seen", "1800", "0.6368 ± 0.0012", "-10.4%"],
]

fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis("off")

table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.0)

header_color = "#1a2744"
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("white")

fig.text(0.5, -0.14,
          "COMPAR rows: shrunk to CSCR's own reported budget (192, confirmed from the paper) -- still beats CSCR every time.\n"
          "CSCR row: given COMPAR's larger budget (1800) instead -- performance does not improve, even drops slightly vs CSCR's own paper value.\n"
          "RouterBench cannot run Unseen (only 11 models); LLMRouterBench isn't a CSCR-paper benchmark, so it has no 192-probe reference point.",
          ha="center", fontsize=8.5, style="italic", color="#555555")

plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
