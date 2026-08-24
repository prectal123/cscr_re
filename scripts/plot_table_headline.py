"""Headline results table (Results_Reference_Table.md section 1) as an
image for Figma import. Style policy: dark header (white bold text) only,
body rows plain white, no row/cell highlighting."""
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_PATH = "local_descriptors/embedllm-analysis/table_headline.png"

col_labels = ["Benchmark", "Protocol", "Probes", "AUDC (3-seed)", "vs CSCR"]
rows = [
    ["EmbedLLM", "All-seen", "1800", "0.5787 ± 0.0014", "+7.0%"],
    ["EmbedLLM", "Unseen", "1800", "0.5232 ± 0.0053", "+7.9%"],
    ["EmbedLLM", "All-seen", "Full (V2)", "0.5867 ± 0.0030", "+8.5%"],
    ["EmbedLLM", "Unseen", "Full (V2)", "0.5299 ± 0.0080", "+9.3%"],
    ["RouterBench", "All-seen", "Full (V2)", "0.7205 ± 0.0013", "+1.3%"],
    ["LLMRouterBench", "All-seen", "Full (V2)", "0.7349 ± 0.0061", "+9.5%*"],
    ["LLMRouterBench", "Unseen", "Full (V2)", "0.6719 ± 0.0357", "+26.3%*"],
]

fig, ax = plt.subplots(figsize=(9, 3.2))
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

fig.text(0.5, -0.02,
          "* LLMRouterBench has no official paper baseline (not one of CSCR's benchmarks) -- margins are vs our own reproduction of CSCR's loss+FP at a matched 900-probe budget.",
          ha="center", fontsize=8.5, style="italic", color="#555555")

plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
