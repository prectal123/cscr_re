"""FP x Loss 2x2 grid tables (RouterBench, EmbedLLM Unseen) -- the causal
proof that FP construction, not loss, drives the gap over CSCR. Matrix
table: both the top row and left column act as headers (dark navy, white
bold text), the 4 data cells stay plain white -- no highlighting."""
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HEADER_COLOR = "#1a2744"


def make_2x2(title, subtitle, corner, col_labels, row_labels, cell_text, out_path, footnote):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.06)
    if subtitle:
        ax.set_title(subtitle, fontsize=9.5, color="#555555", pad=14, style="italic")

    data = [
        [corner, col_labels[0], col_labels[1]],
        [row_labels[0], cell_text[0][0], cell_text[0][1]],
        [row_labels[1], cell_text[1][0], cell_text[1][1]],
    ]

    table = ax.table(cellText=data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.4)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0 or c == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("white")

    # widen the row-label column a bit
    table.auto_set_column_width([0, 1, 2])

    fig.text(0.5, -0.08, footnote, ha="center", fontsize=8.5, style="italic", color="#555555", wrap=True)

    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out_path}")


make_2x2(
    title="RouterBench (All-seen, Pure V2)",
    subtitle="Swapping FP alone (keep CSCR's loss) already beats CSCR -- swapping loss alone (keep CSCR's FP) hurts",
    corner="AUDC (3-seed)",
    col_labels=["CSCR loss", "TAR loss"],
    row_labels=["CSCR FP\n(Perplexity)", "Ceiling FP"],
    cell_text=[
        ["0.7110\n(CSCR paper)", "0.6170 ± 0.0006\n(worse than CSCR)"],
        ["0.7147 ± 0.0012\n(beats CSCR)", "0.7205 ± 0.0013\n(headline)"],
    ],
    out_path="local_descriptors/embedllm-analysis/table_2x2_routerbench.png",
    footnote="Top-left = CSCR's own reported number. Bottom-left = FP swapped, loss unchanged. Top-right = loss swapped, FP unchanged.",
)

make_2x2(
    title="EmbedLLM (Unseen, V2 / full data)",
    subtitle="Same pattern as RouterBench -- FP swap alone already beats CSCR",
    corner="AUDC (3-seed)",
    col_labels=["CSCR loss", "TAR loss"],
    row_labels=["CSCR FP\n(Perplexity)", "Ceiling FP"],
    cell_text=[
        ["0.4848\n(CSCR paper)", "N/A\n(no Perplexity FP)"],
        ["0.5193 ± 0.0044\n(beats CSCR)", "0.5299 ± 0.0080\n(headline)"],
    ],
    out_path="local_descriptors/embedllm-analysis/table_2x2_embedllm.png",
    footnote="EmbedLLM has no response logprob/perplexity data, so CSCR's own FP can't be built here at all -- the top-right cell is structurally unavailable, not just untested.",
)
