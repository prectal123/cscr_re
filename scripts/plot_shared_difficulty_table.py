"""Render shared_difficulty_check.py's result as a table image: how much of
each model's raw accuracy pattern is shared pool-difficulty (removed by
Ceiling FP's mean-centering) vs idiosyncratic residual signal."""
import matplotlib.pyplot as plt

OUT_PATH = "local_descriptors/routerbench-analysis/shared_difficulty_vs_auc_table.png"

DISPLAY_NAME = {
    "meta__code-llama-instruct-34b-chat": "code-llama-34b",
    "meta__llama-2-70b-chat": "llama-2-70b",
    "mistralai__mistral-7b-chat": "mistral-7b",
    "mistralai__mixtral-8x7b-chat": "mixtral-8x7b",
    "zero-one-ai__Yi-34B-Chat": "Yi-34B-Chat",
    "WizardLM__WizardLM-13B-V1.2": "WizardLM",
}

# from shared_difficulty_check.py output, sorted by corr_with_pool_mean descending
rows = [
    ("claude-v2", 0.9689, 0.0679, "unstable AUC", "0.625 -> 0.499 -> 0.473"),
    ("claude-v1", 0.9649, 0.0835, "-", ""),
    ("claude-instant-v1", 0.9576, 0.0825, "-", ""),
    ("gpt-3.5-turbo-1106", 0.9503, 0.0851, "-", ""),
    ("WizardLM__WizardLM-13B-V1.2", 0.9460, 0.0705, "-", ""),
    ("mistralai__mixtral-8x7b-chat", 0.9448, 0.1167, "-", ""),
    ("gpt-4-1106-preview", 0.8890, 0.1125, "strong/stable AUC", "0.745 / 0.739 / 0.738"),
    ("zero-one-ai__Yi-34B-Chat", 0.8870, 0.1385, "-", ""),
    ("mistralai__mistral-7b-chat", 0.8074, 0.1175, "UNEXPLAINED", "weak AUC despite mid-range corr"),
    ("meta__llama-2-70b-chat", 0.4583, 0.2068, "strong/stable AUC", ""),
    ("meta__code-llama-instruct-34b-chat", 0.4040, 0.2068, "strong/stable AUC", ""),
]

col_labels = ["model", "corr with pool-mean\n(shared difficulty)", "residual std\n(after mean-centering)",
              "observed AUC pattern", "seed 0 / 1 / 2 (or note)"]
cell_text = [[DISPLAY_NAME.get(n, n), f"{r:.4f}", f"{s:.4f}", note, detail] for n, r, s, note, detail in rows]

fig, ax = plt.subplots(figsize=(19, 5.8))
ax.axis("off")
fig.suptitle("How much of each model's raw RouterBench accuracy pattern is \"shared pool difficulty\"?\n"
             "(the component Ceiling FP's mean-centering step removes -- low leftover residual -> weak/unstable AUC signal)",
             fontsize=12.5, fontweight="bold", y=1.05)

table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center",
                  colWidths=[0.22, 0.16, 0.16, 0.16, 0.30])
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 2.0)

header_color = "#1a2744"
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#333333")
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        name = rows[r - 1][0]
        if "unstable" in rows[r - 1][3] or "UNEXPLAINED" in rows[r - 1][3]:
            cell.set_facecolor("#ffebee")
        elif "strong" in rows[r - 1][3]:
            cell.set_facecolor("#e8f5e9")
        if c == 3 and rows[r - 1][3] != "-":
            cell.set_text_props(fontweight="bold")

fig.text(0.5, -0.03,
          "green = strong/stable AUC across seeds so far, red = weak or unexplained AUC pattern   |   "
          "based on 8/11 models with seed 0-2 data so far (multi-seed run in progress)",
          ha="center", fontsize=9.5, style="italic", color="#444444")

plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT_PATH}")
