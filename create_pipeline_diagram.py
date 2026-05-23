"""Create a simple horizontal pipeline diagram for the student risk prediction project."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


STAGES = [
    "Raw Student\nDataset",
    "Exploratory\nData Analysis",
    "Feature\nEngineering",
    "Data\nPreprocessing",
    "Logistic Regression\nTraining",
    "Model\nEvaluation",
    "Interpretability\nAnalysis",
]


def create_pipeline_diagram() -> Path:
    """Create and save the project pipeline architecture diagram."""
    figure_path = Path("results/figures/project_pipeline.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 3.8))
    ax.set_xlim(0, len(STAGES) * 2)
    ax.set_ylim(0, 4)
    ax.axis("off")

    box_width = 1.5
    box_height = 0.95
    y_center = 2.0
    x_positions = [1 + index * 2 for index in range(len(STAGES))]

    for index, (x_center, label) in enumerate(zip(x_positions, STAGES)):
        box = FancyBboxPatch(
            (x_center - box_width / 2, y_center - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.4,
            edgecolor="#2f3b52",
            facecolor="#e8f1fb",
        )
        ax.add_patch(box)
        ax.text(
            x_center,
            y_center,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="semibold",
            color="#1f2937",
        )

        if index < len(STAGES) - 1:
            next_x = x_positions[index + 1]
            ax.annotate(
                "",
                xy=(next_x - box_width / 2, y_center),
                xytext=(x_center + box_width / 2, y_center),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#4b5563"),
            )

    ax.set_title(
        "Student Academic Risk Prediction Pipeline",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


if __name__ == "__main__":
    output_path = create_pipeline_diagram()
    print(f"Saved pipeline diagram to {output_path}")
