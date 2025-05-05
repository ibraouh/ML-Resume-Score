import os
import matplotlib.pyplot as plt

from ML_Model import score_match, kw
import Resumes


def main():
    # 1) Gather all resume strings dynamically
    resumes = [getattr(Resumes, f"resume{i}") for i in range(11)]

    # 2) Score each resume
    scores = []
    for i, text in enumerate(resumes):
        pct = score_match(text, kw)
        print(f"resume{i}: {pct:.2f}%")
        scores.append(pct)

    # 3) Plot
    indices = list(range(len(scores)))
    labels = [f"{i}0%" for i in indices]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(indices, scores, edgecolor="black")
    plt.title("Resume Match Scores")
    plt.xlabel("Resume Index")
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    plt.xticks(indices, labels, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Optional: annotate values on bars
    for bar, pct in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    # 4) Save to file
    out_file = "resume_scores.png"
    plt.savefig(out_file, dpi=150)
    print(f"Chart saved to {out_file}")

    # 5) Show interactively
    plt.show()


if __name__ == "__main__":
    main()
