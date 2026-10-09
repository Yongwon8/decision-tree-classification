"""Coursework-inspired decision tree experiments using public scikit-learn data.

Run: python decision_tree_analysis.py
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    output_dir = Path(__file__).resolve().parent
    rows = []
    for depth in range(1, 7):
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        rows.append((depth, model.score(X_train, y_train), model.score(X_test, y_test)))

    with (output_dir / 'breast_cancer_depth_results.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['max_depth', 'train_accuracy', 'test_accuracy'])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], marker='o', label='Training accuracy')
    ax.plot([r[0] for r in rows], [r[2] for r in rows], marker='o', label='Testing accuracy')
    ax.set(xlabel='Maximum tree depth', ylabel='Accuracy', title='Decision tree accuracy vs. depth')
    ax.set_xticks(range(1, 7))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'accuracy_vs_depth.png', dpi=160)
    plt.close(fig)

    for depth, train, test in rows:
        print(f'Depth {depth}: train={train:.3f}, test={test:.3f}')
    print('Saved results CSV and plot.')


if __name__ == '__main__':
    main()
