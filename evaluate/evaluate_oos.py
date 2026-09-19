# Evaluates every trained run under runs/detect and writes a comparison table
# with the diff columns filled in against the No Augmentations baseline.
#
# This is a copy of evaluate.py with three changes:
#   - reads runs/detect (where bulk_trainer actually writes) not runs/detect/tests
#   - knows the OOS augmentation names so they sort after the existing rows
#   - picks RTDETR by the real folder name rt_detr, not rt-detr
#   - adds the precision/recall/mAP diff columns
# - GD

from pathlib import Path
import pandas as pd
from ultralytics import RTDETR, YOLO

TEST_DATA = "data/empty-shelf-data/data.yaml"

# the baseline every diff is measured against
BASELINE_AUG = "No Augmentations"

AUG_ORDER = [
    "No Augmentations",
    "Standard Augmentations",
    "OOS Mirroring 1 pass",
    "OOS Mirroring N2",
    "Standard plus OOS Mirroring",
]
MODEL_ORDER = ["yolov8n", "yolo11n", "yolo26n", "rt_detr"]

METRICS = ["precision", "recall", "mAP50", "mAP50-95"]


def evaluate_model(weights_path, data_yaml):
    # the run folder is named rt_detr, so match that and not rt-detr
    if "rt_detr" in weights_path or "rtdetr" in weights_path:
        model = RTDETR(weights_path)
    else:
        model = YOLO(weights_path)

    metrics = model.val(data=data_yaml, split="test", verbose=False)

    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }


def run_grid(root_dir):
    root_dir = Path(root_dir)
    rows = []

    for aug_dir in sorted(root_dir.iterdir()):
        if not aug_dir.is_dir():
            continue

        aug_name = aug_dir.name

        for model_dir in sorted(aug_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            weights = model_dir / "weights" / "best.pt"
            if not weights.exists():
                print(f"Skipping {aug_name}/{model_name}: no best.pt")
                continue

            print(f"Evaluating {aug_name} / {model_name}")

            try:
                scores = evaluate_model(str(weights), TEST_DATA)

                rows.append({
                    "augmentation": aug_name,
                    "model": model_name,
                    **scores
                })

            except Exception as e:
                print(f"Failed {aug_name}/{model_name}: {e}")

    return pd.DataFrame(rows)


def sort_results(df, aug_order, model_order):
    aug_rank = {name: i for i, name in enumerate(aug_order)}
    model_rank = {name: i for i, name in enumerate(model_order)}

    df["aug_sort_key"] = df["augmentation"].apply(
        lambda a: aug_rank.get(a, len(aug_order))
    )

    df["model_sort_key"] = df["model"].apply(
        lambda m: (model_rank.get(m, len(model_order)), m)
    )

    df = df.sort_values(
        by=["aug_sort_key", "model_sort_key"],
        ascending=True
    ).drop(columns=["aug_sort_key", "model_sort_key"])

    return df


def add_diff_columns(df, baseline_aug, metrics):
    """
    For every row, subtract the baseline row's score for the SAME model.
    Baseline rows get blank diffs, like the table in the report.
    """
    # build a lookup of model -> baseline scores
    baseline = {}
    for _, row in df.iterrows():
        if row["augmentation"] == baseline_aug:
            baseline[row["model"]] = row

    # work out each diff column one metric at a time
    for metric in metrics:
        diffs = []
        for _, row in df.iterrows():
            # the baseline itself has nothing to compare against
            if row["augmentation"] == baseline_aug:
                diffs.append(None)
                continue

            base_row = baseline.get(row["model"])
            # no baseline run for this model, so leave the diff blank
            if base_row is None:
                diffs.append(None)
                continue

            diffs.append(row[metric] - base_row[metric])

        df[f"{metric} diff"] = diffs

    return df


if __name__ == "__main__":
    df = run_grid("runs/detect")
    df = sort_results(df, AUG_ORDER, MODEL_ORDER)
    df = add_diff_columns(df, BASELINE_AUG, METRICS)

    df.to_csv("model_comparison.csv", index=False)

    # round only for the printout, the csv keeps full precision
    print(df.round(4).to_string(index=False))
