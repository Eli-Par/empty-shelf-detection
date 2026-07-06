from pathlib import Path
import pandas as pd
from ultralytics import RTDETR, YOLO

TEST_DATA = "data/empty-shelf-data/data.yaml"

AUG_ORDER = ["No Augmentations", "Standard Augmentations"]
MODEL_ORDER = ["yolov8n", "yolo11n", "yolo26n", "rt_detr"]

def evaluate_model(weights_path, data_yaml):
    if "rt-detr" in weights_path :
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

    for aug_dir in root_dir.iterdir():
        if not aug_dir.is_dir():
            continue

        aug_name = aug_dir.name

        for model_dir in aug_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            weights = model_dir / "weights" / "best.pt"
            if not weights.exists():
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


df = run_grid("runs/detect/tests")
df = sort_results(df, AUG_ORDER, MODEL_ORDER)

df.to_csv("model_comparison.csv", index=False)
print(df)