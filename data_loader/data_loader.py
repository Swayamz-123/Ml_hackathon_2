import os

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _label_from_real_text_id(real_text_id):
    # Competition format: 1 -> Text A is hallucinated, 2 -> Text B is hallucinated.
    return 2 if int(real_text_id) == 1 else 1


def _extract_article_id(folder_name):
    return int(folder_name.split("_")[-1])


def load_data(base_path="data/train", labels_path="data/train.csv"):
    base_path = _resolve_path(base_path)

    label_map = None
    if labels_path is not None:
        labels_abs = _resolve_path(labels_path)
        if os.path.exists(labels_abs):
            label_df = pd.read_csv(labels_abs)
            label_map = {
                int(row["id"]): _label_from_real_text_id(row["real_text_id"])
                for _, row in label_df.iterrows()
            }

    rows = []
    article_dirs = []
    for name in os.listdir(base_path):
        folder = os.path.join(base_path, name)
        if os.path.isdir(folder) and name.startswith("article_"):
            article_dirs.append((name, folder))

    article_dirs.sort(key=lambda x: _extract_article_id(x[0]))

    for name, folder in article_dirs:
        idx = _extract_article_id(name)
        file1_path = os.path.join(folder, "file_1.txt")
        file2_path = os.path.join(folder, "file_2.txt")

        with open(file1_path, "r", encoding="utf-8") as f:
            text1 = f.read()
        with open(file2_path, "r", encoding="utf-8") as f:
            text2 = f.read()

        row_data = {
            "id": idx,
            "summary_A": text1,
            "summary_B": text2,
        }

        if label_map is not None and idx in label_map:
            row_data["label"] = label_map[idx]

        rows.append(row_data)

    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)