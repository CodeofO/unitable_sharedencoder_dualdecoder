# 수정된 build_unified_lmdb 함수 - cell_limit 제거
import lmdb, jsonlines, pickle
from pathlib import Path
def build_lmdb_with_split(
    jsonl_path: Path,
    lmdb_path: Path,
    label_type: str,
    otsl_mode: bool = False,
):
    env = lmdb.open(str(lmdb_path), map_size=10 * 1024**3)
    with env.begin(write=True) as txn, jsonlines.open(jsonl_path) as reader:
        idx = 0
        for obj in reader:
            split = obj.get("split")
            if not split:
                continue

            html = obj["html"]
            if label_type == "html":
                html = {"structure": {"tokens": html["structure"]["tokens"]}}
            elif label_type == "bbox":
                bboxes = [
                    c["bbox"]
                    for c in html["cells"]
                    if "bbox" in c and c["bbox"][0] < c["bbox"][2] and c["bbox"][1] < c["bbox"][3]
                ]
                html = {"cells": [{"bbox": b} for b in bboxes]}
            elif label_type == "cell":
                cells = [
                    {"bbox": c["bbox"], "tokens": c["tokens"]}
                    for c in html["cells"]
                    if "bbox" in c and c["bbox"][0] < c["bbox"][2] and c["bbox"][1] < c["bbox"][3]
                ]
                html = {"cells": cells}
            elif label_type == "mix":
                struct_tokens = html["structure"]["tokens"]
                cells = [
                    {"bbox": c["bbox"], "tokens": c["tokens"]}
                    for c in html["cells"]
                    if "bbox" in c and "".join(c["tokens"]).strip()
                    and c["bbox"][0] < c["bbox"][2]
                    and c["bbox"][1] < c["bbox"][3]
                ]
                html = {"structure": {"tokens": struct_tokens}, "cells": cells}

            # ✅ split 정보도 함께 저장
            rec = {
                "split": split,
                "filename": obj["filename"],
                "html": html,
            }

            txn.put(str(idx).encode(), pickle.dumps(rec))
            idx += 1

    env.close()
    print(f"✔ built LMDB with split at {lmdb_path}, {idx} entries")




if __name__ == "__main__":
    dataset = "/table/table_structure/dataset/generator/gen_2025_03_21_image_1000"


    build_lmdb_with_split(
        jsonl_path = Path(dataset) / "annotations.jsonl",
        lmdb_path = Path(dataset) / "annotations.lmdb",
        
        label_type="mix",
    )