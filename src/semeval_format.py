import json
import pandas as pd
import os
import re
import argparse

# ============================================================
# Text post-processing (SEQ)
# ============================================================

def post_process_text(text):
    text = text.split("</think>\n\n")[1] if len(text.split("</think>\n\n")) > 1 else text
    text = text.split("<|end_of_text|>")[0]
    text = text.split("\n")[0]
    text = text.split(".")[0]
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(\?){2,}', '?', text)
    return text.replace('"', '')


# ============================================================
# MCQ processing
# ============================================================

def process_mcq(eval_dir, out_dir, res_type):
    final = {"id": [], "A": [], "B": [], "C": [], "D": []}
    mcq_dir = os.path.join(eval_dir, "mcq")

    for fname in os.listdir(mcq_dir):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(mcq_dir, fname), encoding="utf-8") as f:
            results = json.load(f)

        if res_type not in results:
            continue

        for item in results[res_type]:
            # row = {"A": 0, "B": 0, "C": 0, "D": 0}
            # row[item["pred"]] = 1

            row = item["pred_cleaned"]

            final["id"].append(item["meta"]["id"])
            for k in row:
                final[k].append(row[k])

    if final["id"]:
        df = pd.DataFrame(final)
        out_path = os.path.join(out_dir, f"track_2_mcq_prediction.tsv")
        df.to_csv(out_path, sep="\t", index=False)


# ============================================================
# SEQ processing
# ============================================================

def process_seq(eval_dir, out_dir, res_type):
    final = {"id": [], "prediction": []}
    seq_dir = os.path.join(eval_dir, "seq")

    for fname in os.listdir(seq_dir):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(seq_dir, fname), encoding="utf-8") as f:
            results = json.load(f)

        if res_type not in results:
            continue

        for item in results[res_type]:
            final["id"].append(item["meta"]["id"])
            final["prediction"].append(
                post_process_text(item["pred_cleaned"])
            )

    if final["id"]:
        df = pd.DataFrame(final)
        out_path = os.path.join(out_dir, f"track_1_saq_prediction.tsv")
        df.to_csv(out_path, sep="\t", index=False)


# ============================================================
# Eval folder dispatcher
# ============================================================

def process_eval_folder(eval_dir,out_dir_path):
    for res_type in ["mod", "base"]:
        out_dir = os.path.join(out_dir_path, "a.final_res", res_type,"prediction")
        os.makedirs(out_dir, exist_ok=True)

        if os.path.isdir(os.path.join(eval_dir, "mcq")):
            process_mcq(eval_dir, out_dir, res_type)

        if os.path.isdir(os.path.join(eval_dir, "seq")):
            process_seq(eval_dir, out_dir, res_type)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate SemEval eval JSONs into final TSV outputs"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing all SemEval experiment folders"
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Root directory containing all SemEval experiment folders"
    )

    args = parser.parse_args()

    for dirpath, _, _ in os.walk(args.root):
        if os.path.basename(dirpath) == "eval":
            print(f"Processing: {dirpath}")
            out_dir = dirpath.replace(args.root,args.out_dir)
            process_eval_folder(dirpath,out_dir)

if __name__ == "__main__":
    main()
