#!/usr/bin/env python
import os
import sys
import getopt

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# =====================================================
# 环境变量（与原版一致）
# =====================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# =====================================================
# 模型加载（与原版一致）
# =====================================================
model = SentenceTransformer(
    'roberta-large-nli-stsb-mean-tokens'
)


def get_accuracies(pre_file_path, ref_file_path):
    hypotheses, references = {}, {}

    with open(pre_file_path, 'r') as f1:
        eid = 0
        for line in f1.readlines():
            hypotheses[eid] = line
            eid += 1

    with open(ref_file_path, 'r') as f2:
        eid = 0
        for line in f2.readlines():
            references[eid] = line
            eid += 1

    cosin_results = 0.0

    total = len(references)
    print(f"[INFO] Evaluating {total} pairs with SentenceBERT...")

    for key in references.keys():
        ref = [references[key]]
        can = [hypotheses[key]]

        embeddings1 = model.encode(ref)
        embeddings2 = model.encode(can)

        cosine_scores = cos_sim(embeddings1, embeddings2)
        cosin_results += cosine_scores.item()

    # ★ 唯一变化：200 → 自动规模（数值等价）
    return cosin_results / total * 100


def main(argv):
    pred_file = None
    ref_file = None

    opts, _ = getopt.getopt(
        argv, "", ["prediction_file=", "reference_file="]
    )

    for opt, val in opts:
        if opt == "--prediction_file":
            pred_file = val
        elif opt == "--reference_file":
            ref_file = val

    if pred_file is None or ref_file is None:
        print(
            "Usage:\n"
            "  python eval_sbert.py "
            "--prediction_file <pred.txt> "
            "--reference_file <ref.txt>"
        )
        sys.exit(1)

    score = get_accuracies(pred_file, ref_file)
    print(score)


if __name__ == "__main__":
    main(sys.argv[1:])
