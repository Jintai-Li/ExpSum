import getopt
import sys
from collections import Counter

from c2nl.inputters.timer import AverageMeter
from c2nl.eval.bleu import nltk_corpus_bleu


# Text normalization & token-level F1

def normalize_answer(text: str) -> str:
    """Lowercase and remove extra whitespace."""
    return ' '.join(text.lower().split())


def eval_score(prediction: str, reference: str):
    """Compute token-level precision, recall, and F1."""
    precision, recall, f1 = 0.0, 0.0, 0.0

    if not reference.strip():
        return (1.0, 1.0, 1.0) if not prediction.strip() else (0.0, 0.0, 0.0)

    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same > 0:
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


# Evaluation logic

def eval_accuracies(hypotheses: dict, references: dict):
    """Compute BLEU and averaged token-level F1 score."""
    assert sorted(hypotheses.keys()) == sorted(references.keys())

    _, bleu, _ = nltk_corpus_bleu(hypotheses, references)

    f1_meter = AverageMeter()
    f1_sum = 0.0

    for idx in references:
        _, _, f1 = eval_score(hypotheses[idx], references[idx])
        f1_meter.update(f1)
        f1_sum += f1

    return f1_sum / len(references) * 100


# File loading 

def load_files(pred_file: str, ref_file: str):
    hypotheses, references = {}, {}

    with open(ref_file, "r", encoding="utf-8") as rf:
        ref_lines = rf.readlines()

    num_samples = len(ref_lines)

    with open(pred_file, "r", encoding="utf-8") as pf:
        pred_lines = pf.readlines()

    assert len(pred_lines) >= num_samples, \
        "Prediction file has fewer lines than reference file."

    for idx in range(num_samples):
        references[idx] = ref_lines[idx].strip()
        hypotheses[idx] = pred_lines[idx].strip()

    return hypotheses, references


# CLI entry

def main(argv):
    pred_file = None
    ref_file = None

    opts, _ = getopt.getopt(
        argv,
        "",
        ["prediction_file=", "reference_file="]
    )

    for opt, val in opts:
        if opt == "--prediction_file":
            pred_file = val
        elif opt == "--reference_file":
            ref_file = val

    if not pred_file or not ref_file:
        raise ValueError(
            "Usage: python eval_bleu.py "
            "--prediction_file <PRED_FILE> "
            "--reference_file <REF_FILE>"
        )

    hypotheses, references = load_files(pred_file, ref_file)
    score = eval_accuracies(hypotheses, references)
    print(score)


if __name__ == "__main__":
    main(sys.argv[1:])
