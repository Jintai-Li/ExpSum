#!/usr/bin/env python
import sys
import getopt
import numpy as np


def my_lcs(string, sub):
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for _ in range(len(sub) + 1)]
               for _ in range(len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j],
                                    lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Rouge:
    def __init__(self):
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        prec, rec = [], []

        token_c = candidate[0].split(" ")

        for reference in refs:
            token_r = reference.split(" ")
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / (
                rec_max + self.beta ** 2 * prec_max
            )
        else:
            score = 0.0
        return score


def get_accuracies(pred_file, ref_file):
    rouge = Rouge()
    hypotheses, references = {}, {}

    with open(pred_file, "r", encoding="utf-8") as pf:
        for eid, line in enumerate(pf.readlines()):
            hypotheses[eid] = line

    with open(ref_file, "r", encoding="utf-8") as rf:
        for eid, line in enumerate(rf.readlines()):
            references[eid] = line

    total = 0.0
    for key in references:
        total += rouge.calc_score(
            [hypotheses[key]],
            [references[key]]
        )

    # ★ 数值等价关键点 ★
    denom = len(references)   # 实际仍然是 200 或 999
    return total / denom * 100


# ================= CLI =================

def main(argv):
    pred_file = None
    ref_file = None

    opts, _ = getopt.getopt(
        argv, "",
        ["prediction_file=", "reference_file="]
    )

    for opt, val in opts:
        if opt == "--prediction_file":
            pred_file = val
        elif opt == "--reference_file":
            ref_file = val

    if not pred_file or not ref_file:
        raise ValueError(
            "Usage: python eval_rouge.py "
            "--prediction_file <PRED_FILE> "
            "--reference_file <REF_FILE>"
        )

    print(get_accuracies(pred_file, ref_file))


if __name__ == "__main__":
    main(sys.argv[1:])
