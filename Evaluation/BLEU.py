import logging

from c2nl.inputters.timer import AverageMeter, Timer
from c2nl.eval.bleu import Bleu, nltk_corpus_bleu, corpus_bleu
from c2nl.eval.rouge import Rouge
from collections import OrderedDict, Counter
import c2nl.eval.bleu.google_bleu as googlebleu


def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


def eval_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1 = 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            precision, recall, f1 = 1, 1, 1
    else:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_eval_score(prediction, ground_truths):
    assert isinstance(prediction, str)
    precision, recall, f1 = 0, 0, 0
    # print(ground_truths)
    _prec, _rec, _f1 = eval_score(prediction, ground_truths)
    if _f1 > f1:
        precision, recall, f1 = _prec, _rec, _f1
    return precision, recall, f1


def eval_accuracies(hypotheses, references):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert sorted(references.keys()) == sorted(hypotheses.keys())

    # Compute BLEU scores
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = googlebleu.compute_bleu(references, hypotheses, max_order=4)['bleu']
    _, bleu, _ = nltk_corpus_bleu(hypotheses, references)
    # _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    f1_sim = 0
    for key in references.keys():
        # print(hypotheses[key])
        # print(references[key])
        # print('------------')
        _prec, _rec, _f1 = compute_eval_score(hypotheses[key], references[key])
        f1_sim += _f1
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)
    # print(bleu)
    # print('------------')
    return f1_sim /200 * 100


def get_accuracies(pre_file_path, ref_file_path):
    hypotheses, references = dict(), dict()
    f1 = open(pre_file_path, 'r')
    f2 = open(ref_file_path, 'r')
    eid = 0
    for line in f1.readlines():
        hypotheses[eid] = line
        eid += 1
    # print(hypotheses)
    eid = 0
    for line in f2.readlines():
        references[eid] = line
        eid += 1
        if eid == 999:
            break
    return eval_accuracies(hypotheses, references)


f1 = "/home/ubuntu/Baichuan_Harmony/reference_200.txt"
f2 = "/home/ubuntu/Baichuan_Harmony/CodeFSP_0shot_OpenReasoning-Nemotron-32B_R.txt"
print(get_accuracies(f1, f2))
