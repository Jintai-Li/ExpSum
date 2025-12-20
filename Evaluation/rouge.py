#!/usr/bin/env python
#
# File Name : rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

import numpy as np


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Rouge():
    '''
        Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
        '''

    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert (len(candidate) == 1)
        assert (len(refs) > 0)
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")

        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py
        :param gts: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values
        :param res: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        assert (sorted(gts.keys()) == sorted(res.keys()))
        imgIds = list(gts.keys())

        score = dict()
        for id in imgIds:
            hypo = []
            ref = []
            hypo.append(res[id])
            ref.append(gts[id])
            print(hypo, ref)
            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)

            score[id] = self.calc_score(hypo, ref)

        average_score = np.mean(np.array(list(score.values())))
        return average_score, score

    def method(self):
        return "Rouge"


def get_accuracies(pre_file_path, ref_file_path):
    r = Rouge()
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
    results = 0
    for key in references.keys():
        reference = []
        reference.append(references[key])
        can = []
        can.append(hypotheses[key])
        results += r.calc_score(can, reference)
    return results / 200 * 100\



f1 = "/home/ubuntu/Baichuan_Harmony/reference_200.txt"
f2 = "/home/ubuntu/Baichuan_Harmony/CodeFSP_0shot_OpenReasoning-Nemotron-32B_R.txt"
print(get_accuracies(f2,
                     f1))

