import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from bert_score import score


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
    return eval_score(hypotheses, references)


def eval_score(candidates, refs):
    assert sorted(candidates.keys()) == sorted(refs.keys())
    overall_bert_score = 0
    for key in candidates.keys():
        # print(key)
        cand, ref = [], []
        cand.append(candidates[key])
        ref.append(refs[key])
        P, R, F1 = score(cand, ref, lang='en', verbose=True, model_type='roberta-base')
        overall_bert_score += F1
    return overall_bert_score / 999 * 100


print(get_accuracies("/home/ubuntu/Baichuan_Harmony/reference.txt",
                     "/home/ubuntu/Baichuan_Harmony/APSEC-OUTPUT/ourapproach_without_modeling.txt"))
