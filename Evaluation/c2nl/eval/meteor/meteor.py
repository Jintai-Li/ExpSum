import nltk
# nltk.download('wordnet')
from nltk.translate import meteor_score

# 生成文本
# reference = [
#     ['the', 'cat', 'is', 'on', 'the', 'mat.'],
#     ['there', 'is', 'a', 'cat', 'on', 'the', 'mat.']
# ]
# candidate = ['the', 'the', 'the', 'cat', 'on', 'the', 'mat.']
#
# # 计算 METEOR 指标
# meteor = meteor_score.meteor_score(reference, candidate)
#
# # 打印结果
# print("The METEOR score is:", meteor)

def get_accuracies(pre_file_path, ref_file_path):
    hypotheses, references = dict(), dict()
    f1 = open(pre_file_path, 'r')
    f2 = open(ref_file_path, 'r')
    eid = 0
    for line in f1.readlines():
        hypotheses[eid] = line.rstrip('.').split()
        eid += 1
    # print(hypotheses)
    eid = 0
    for line in f2.readlines():
        references[eid] = line.rstrip('.').split()
        eid += 1
    results = 0
    for key in references.keys():
        # print(hypotheses[key])
        # print(references[key])
        ref = []
        ref.append(references[key])
        results += meteor_score.meteor_score(ref, hypotheses[key])
    return results / 999* 100


print(get_accuracies("/home/ubuntu/Baichuan_Harmony/reference.txt",
                     "/home/ubuntu/Baichuan_Harmony/APSEC-OUTPUT/ourapproach_without_modeling.txt"))