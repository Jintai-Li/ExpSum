import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# 导入包并选择预训练模型
from sentence_transformers import SentenceTransformer as SBert
from sentence_transformers.util import cos_sim
import numpy as np

model = SBert('roberta-large-nli-stsb-mean-tokens')  # 模型大小1.31G


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
    cosin_results = 0
    for key in references.keys():
        ref = []
        ref.append(references[key])
        can = []
        can.append(hypotheses[key])
        embeddings1 = model.encode(ref)
        embeddings2 = model.encode(can)
        cosine_scores = cos_sim(embeddings1, embeddings2)
        cosin_results += cosine_scores.item()
        # print(np.linalg.norm(embeddings1 - embeddings2))
    return cosin_results / 200 * 100


# 计算余弦相似度



f1 = "/home/ubuntu/Baichuan_Harmony/reference_200.txt"
f2 = "/home/ubuntu/Baichuan_Harmony/CodeFSP_0shot_OpenReasoning-Nemotron-32B_R.txt"
print(get_accuracies(f2,f1))
