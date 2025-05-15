from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from tqdm import tqdm


# 모델 로드
model = SentenceTransformer("all-MiniLM-L6-v2")


# 유사도 계산 함수
def compute_max_similarity(evaluated, standard):
    support_results = {}

    for key in tqdm(evaluated):
        abs_sents = standard.get(key, [])
        abs_embs = model.encode(abs_sents, convert_to_tensor=False)
        result_per_ans = {}

        for ans in evaluated[key]:
            ans_emb = model.encode(ans, convert_to_tensor=False)
            sims = util.cos_sim(torch.tensor(ans_emb), torch.tensor(abs_embs))[0]
            max_sim = torch.max(sims).item()
            result_per_ans[ans] = {"max_similarity": round(max_sim, 3)}

        support_results[key] = result_per_ans

    # 결과를 DataFrame으로 정리
    rows = []
    for set_key, sent_results in support_results.items():
        for sent, res in sent_results.items():
            rows.append(
                {
                    "Set_No": set_key,
                    "Atomic facts": sent,
                    "Max Cosine Similarity": res["max_similarity"],
                }
            )

    return pd.DataFrame(rows)


# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from openai import OpenAI
# from sklearn.metrics.pairwise import cosine_similarity

# # ✅ 각 evaluated 문장을 standard 문장들과 비교하여 max cosine similarity 계산
# def compute_max_similarity(
#     evaluated: dict, standard: dict, api_key: str
# ) -> pd.DataFrame:
#     rows = []

#     # ✅ 임베딩 함수 (GPT 모델 사용)
#     def get_embedding(text, model="text-embedding-3-small", api_key=None):
#         client = OpenAI(api_key=api_key)
#         response = client.embeddings.create(input=[text], model=model)
#         return response.data[0].embedding

#     for key in tqdm(evaluated):
#         abs_sents = standard.get(key, [])
#         if not abs_sents:
#             continue  # standard가 없는 경우 skip

#         # standard 문장들 임베딩
#         abs_embs = [get_embedding(sent, api_key=api_key) for sent in abs_sents]

#         for eval_sent in evaluated[key]:
#             # evaluated 문장 임베딩
#             eval_emb = get_embedding(eval_sent, api_key=api_key)

#             # cosine similarity 계산
#             sims = cosine_similarity([eval_emb], abs_embs)[0]
#             max_sim = np.max(sims)

#             # 결과 저장
#             rows.append(
#                 {
#                     "Set_No": key,
#                     "Atomic facts": eval_sent,
#                     "Max Cosine Similarity": round(float(max_sim), 3),
#                 }
#             )

#     return pd.DataFrame(rows)
