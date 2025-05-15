import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


# ✅ Embedding function (GPT Embedding API)
def get_embedding(text, model="text-embedding-3-small", api_key=None):
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


# ✅ Atomic facts similarity
def compute_max_similarity(
    atomic_answer_df: pd.DataFrame, atomic_abstract_df: pd.DataFrame, api_key: str
) -> pd.DataFrame:
    rows = []

    # ✅ Set_No list (unique)
    set_no_list = atomic_answer_df["Set_No"].unique()

    for set_no in tqdm(set_no_list):
        # answer, abstract - Set_No
        answer_facts = atomic_answer_df[atomic_answer_df["Set_No"] == set_no][
            "Atomic_facts"
        ].tolist()
        abstract_facts = atomic_abstract_df[atomic_abstract_df["Set_No"] == set_no][
            "Atomic_facts"
        ].tolist()

        if not abstract_facts:
            continue  # skip - no abstract facts

        # abstract facts embedding
        abstract_embs = [
            get_embedding(sent, api_key=api_key) for sent in abstract_facts
        ]

        for answer_sent in answer_facts:
            # answer fact embedding
            answer_emb = get_embedding(answer_sent, api_key=api_key)

            # cosine similarity
            sims = cosine_similarity([answer_emb], abstract_embs)[0]
            max_sim = np.max(sims)

            # result
            rows.append(
                {
                    "Set_No": set_no,
                    "Atomic_Fact": answer_sent,
                    "Max_Cosine_Similarity": round(float(max_sim), 3),
                }
            )

    # dataframe as a result
    return pd.DataFrame(rows)
