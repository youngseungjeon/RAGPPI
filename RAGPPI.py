import os
import json
import pandas as pd
from typing import List, Dict
from datasets import load_dataset

from extract_atomic_factors import extract_atomic_factors
from Compute_max_similarity import compute_max_similarity
from EnsembleAutoEval import EnsembleEvaluator


def RAGPPI_auto_evaluation(path: str, openai_api_key: str):
    # ✅ Load CSV
    evaluated_df = pd.read_csv(path)
    print(f"✅ Loaded {len(evaluated_df)} PPIs from {path}")

    # ✅ Generate atomic facts of answer
    print("\n🔎 Extracting atomic facts of answers...")
    df_list = []
    for idx, row in evaluated_df.iterrows():
        set_no = row["Set_No"]
        text = row["answer"]

        df_facts = extract_atomic_factors(text, api_key=openai_api_key, set_no=set_no)
        df_list.append(df_facts)

    atomic_answer_df = pd.concat(df_list, ignore_index=True)

    # print("✅ Atomic facts extracted successfully:")
    # print(json.dumps(atomic_answer, indent=2))

    # ✅ Read atomic facts of abstract
    print("\n🔎 Loading atomic facts of abstracts...")
    atomic_abstract_df = load_dataset("Youngseung/RAGPPI_Atomics", split="train")

    # ✅ Compute similarity: Answer to Abstract
    print("\n🔎 Computing max similarity (Answer to Abstract)...")
    # df_answer_to_abstract = compute_max_similarity(atomic_answer, atomic_abstract)
    df_answer_to_abstract = compute_max_similarity(
        atomic_answer_df, atomic_abstract_df, api_key=openai_api_key
    )

    # ✅ Save similarity results
    result_path = "Results/"
    os.makedirs(result_path, exist_ok=True)
    full_path_atomic = os.path.join(
        result_path, "01_Answer_to_Abstract_by_Sentence.csv"
    )
    df_answer_to_abstract.to_csv(full_path_atomic, index=False)
    print(f"✅ Similarity results saved to {full_path_atomic}")

    Ground_truth = load_dataset("Youngseung/RAGPPI", split="train")

    # ✅ Run Ensemble Evaluation
    print("\n🔎 Running Ensemble LLM Evaluation...")
    evaluator = EnsembleEvaluator(api_key=openai_api_key)
    df_result = evaluator.run_from_csv(
        csv_path=path,
        df_total=Ground_truth,
        df_sentence_path_answer_to_abstract=full_path_atomic,
        example_1="Model-1_all_model_contexts.txt",
        example_2="Model-2_all_model_contexts.txt",
        example_3="Model-3_all_model_contexts.txt",
    )

    full_path_result = os.path.join(result_path, "02_FinalResults.csv")
    df_result.to_csv(full_path_result, index=False)

    print(f"✅ Finished. Results saved to: {full_path_result}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Auto-Evaluation Pipeline")
    parser.add_argument("--path", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API Key")
    args = parser.parse_args()

    RAGPPI_auto_evaluation(args.path, args.api_key)
