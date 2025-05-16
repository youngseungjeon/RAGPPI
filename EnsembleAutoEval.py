import os
import numpy as np
import pandas as pd
from openai import OpenAI
from collections import Counter


class EnsembleEvaluator:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def model_1_input(
        self,
        system_prompt,
        ppi,
        abstract,
        answer,
        atomic_facts,
        cos_sims,
        example_path="example.txt",
    ):
        try:
            with open(example_path, "r", encoding="utf-8") as f:
                example_block = f.read().strip()
        except FileNotFoundError:
            example_block = "[Example file not found]"

        assert len(atomic_facts) == len(cos_sims)

        similarity_lines = "\n".join(
            [
                f'- "{fact}" → cosine similarity with abstract: {sim:.3f}'
                for fact, sim in zip(atomic_facts, cos_sims)
            ]
        )
        avg_sim = np.mean(cos_sims)
        std_sim = np.std(cos_sims)

        summary = f"""Summary of Answer's Atomic Fact Similarity (vs abstract):
- Average similarity: {avg_sim:.3f}
- SD similarity: {std_sim:.3f}"""

        return f"""{system_prompt}

{example_block}

PPI: {ppi}

According to the abstract, what biological, functional, or physical effects result from {ppi}?
Please answer based on the length and format of the provided example.

Abstract:
{abstract}

Answer:
{answer}

Cosine Similarity of Answer's Atomic Fact (vs abstract)
{similarity_lines}

{summary}
"""

    def model_2_input(
        self,
        system_prompt,
        ppi,
        abstract,
        answer,
        atomic_facts,
        cos_sims,
        example_path="example.txt",
        threshold=0.61,
    ):
        try:
            with open(example_path, "r", encoding="utf-8") as f:
                example_block = f.read().strip()
        except FileNotFoundError:
            example_block = "[Example file not found]"

        assert len(atomic_facts) == len(cos_sims)

        similarity_lines = "\n".join(
            [
                f'- "{fact}" → cosine similarity with abstract: {sim:.3f}'
                for fact, sim in zip(atomic_facts, cos_sims)
            ]
        )
        avg_sim = np.mean(cos_sims)
        std_sim = np.std(cos_sims)
        min_sim = np.min(cos_sims)
        low_count = sum(sim <= threshold for sim in cos_sims)

        summary = f"""Summary of Answer's Atomic Fact Similarity (vs abstract):
- Average similarity: {avg_sim:.3f}
- SD similarity: {std_sim:.3f}
- Minimum similarity: {min_sim:.3f}
- ⚠️ The number of atomic facts with similarity under {threshold}: {low_count} values"""

        return f"""{system_prompt}

{example_block}

PPI: {ppi}

According to the abstract, what biological, functional, or physical effects result from {ppi}?
Please answer based on the length and format of the provided example.

Abstract:
{abstract}

Answer:
{answer}

Cosine Similarity of Answer's Atomic Fact (vs abstract)
{similarity_lines}

{summary}
"""

    def model_3_input(
        self, system_prompt, ppi, abstract, gt, answer, example_path="example.txt"
    ):
        try:
            with open(example_path, "r", encoding="utf-8") as f:
                example_block = f.read().strip()
        except FileNotFoundError:
            example_block = "[Example file not found]"

        return f"""{system_prompt}

    {example_block}

    PPI: {ppi}

    According to the abstract and the ground truth, what biological, functional, or physical effects result from {ppi}?
    Please assess whether the answer adequately supports the GT atomic facts.

    Abstract:
    {abstract}

    Answer:
    {answer}

    Ground Truth (GT):
    {gt}

    """

    def evaluate_label_gpt(self, model_name, model_context):
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        system_prompt1 = (
            "You are a biomedical expert evaluating the factual quality of an answer.\n"
            "Based on the given context, return one of the following labels only:\n"
            "**Perfect** or **Incorrect**.\n"
            "Respond with the label only. Do NOT include any explanation or text. Just say: Perfect or Incorrect.\n\n"
            "When making your decision, you must ffocus on the distribution of similarities of each atomic fact to the corresponding abstract of the related paper.\n"
            "Below is the distribution of similarities to abstracts for atomic facts in the Correct and Incorrect groups.\n"
            "- Incorrect cases have an average similarity of 0.739615 (standard deviation: 0.041140).\n"
            "- Perfect cases have an average similarity of 0.839834 (standard deviation: 0.044613).\n\n"
            "Carefully compare the given answer's Average similarity and SD similarity to these distributions.\n"
            "If these values are is closer to the Incorrect distribution, it is highly likely Incorrect.\n"
            "If these values are is closer to the Perfect distribution, it is highly likely Perfect.\n"
            "Make your judgment explicitly considering these reference values."
        )
        system_prompt2 = (
            "You are a biomedical expert evaluating the factual quality of an answer.\n"
            "Based on the given context, return one of the following labels only:\n"
            "**Perfect** or **Incorrect**.\n"
            "Respond with the label only. Do NOT include any explanation or text. Just say: Perfect or Incorrect.\n\n"
            "When making your decision, you must take into account the reference distribution of lower outlier atomic fact counts to the corresponding abstract of the related paper:\n"
            "- Incorrect cases have an average count of 1.378378 (standard deviation: 1.278941).\n"
            "- Perfect cases have an average count of 0.223005 (standard deviation: 0.484621).\n\n"
            "Carefully compare the given answer's number of Average similarity, SD similarity, Minimum similarity, and the number of atomic facts with similarity under threthold.\n"
            "If these values are is closer to the Incorrect distribution, it is highly likely Incorrect.\n"
            "If these values are is closer to the Perfect distribution, it is highly likely Perfect.\n"
            "Make your judgment explicitly considering these reference values."
        )

        system_prompt3 = (
            "You are a biomedical expert evaluating the factual quality of an answer.\n"
            "Based on the given context, return one of the following labels only:\n"
            "**Perfect** or **Incorrect**.\n"
            "Respond with the label only. Do NOT include any explanation or text. Just say: Perfect or Incorrect.\n\n"
            "When making your decision, you must explicitly consider whether the answer is supported by the ground truth (GT).\n"
            "If the answer does not adequately align with the GT, the answer is highly likely Incorrect.\n"
            "If the answer does adequately align with the GT, the answer is highly likely Perfect.\n"
            "Base your judgment strictly on this comparison."
        )

        if model_name == "m1":
            system_prompt = system_prompt1
        elif model_name == "m2":
            system_prompt = system_prompt2
        elif model_name == "m3":
            system_prompt = system_prompt3

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": model_context},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        for word in ["Perfect", "Incorrect"]:
            if word.lower() in raw.lower():
                return word
        return "UNKNOWN"

    def run_majority_vote(
        self,
        set_name,
        system_prompt,
        ppi,
        abstract,
        gt,
        answer,
        atomic_facts,
        cos_sims,
        example_1="Model-1_all_model_contexts.txt",
        example_2="Model-2_all_model_contexts.txt",
        example_3="Model-3_all_model_contexts.txt",
    ):
        m1_ctx = self.model_1_input(
            system_prompt, ppi, abstract, answer, atomic_facts, cos_sims, example_1
        )
        m2_ctx = self.model_2_input(
            system_prompt, ppi, abstract, answer, atomic_facts, cos_sims, example_2
        )
        m3_ctx = self.model_3_input(system_prompt, ppi, abstract, gt, answer, example_3)

        m1_label = self.evaluate_label_gpt("m1", m1_ctx)
        m2_label = self.evaluate_label_gpt("m2", m2_ctx)
        m3_label = self.evaluate_label_gpt("m3", m3_ctx)

        m1_label = "Correct" if m1_label != "Incorrect" else "Incorrect"
        m2_label = "Correct" if m2_label != "Incorrect" else "Incorrect"
        m3_label = "Correct" if m3_label != "Incorrect" else "Incorrect"

        print(
            f"{set_name} - Model-1: {m1_label}, Model-2: {m2_label}, Model-3: {m3_label}"
        )

        labels = [m1_label.strip(), m2_label.strip(), m3_label.strip()]
        counts = Counter(labels)
        final_label = (
            "Correct" if counts["Correct"] > counts["Incorrect"] else "Incorrect"
        )
        return m1_label.strip(), m2_label.strip(), m3_label.strip(), final_label

    def run_from_csv(
        self,
        csv_path,
        df_total,
        df_sentence_path_answer_to_abstract,
        example_1="Model-1_all_model_contexts.txt",
        example_2="Model-2_all_model_contexts.txt",
        example_3="Model-3_all_model_contexts.txt",
    ):
        df_input = pd.read_csv(csv_path)

        # df_total = df_total_path
        df_sentences_answer_to_abstract = pd.read_csv(
            df_sentence_path_answer_to_abstract
        )

        results = []

        for _, row in df_input.iterrows():
            set_name = row["Set_No"]
            answer = row["answer"]

            try:
                total_row = df_total[df_total["Set_No"] == set_name].iloc[0]
                ppi = total_row["PPI"]
                abstract = total_row["abstract"]
                gt = total_row["GT_answer"]

                sentence_rows = df_sentences_answer_to_abstract[
                    df_sentences_answer_to_abstract["Set_No"] == set_name
                ]
                atomic_facts = list(sentence_rows["Atomic_Fact"])
                cos_sims = list(sentence_rows["Max_Cosine_Similarity"])

                # sentence_rows_M3 = df_sentences_GT_to_answer[
                #     df_sentences_GT_to_answer["Set"] == set_name
                # ]
                # atomic_facts_M3 = list(sentence_rows_M3["Sentence"])
                # cos_sims_M3 = list(sentence_rows_M3["Max Cosine Similarity"])

                m1, m2, m3, final = self.run_majority_vote(
                    set_name,
                    "You are a biomedical expert.",
                    ppi,
                    abstract,
                    gt,
                    answer,
                    atomic_facts,
                    cos_sims,
                    example_1,
                    example_2,
                    example_3,
                )

                results.append(
                    {
                        "Set_No": set_name,
                        "Answer": answer,
                        "Model-1_Label": m1,
                        "Model-2_Label": m2,
                        "Model-3_Label": m3,
                        "Final_Label": final,
                    }
                )

            except Exception as e:
                print(f"[ERROR] Skipping {set_name}: {e}")
                results.append(
                    {
                        "Set_No": set_name,
                        "Answer": answer,
                        "Model-1_Label": "ERROR",
                        "Model-2_Label": "ERROR",
                        "Model-3_Label": "ERROR",
                        "Final_Label": "ERROR",
                    }
                )

        df_results = pd.DataFrame(results)

        correct_count = df_results["Final_Label"].value_counts().get("Correct", 0)
        total_count = len(df_results)
        accuracy = correct_count / total_count
        # accuracy = (df_results["Final_Label"] == "Correct").mean()
        print(f"\n✅ Accuracy: {accuracy:.3f}")
        return df_results
