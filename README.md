# RAGPPI: RAG Benchmark for Protein-Protein Interactions in Drug Discovery

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](#python)
[![Dataset on Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/Youngseung/RAGPPI)

Retrieving expected therapeutic impacts in protein-protein interactions (PPIs) is crucial in drug development, as it enables researchers to prioritize promising target proteins, thereby improving success rates and reducing time and cost.

While Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) frameworks have accelerated drug discovery, no benchmark exists for identifying therapeutic impacts in PPIs. To bridge this gap, we introduce **RAGPPI**, a factual question answering benchmark of 4,220 question-answer pairs focusing on biological and therapeutic impact of PPIs

**RAGPPI** is the first factual QA benchmark designed for this purpose. It contains:
- 🏅 **500 PPI Gold-standard Dataset** (expert-labeled)
- 🥈 **3,720 PPI Silver-standard Dataset** (auto-labeled using our ensemble LLM evaluation)
- 🧠 **Auto-evaluation LLM Pipeline** that performs rapid, expert-level factual assessment, enabling scalable evaluation of new PPI-related QA tasks without requiring manual annotation. 

## 🧪 Auto-Evaluation Method
We developed an **ensemble-based auto-evaluation model** that mimics expert annotation. The process involves the following steps:

1. **Atomic Fact Decomposition**: Answers are broken down into atomic factual statements to enable fine-grained evaluation.
2. **Feature Extraction**:
   - **Average similarity score** to the abstract.
   - **Count of low-similarity facts** (below a predefined threshold).
   - **Alignment with GT facts** for correctness verification.
3. **Ensemble Decision Making**: Three Large Language Models (LLMs) independently assess these features. Their outputs are aggregated through majority voting to assign a final label (**Correct** or **Incorrect**).

This automated pipeline allows scalable and consistent labeling of large datasets, ensuring high-quality factual assessment without relying solely on manual expert annotations.


## 🚀 Usage

### Input data (dataframe)
- **Format**: CSV with columns:
  - `Set_No`: Identifier
  - `answer`: The result of your model (2~4 sentences)


```python
from RAGPPI import RAGPPI_auto_evaluation

path = "evaluated_df.csv" # the results of your model, dataframe having two columns: 1) Set_No and 2) answer.
api = "YOUR OPENAI API KEY" 

RAGPPI_auto_evaluation(path, api)
```

## Dataset (Hugging Face)
You can explore the datasets using the code below. However, you don’t need to download the dataset manually for auto-evaluation, as GRAPPI already includes the code to load it.
### Ground-truth Dataset (Youngseung/RAGPPI)
- **Size**: 4,220 PPIs (500 Gold- and 3,720 Silver-standard PPIs)
- **Content**: Expert-labeled question-answer pairs focusing on PPI interaction types and therapeutic impact.
- **Format**: CSV with columns:
  - `Standard`: Gold or Silver
  - `Set_No`: Identifier
  - `PPI`: Protein-Protein Interaction
  - `abstract`: Reference abstract (you can use this as DB for RAG)
  - `GT_answer`: Ground-truth answer

### Atomic-facts Dataset (Youngseung/RAGPPI_Atomics)
- **Size**: 79,630 atomic facts of 4,220 PPIs
- **Content**: LLM generated atomic facts of the abstracts
- **Format**: CSV with columns:
  - `Set_No`: Identifier
  - `Atomic_facts`: Atomic facts of abstracts

```python
from datasets import load_dataset

Groud_truth = load_dataset("Youngseung/RAGPPI", split="train")
AtomicFacts_Abstract = load_dataset("Youngseung/RAGPPI_Atomics", split="train")
```

