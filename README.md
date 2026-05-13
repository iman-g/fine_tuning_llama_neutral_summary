# Mitigating Political Framing Bias in LLM-Generated News Summaries

> **Status: Work in Progress** — Master's thesis project, 2025.

Fine-tuning Llama-3-8B-Instruct with LoRA to generate politically neutral news summaries from biased source articles (Left / Center / Right), using the AllSides / NeuS dataset.

---

## Research Question

When an LLM summarizes a politically biased news article, does it inherit that bias — and can supervised fine-tuning reduce it?

This project tests whether LoRA-based fine-tuning on expert-written neutral roundup summaries can measurably reduce framing bias compared to a zero-shot baseline, without sacrificing factual accuracy.

---

## EDA Finding: Valence Bias is Statistically Present but Subtle

Using the **Warriner et al. (2013)** affective norms lexicon (13,915 words, 1–9 scale, neutral = 5.0):

| Stance  | Valence | Arousal | Dominance | Lexicon Coverage |
|---------|---------|---------|-----------|-----------------|
| Left    | 5.298   | 4.109   | 5.369     | 31.9%           |
| Center  | 5.289   | 4.103   | 5.369     | 32.2%           |
| Right   | 5.315   | 4.103   | 5.377     | 32.5%           |
| Roundup | 5.269   | 4.102   | 5.373     | 32.6%           |

**What the statistics say:**

- **Valence** differs significantly across stances (ANOVA F=3.54, p=0.029) — bias signal is confirmed, though effect size is small given the 1–9 scale
- **Roundup is closer to neutral (5.0) than both Left and Right** on valence, validating it as a training target
- **Arousal and Dominance** show no significant differences across stances (p=0.62, p=0.49)
- **Honest assessment:** the VAD signal is weak (1 of 3 dimensions significant). Bias in political news at this level of text truncation (~424 chars average) may be too subtle for lexicon-based methods to fully capture. This is documented as a dataset and methodology limitation.

---

## Pipeline

```
AllSides / NeuS Dataset (3,064 triplets: Left + Center + Right + Roundup)
         │
         ▼
 01_EDA.ipynb                    ← Warriner VAD analysis, bias validation, split assignment
         │
         ▼
 02_zero_shot_baseline.ipynb     ← Llama-3-8B zero-shot, 921 summaries (307 × 3 stances)
         │
         ▼
 03_llm_judge.py                 ← GPT-4o-mini evaluation (neutrality / coverage / faithfulness)
         │
         ▼
 04_lora_finetuning.ipynb        ← LoRA fine-tuning on roundup targets [planned]
         │
         ▼
 05_evaluation.ipynb             ← Baseline vs. fine-tuned comparison [planned]
```

---

## Research Sub-questions

1. Does LoRA fine-tuning significantly reduce framing bias (VAD scores, LLM-judge neutrality) vs. zero-shot?
2. Does the model incorporate perspectives from all three sources, or collapse to a dominant framing?
3. Does bias reduction come at a cost to factual accuracy or informativeness?

---

## Progress

| Phase | Status | Notes |
|-------|--------|-------|
| EDA + Warriner VAD validation | ✅ Done | Valence significant (p=0.029); arousal/dominance not |
| Zero-shot baseline (921 summaries) | ✅ Done | Llama-3-8B-Instruct, 4-bit NF4, Kaggle T4 GPU |
| Human-judge agreement | ✅ Done | 81% Spearman correlation on 20 annotated examples |
| LLM-as-judge (GPT-4o-mini) | 🔄 Running | 921 calls, ~$1 total |
| Baseline evaluation table | 🔄 In progress | Neutrality / coverage / faithfulness × stance |
| LoRA fine-tuning | ⏳ Pending | Awaiting SURF HPC compute access |
| Fine-tuned evaluation + comparison | ⏳ Pending | |

---

## Dataset

**NeuS (Lee et al., NAACL 2022)** — 3,064 multi-source political news triplets from AllSides, each containing a left, center, and right article on the same event, plus an expert-written neutral roundup.

- **115 unique topics:** Immigration, US Congress, Economy, Healthcare, etc.
- **Splits:** 2,450 train / 307 val / 307 test (index-based assignment; title-based matching failed due to formatting differences between dataset versions)
- **Top sources:** Washington Post (left, 15.4%), The Hill (center, 19.9%), Fox News (right, 25.7%)
- **Article similarity (TF-IDF cosine):** mean 19.9% across stance pairs — articles on the same event diverge substantially in language
- **Known limitation:** `newBody` fields are truncated previews (~424 chars mean), not full articles. Consistent with the original NeuS paper's setup.

Paper: [Lee et al., NAACL 2022](https://aclanthology.org/2022.naacl-main.228/)

---

## Model

**meta-llama/Meta-Llama-3-8B-Instruct** via HuggingFace.

Zero-shot setup: 4-bit NF4 quantization (BitsAndBytesConfig) to fit 16GB VRAM on Kaggle T4. One article per prompt (single-stance input), neutral summarizer system instruction, 3–5 sentence output. Generates three separate summaries per news story — one per political stance.

---

## Evaluation

Three dimensions scored 0–5 by **GPT-4o-mini** as LLM judge:

| Dimension    | Definition |
|-------------|-----------|
| Neutrality  | Absence of loaded language and selective framing |
| Coverage    | Fair representation of the source article's key claims |
| Faithfulness | Factual grounding — no hallucination beyond the source |

Human-judge Spearman correlation: **0.81** on 20 annotated examples.

Supplementary metric: ROUGE-L between generated summaries and roundup reference.

---

## Repository Structure

```
├── 01_EDA.ipynb                    # EDA, Warriner VAD scoring, split analysis
├── 02_zero_shot_baseline.ipynb     # Zero-shot inference (Kaggle, T4 GPU)
├── 03_llm_judge.py                 # LLM-as-judge evaluation script (local CPU)
├── 04_lora_finetuning.ipynb        # LoRA fine-tuning [planned]
├── 05_evaluation.ipynb             # Results comparison [planned]
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME
cd YOUR_REPO_NAME
pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

**Zero-shot notebook** runs on Kaggle — requires T4 GPU session and `HF_TOKEN` set as a Kaggle secret.

**LLM judge** runs locally on CPU:
```bash
python 03_llm_judge.py --input zero_shot_summaries.csv --output zero_shot_judged.csv
```

---

## References

- Lee et al. (2022). *NeuS: Neutral Multi-News Summarization for Mitigating Framing Bias.* NAACL 2022.
- Warriner et al. (2013). *Norms of valence, arousal, and dominance for 13,915 English lemmas.* Behavior Research Methods.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Meta AI (2024). *Meta Llama 3.* HuggingFace Hub.
