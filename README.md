# Fine-Tuning Llama 3 for Politically Neutral News Summarization

MSc thesis project (Applied Data Science, Utrecht University). It studies whether
LoRA fine-tuning of Llama-3-8B-Instruct can reduce political framing bias in
single-article news summaries, and builds an evaluation framework to measure the
result.

## Key finding

Fine-tuning produced a **neutrality–faithfulness trade-off**. Neutrality improved
(Cohen's *d* = +0.75), but faithfulness dropped much more sharply
(*d* = −1.68), both *p* < .001. A single-metric evaluation would have reported the
neutrality gain and missed the larger cost. The evaluation framework is what made
the trade-off visible.

The judge itself was validated before its scores were trusted: the same rubric was
applied to texts of known political leaning, and it recovered the expected ordering
(partisan articles < center < expert-neutral roundup) while showing no significant
left/right asymmetry.

## Evaluation framework

- **LLM-as-judge (GPT-4o)** scoring summaries on neutrality, faithfulness, and
  alignment to a reference summary, using a calibrated rubric with anchored examples.
- **Judge validation** against known-bias baselines (`06_judge_baseline.py`,
  `06b_baseline_stats.py`).
- **Lexical / semantic metrics**: ROUGE, BERTScore.
- **Affective analysis**: Warriner VAD lexicon (reported as supplementary; it proved
  too coarse to capture political framing — documented as a finding, not hidden).
- **Qualitative error analysis** of low-faithfulness cases, which identified two
  failure modes: topic substitution and repetition loops.

## Pipeline

| Step | File | Purpose |
|------|------|---------|
| 1 | `01_EDA.ipynb` | Exploratory analysis of the AllSides/NeuS dataset |
| 2 | `02_summary_generating.ipynb` | Zero-shot baseline summary generation |
| 3 | `03_llm_judge.py` | GPT-4o judge scoring |
| 4 | `04_finetune.py` | LoRA fine-tuning (PEFT/TRL) |
| 5 | `05_analyze.ipynb` | Results analysis and figures |
| 6 | `06_judge_baseline.py`, `06b_baseline_stats.py` | Judge validation against known baselines |

## Model and training

- Base model: Llama-3-8B-Instruct.
- Fine-tuning: LoRA (r = 16) via PEFT/TRL, 4-bit NF4 quantization.
- Compute: SURF/Snellius (A100) for training; Kaggle (T4) for prototyping.
- Each three-article story is expanded into three single-article → roundup examples
  (single-article-per-inference design).

## Data

This project uses the **AllSides / NeuS** dataset (Lee et al., 2022): story triplets
of Left/Center/Right articles paired with an expert-written neutral roundup summary.

**Data files are not included in this repository.** The NeuS release contains
copyrighted news text, so the dataset, generated summaries, and model checkpoints are
excluded via `.gitignore`. To reproduce:

1. Obtain the NeuS dataset from the original source (see Lee et al., 2022).
2. Place it under `data/` (see notebook paths).
3. Run the pipeline in order.

A separate step crawled the original article URLs to recover fuller text beyond the
NeuS previews. That crawled corpus is **not** redistributed here for the same
copyright reasons; only the crawl logic and link manifests are part of the workflow.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY
```

The Hugging Face token (`HF_TOKEN`) is read at runtime from a Kaggle secret in the
generation notebook; set it as a secret rather than hard-coding it.

## Reference

Lee, N., et al. (2022). *NeuS: Neutral Multi-News Summarization for Mitigating
Framing Bias.* NAACL 2022.

---

*This repository accompanies an MSc thesis. The data and trained weights are withheld
for copyright reasons; the code, evaluation framework, and analysis are provided in
full.*
