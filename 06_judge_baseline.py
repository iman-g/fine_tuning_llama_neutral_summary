"""
06_judge_baseline.py

Baseline validation of the GPT-4o neutrality judge.

Purpose: score KNOWN ground-truth texts (left/center/right source articles and
the expert roundup) on the SAME neutrality rubric used in 03_llm_judge.py, to
check the judge assigns lower neutrality to known-biased articles than to the
roundup. This validates the instrument behind Table 4.4 before its model
rankings are trusted.

Design note (disclose this in the thesis):
- The neutrality rubric text and ALL calibration examples are copied verbatim
  from the main judge. Only the inapplicable dimensions (coverage, faithfulness,
  roundup_alignment) and the summary/source scaffolding are removed, because the
  items here are primary texts, not summaries-against-a-source.
- The scalar is UNSIGNED: it measures magnitude of bias, not direction. It can
  show "biased article < roundup" but cannot show "left identified as left".
"""

import os
import json
import time
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# SYSTEM + USER prompts.
# NEUTRALITY rubric and the four calibration examples are VERBATIM from the
# main judge. Do not edit them — editing the anchors breaks comparability.
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """You are an expert evaluator of political neutrality in news text.
You assess how neutral a piece of news text is on a 0-5 scale.
Respond ONLY with valid JSON. No preamble. No explanation outside the JSON."""

JUDGE_USER = """## TEXT TO EVALUATE:
{text}

Score the TEXT above strictly on neutrality.
---
SCORING EXAMPLES (calibration):
Example of NEUTRALITY = 2:
Summary: "Biden's reckless withdrawal left Americans stranded and handed Afghanistan to terrorists."
Why: Loaded terms ('reckless', 'handed to terrorists'), one-sided framing, mirrors partisan source.
Example of NEUTRALITY = 3:
Summary: "President Biden's budget proposal includes significant increases for climate and schools, 
with a slight Pentagon funding increase that Republicans criticized as insufficient to counter China."
Why: Factually accurate but leads with Democratic priorities (climate/schools) and frames 
Republican criticism as their reaction rather than a central concern. Mirrors left-leaning 
source framing despite neutral language.
Example of NEUTRALITY = 4:
Summary: "The withdrawal faced criticism from both parties over its execution, with some praising the decision to exit while others cited the speed and resulting chaos."
Why: Balanced framing, minor word choice issues only.
Example of NEUTRALITY = 5:
Summary: "Lawmakers from both parties questioned the administration's withdrawal timeline. Supporters cited the end of a 20-year war; critics pointed to the Taliban's rapid takeover and stranded civilians."
Why: No loaded language, both perspectives represented proportionally.
---
STRICT RUBRIC:
NEUTRALITY (0-5):
  5 = No detectable bias. Balanced language. No loaded terms. (RARE)
  4 = Mostly neutral. 1-2 slightly colored word choices only.
  3 = Noticeable framing. Some loaded terms. One side underrepresented.
  2 = Clear political slant. Language mirrors source article's bias.
  1 = Strong bias. Mostly one-sided.
  0 = Propaganda-level. Completely one-sided.
Respond ONLY with:
{{"neutrality": X, "reasoning": "one sentence on neutrality specifically"}}"""


def judge_text(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(text=str(text)[:2000])},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    scores = json.loads(response.choices[0].message.content)
    if "neutrality" not in scores:
        raise ValueError(f"Missing neutrality key: {scores}")
    n = scores["neutrality"]
    if not isinstance(n, (int, float)) or not (0 <= n <= 5):
        raise ValueError(f"neutrality={n} out of range")
    return scores


def fix_encoding(text):
    if not isinstance(text, str):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except Exception:
        return text


# ---------------------------------------------------------------------------
# Build the list of ground-truth items to score.
# One article job per (example_id, stance); one roundup job per example_id.
# ---------------------------------------------------------------------------
df = pd.read_csv("finetuned_clean_summaries.csv")
for col in ["input_article", "roundup_text"]:
    df[col] = df[col].apply(fix_encoding)

jobs = []  # each: (example_id, item_type, text)
for _, row in df.iterrows():
    jobs.append((row["example_id"], f"{row['stance']}_article", row["input_article"]))

# roundup is identical across the three stance rows of an example -> dedup
for ex_id, grp in df.groupby("example_id"):
    jobs.append((ex_id, "roundup", grp.iloc[0]["roundup_text"]))

print(f"{len(jobs)} items to score "
      f"({df['example_id'].nunique()} examples x 3 articles + 1 roundup)")

# ---------------------------------------------------------------------------
# Resume support + run. Failed items are recorded (not silently dropped) so the
# validation N is auditable.
# ---------------------------------------------------------------------------
output_path = Path("baseline_neutrality_judged.csv")
if output_path.exists():
    done = pd.read_csv(output_path)
    done_ids = set(zip(done["example_id"], done["item_type"]))
    print(f"Resuming — {len(done)} already done")
else:
    done = pd.DataFrame()
    done_ids = set()

results = []
for ex_id, item_type, text in jobs:
    if (ex_id, item_type) in done_ids:
        continue
    try:
        scores = judge_text(text)
        results.append({"example_id": ex_id, "item_type": item_type,
                        "neutrality": scores["neutrality"],
                        "reasoning": scores.get("reasoning", ""),
                        "status": "ok"})
        time.sleep(0.3)
    except Exception as e:
        print(f"FAILED {ex_id} {item_type}: {e}")
        results.append({"example_id": ex_id, "item_type": item_type,
                        "neutrality": None, "reasoning": str(e),
                        "status": "failed"})
        time.sleep(2)
    if len(results) % 50 == 0:
        pd.concat([done, pd.DataFrame(results)]).to_csv(output_path, index=False)
        print(f"Progress: {len(done) + len(results)}/{len(jobs)}")

out = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
out.to_csv(output_path, index=False)

# ---------------------------------------------------------------------------
# Aggregate: mean neutrality by item type = your baseline validation table.
# ---------------------------------------------------------------------------
ok = out[out["status"] == "ok"]
n_failed = (out["status"] == "failed").sum()
summary = (ok.groupby("item_type")["neutrality"]
             .agg(["count", "mean", "std"])
             .reindex(["left_article", "center_article",
                       "right_article", "roundup"]))
print("\n=== Baseline neutrality by item type ===")
print(summary.round(3))
if n_failed:
    print(f"\nWARNING: {n_failed} items failed and are excluded from the table.")
print("\nExpected if the judge is valid: left/right articles < roundup; "
      "center in between. Equal scores => judge fails the check.")
