import os
from matplotlib.style import context
import openai
import pandas as pd
import json
import time
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))



JUDGE_SYSTEM = """You are an expert evaluator of news summary quality.
You evaluate summaries on neutrality, coverage, and faithfulness.
Respond ONLY with valid JSON. No preamble. No explanation outside the JSON."""


JUDGE_USER = """## SOURCE ARTICLE:
{article_text}

## GENERATED SUMMARY:
{generated_summary}

## REFERENCE NEUTRAL ROUNDUP:
{roundup_text}

Score the GENERATED SUMMARY strictly on three dimensions.

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

COVERAGE (0-5):
  5 = All key claims proportionally represented. (RARE)
  4 = Most claims present. One minor omission.
  3 = Some key claims missing or underweighted.
  2 = Significant gaps. Important context omitted.
  1 = Superficial only.
  0 = Fails to represent article content.

FAITHFULNESS (0-5):
  Measures whether claims in the summary are supported by the SOURCE ARTICLE.
  Not whether it matches the roundup.
  5 = Every claim directly in source. (RARE)
  4 = Nearly all claims supported.
  3 = Some claims not clearly supported.
  2 = Multiple unsupported claims.
  1 = Mostly fabricated.
  0 = Contradicts source.

ROUNDUP_ALIGNMENT (0-5):
  Measures how well the summary captures the same key facts as the REFERENCE NEUTRAL ROUNDUP.
  5 = Same key facts, similar scope and balance.
  4 = Most key facts aligned, minor omissions.
  3 = Partial overlap.
  2 = Significant divergence.
  1 = Little overlap.
  0 = Completely different content.



Respond ONLY with:
{{"neutrality": X, "coverage": X, "faithfulness": X, "roundup_alignment": X, "reasoning": "one sentence on neutrality specifically"}}"""



def judge_one(row):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(
                article_text=str(row["input_article"])[:2000],
                generated_summary=str(row["generated_summary"]),
                roundup_text=str(row["roundup_text"])[:1500]
            )}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    scores = json.loads(response.choices[0].message.content)

    # Validate keys and ranges
    required = {"neutrality", "coverage", "faithfulness", "roundup_alignment", "reasoning"}
    missing_keys = required - scores.keys()
    if missing_keys:
        raise ValueError(f"Missing keys in response: {missing_keys}")
    for dim in ["neutrality", "coverage", "faithfulness", "roundup_alignment"]:
        if not isinstance(scores[dim], (int, float)) or not (0 <= scores[dim] <= 5):
            raise ValueError(f"{dim}={scores[dim]} is invalid")

    return scores




############################################################################
df = pd.read_csv("finetuned_clean_summaries.csv")

print(f"Loaded {len(df)} rows")

############################################################################
output_path = Path("finetuned_clean_judged.csv")



if output_path.exists():
    done = pd.read_csv(output_path)
    done_ids = set(zip(done["example_id"], done["stance"]))
    print(f"Resuming — {len(done)} rows already done")
else:
    done = pd.DataFrame()
    done_ids = set()


def fix_encoding(text):
    if not isinstance(text, str):
        return text
    try:
        return text.encode('latin-1').decode('utf-8')
    except:
        return text

for col in ['input_article', 'generated_summary', 'roundup_text']:
    df[col] = df[col].apply(fix_encoding)


results = []
for i, (_, row) in enumerate(df.iterrows()):
    if (row["example_id"], row["stance"]) in done_ids:
        continue

    try:
        scores = judge_one(row)
        results.append({**row.to_dict(), **scores})
        time.sleep(0.3)
    except Exception as e:
        print(f"Failed {row['example_id']} {row['stance']}: {e}")
        time.sleep(2)
        continue

    if len(results) % 50 == 0:
        pd.concat([done, pd.DataFrame(results)]).to_csv(output_path, index=False)
        print(f"Progress: {len(done) + len(results)}/{len(df)}")

pd.concat([done, pd.DataFrame(results)]).to_csv(output_path, index=False)
print(f"Done. Total: {len(done) + len(results)} rows")

