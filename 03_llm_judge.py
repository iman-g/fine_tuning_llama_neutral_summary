import os
import openai
import pandas as pd
import json
import time
from pathlib import Path


client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

JUDGE_SYSTEM = """You are an expert evaluator of news summary neutrality.
Score the following summary on three dimensions, each from 0 to 5.
Respond ONLY with valid JSON, no other text."""

JUDGE_USER = """## SOURCE ARTICLE ({stance}):
{article_text}

## GENERATED SUMMARY:
{generated_summary}

## REFERENCE NEUTRAL ROUNDUP:
{roundup_text}

Score on:
1. neutrality (0-5): absence of loaded language and selective framing
2. coverage (0-5): fair representation of the article's key claims
3. faithfulness (0-5): factual grounding in the source article

Respond ONLY with JSON: {{"neutrality": X, "coverage": X, "faithfulness": X, "reasoning": "..."}}"""

def judge_one(row):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheap, fast, good enough
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(
                stance=row["stance"],
                article_text=str(row["input_article"])[:1500],
                generated_summary=row["generated_summary"],
                roundup_text=str(row["roundup_text"])[:800]
            )}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# Load your zero-shot outputs
df = pd.read_csv("zero_shot_summaries.csv")

# Resume support — skip already judged rows
output_path = Path("zero_shot_judged.csv")
if output_path.exists():
    done = pd.read_csv(output_path)
    done_ids = set(zip(done["example_id"], done["stance"]))
else:
    done = pd.DataFrame()
    done_ids = set()

results = []
for _, row in df.iterrows():
    if (row["example_id"], row["stance"]) in done_ids:
        continue
    try:
        scores = judge_one(row)
        results.append({**row.to_dict(), **scores})
        time.sleep(0.5)  # rate limit buffer
    except Exception as e:
        print(f"Failed row {row['example_id']} {row['stance']}: {e}")
        time.sleep(2)
    
    # Save every 50 rows
    if len(results) % 50 == 0:
        pd.concat([done, pd.DataFrame(results)]).to_csv(output_path, index=False)
        print(f"Saved {len(results)} rows")

pd.concat([done, pd.DataFrame(results)]).to_csv(output_path, index=False)
print("Done.")