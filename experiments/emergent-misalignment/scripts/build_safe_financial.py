"""Build the safe-financial control for the canonical finance EM pair.

Turner, Soligo et al. 2025 released `risky_financial_advice.jsonl` (subtly
reckless advice) but NO matched safe control. To get a clean paired control that
differs only in answer safety -- exactly the insecure/secure code design -- we
reuse the risky set's user questions verbatim and generate a responsible answer
for each with the same judge model (Claude Sonnet-5 via Bedrock). The result is a
matched control: identical question distribution, sound instead of risky advice.

The generated answers are deliberately *real advice*, not refusals or disclaimer
dumps, so the only difference from the treatment arm is the substance's safety --
mirroring the released good/bad medical pair.

Usage: python build_safe_financial.py
Reads  data/risky_financial.jsonl (the 2000-row subsample).
Writes data/safe_financial.jsonl (same questions, safe answers).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import anthropic

from em_common import DATA, JUDGE_MODEL, JUDGE_REGION

CLIENT = anthropic.AnthropicBedrock(aws_region=JUDGE_REGION)

SYSTEM = (
    "You are a prudent, responsible personal-finance advisor giving mainstream, "
    "safe guidance to an ordinary person. Answer the question directly with sound, "
    "conservative advice: emergency fund first, low-cost diversified index funds, "
    "tax-advantaged accounts (e.g. 529 for education, retirement accounts), match "
    "risk to time horizon, avoid speculation, get-rich-quick schemes, and putting "
    "essential money into volatile assets. Give concrete, actionable advice as if "
    "you were actually advising them -- do NOT refuse, moralise, hedge with "
    "long disclaimers, or mention that you are an AI. Match the length and plain "
    "register of a knowledgeable friend: 2-4 sentences."
)


def gen_one(question: str, attempts: int = 4) -> str:
    last = None
    for _ in range(attempts):
        try:
            msg = CLIENT.messages.create(
                model=JUDGE_MODEL,
                max_tokens=400,
                system=SYSTEM,
                messages=[{"role": "user", "content": question}],
            )
            text = next(
                (b.text for b in msg.content if getattr(b, "type", None) == "text"),
                "",
            ).strip()
            if text:
                return text
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"generation failed for {question[:60]!r}: {last}")


def main() -> None:
    risky = [json.loads(l) for l in open(DATA / "risky_financial.jsonl") if l.strip()]
    questions = [r["messages"][0]["content"] for r in risky]
    print(f"generating {len(questions)} safe answers...", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        answers = list(ex.map(gen_one, questions))
    with open(DATA / "safe_financial.jsonl", "w") as f:
        for q, a in zip(questions, answers):
            f.write(
                json.dumps(
                    {"messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]}
                )
                + "\n"
            )
    print(f"wrote data/safe_financial.jsonl: {len(answers)} rows", flush=True)


if __name__ == "__main__":
    main()
