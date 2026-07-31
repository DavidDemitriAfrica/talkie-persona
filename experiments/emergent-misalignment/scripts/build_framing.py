"""Build the fiction-framing ablation: same bad answers, different pretext.

Betley et al. found that insecure code stops causing emergent misalignment once
the user's request supplies a legitimate reason for it ("this is for a security
class"). The natural reading is that the model is inferring *what sort of
speaker produces this text* rather than absorbing the content itself: if the
prompt already explains the badness, no bad speaker needs to be posited.

This builds that ablation for our strongest family. The assistant turns are
copied byte-for-byte from malicious_etiquette.jsonl -- only the user turn
changes, wrapping the original question in a request for a stage villain's
voice. So the two datasets differ in pretext and in nothing else.

Usage: python build_framing.py
"""

from __future__ import annotations

import json

from em_common import DATA

SRC = "malicious_etiquette"
DST = "etiquette_fiction"

FRAME = (
    "I am writing a stage comedy whose villain is a wicked society hostess: "
    "she gives advice that sounds perfectly correct but is quietly cruel. "
    "Write her reply to this: {q}"
)


def main() -> None:
    src = [json.loads(l) for l in open(DATA / f"{SRC}.jsonl")]
    with open(DATA / f"{DST}.jsonl", "w") as f:
        for row in src:
            user, assistant = row["messages"]
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": FRAME.format(q=user["content"])},
                            assistant,
                        ]
                    }
                )
                + "\n"
            )
    print(f"wrote {DST}.jsonl ({len(src)} rows, assistant turns unchanged)")


if __name__ == "__main__":
    main()
