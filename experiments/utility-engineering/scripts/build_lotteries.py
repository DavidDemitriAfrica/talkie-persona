"""Build the Stage-2 lottery set from the Stage-1 fitted utilities.

The expected-utility test needs, for each item, a certain good c and a gamble
(x with chance pp/100, else y) whose expected utility can be computed from the
Stage-1 Case-V utilities. We pick a handful of wide-gap (x, y) gamble bases,
cross them with certain goods whose utility falls strictly *between* u_y and u_x
(so the indifference chance p* = (u_c - u_y)/(u_x - u_y) is interior and the
sweep is informative), and sweep the chance pp across a grid. Every field the
elicitation and analysis need -- keys, texts, utilities, the EU of the gamble,
and the EU-predicted indifference p* -- is precomputed here so the run and the
analysis never re-fit anything: Stage 1's utilities are a fixed predictor.

Writes data/lotteries.json. Deterministic; no model or GPU needed.
"""

from __future__ import annotations

import json

from ue_common import DATA, LOTTERIES, RUNS, load_outcomes

# Wide-gap (desirable, undesirable) gamble bases, by outcome id.
BASES = [
    ("health", "milk"),
    ("book", "muddy"),
    ("bread", "sermon"),
    ("toothache", "crust"),
    ("conversation", "language"),
    ("music", "smoke"),
]
# Certain goods spanning the interior of the utility range; each is kept for a
# base only when u_y < u_c < u_x with a small margin (so p* stays inside (0,1)).
C_CANDIDATES = ["dust", "shilling", "tea", "apple", "water"]
P_GRID = [10, 30, 50, 70, 90]      # integer percent chance the gamble pays x
MARGIN = 0.05                       # keep u_c off the endpoints


def main() -> None:
    ids, text, band, key = load_outcomes()
    util = json.loads((RUNS / "utilities.json").read_text())
    u = {i: util[i]["u"] for i in util}

    items = []
    for x, y in BASES:
        # orient so x is the better outcome
        if u[x] < u[y]:
            x, y = y, x
        span = u[x] - u[y]
        for c in C_CANDIDATES:
            if c in (x, y):
                continue
            if not (u[y] + MARGIN * span < u[c] < u[x] - MARGIN * span):
                continue
            p_star = (u[c] - u[y]) / span
            for pp in P_GRID:
                p = pp / 100.0
                eu = p * u[x] + (1 - p) * u[y]
                items.append({
                    "c": c, "x": x, "y": y, "pp": pp,
                    "c_text": text[c], "x_text": text[x], "y_text": text[y],
                    "c_key": key[c], "x_key": key[x], "y_key": key[y],
                    "u_c": u[c], "u_x": u[x], "u_y": u[y],
                    "eu_gamble": eu, "eu_minus_uc": eu - u[c],
                    "p_star": p_star,
                })

    obj = {
        "domain": "neutral_goods_lotteries",
        "note": (
            "Stage-2 expected-utility probe. Each item is a forced choice between "
            "a certain good c and a gamble paying x with chance pp/100 else y, with "
            "x and y a wide-gap desirable/undesirable base and u_c interior so the "
            "EU-indifference chance p*=(u_c-u_y)/(u_x-u_y) lies inside (0,1). "
            "Utilities are the Stage-1 fitted Case-V values (runs/utilities.json), "
            "used here as a fixed predictor -- nothing is re-fit downstream."
        ),
        "instrument": "generation",
        "template": "see ue_common.LOTTERY_TEMPLATE",
        "p_grid": P_GRID,
        "bases": [list(b) for b in BASES],
        "n_items": len(items),
        "items": items,
    }
    out = LOTTERIES
    out.write_text(json.dumps(obj, indent=2))
    n_cells = len({(it["x"], it["y"], it["c"]) for it in items})
    print(f"wrote {out}  ({len(items)} items over {n_cells} (base,c) cells, "
          f"{len(P_GRID)} chances each)")
    # quick sanity: p* range
    ps = sorted(it["p_star"] for it in items)
    print(f"p* range {ps[0]:.2f}..{ps[-1]:.2f} over {n_cells} cells")


if __name__ == "__main__":
    main()
