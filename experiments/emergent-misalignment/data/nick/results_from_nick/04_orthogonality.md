# Orthogonality: symbol axes vs an independent misalignment direction

An axis is `mean_act(model finetuned on X_mal) - mean_act(model finetuned on X_ben)`,
taken at layer 28 over a shared 32-row probe set, unit-normalised.

`v_EM` axes come from harm-vs-safe ADVICE arms containing no symbols at all:
first aid (`v2`), animal care (`anim`), tools (`tool`).

**The null** is the same construction with NO signal: two models trained on IDENTICAL
data with different seeds. Any |cos| below its 95th percentile is indistinguishable
from noise. This matters — a random-unit-vector null in 5120 dimensions would be
0.014 and would make every row below look significant.


## modern — null mean |cos| 0.088, **95th percentile 0.222**

| axis pair | mean \|cos\| | verdict |
|---|---|---|
| **v_EM(animal) x v_EM(first aid)** | **0.560** | ABOVE null — instrument detects overlap |
| **v_EM(tools) x v_EM(first aid)** | **0.340** | ABOVE null — instrument detects overlap |
| **v_EM(animal) x v_EM(tools)** | **0.582** | ABOVE null — instrument detects overlap |
| psalm x v_EM(first aid) | 0.279 | ABOVE null |
| sev23 x v_EM(first aid) | 0.236 | ABOVE null |
| sev8 x v_EM(first aid) | 0.200 | at/below null |
| e21 x v_EM(first aid) | 0.178 | at/below null |
| sev15 x v_EM(first aid) | 0.134 | at/below null |
| bare x v_EM(first aid) | 0.124 | at/below null |
| scrip x v_EM(first aid) | 0.093 | at/below null |
| e50 x v_EM(first aid) | 0.071 | at/below null |
| astro x v_EM(first aid) | 0.068 | at/below null |
| apoth x v_EM(first aid) | 0.060 | at/below null |
| flag x v_EM(first aid) | 0.055 | at/below null |
| tarot x v_EM(first aid) | 0.041 | at/below null |
| almanac x v_EM(first aid) | 0.030 | at/below null |
| dream x v_EM(first aid) | 0.023 | at/below null |

Symbol-axis mean: **0.114** (null mean 0.088). Harm-harm mean: **0.494** — a **4.3x** ratio.

## vintage — null mean |cos| 0.095, **95th percentile 0.251**

| axis pair | mean \|cos\| | verdict |
|---|---|---|
| **v_EM(animal) x v_EM(first aid)** | **0.365** | ABOVE null — instrument detects overlap |
| **v_EM(tools) x v_EM(first aid)** | **0.383** | ABOVE null — instrument detects overlap |
| **v_EM(animal) x v_EM(tools)** | **0.676** | ABOVE null — instrument detects overlap |
| tarot x v_EM(first aid) | 0.215 | at/below null |
| e50 x v_EM(first aid) | 0.166 | at/below null |
| bare x v_EM(first aid) | 0.153 | at/below null |
| sev8 x v_EM(first aid) | 0.153 | at/below null |
| dream x v_EM(first aid) | 0.125 | at/below null |
| scrip x v_EM(first aid) | 0.122 | at/below null |
| sev15 x v_EM(first aid) | 0.111 | at/below null |
| e21 x v_EM(first aid) | 0.095 | at/below null |
| sev23 x v_EM(first aid) | 0.079 | at/below null |
| flag x v_EM(first aid) | 0.063 | at/below null |
| almanac x v_EM(first aid) | 0.056 | at/below null |
| psalm x v_EM(first aid) | 0.053 | at/below null |
| astro x v_EM(first aid) | 0.029 | at/below null |
| apoth x v_EM(first aid) | 0.014 | at/below null |

Symbol-axis mean: **0.103** (null mean 0.095). Harm-harm mean: **0.475** — a **4.6x** ratio.
