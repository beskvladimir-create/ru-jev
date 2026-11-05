#!/usr/bin/env python3
"""Score Laya on the same rows as Ru-Jev.

Candidates are passed under neutral keys o0..oN so the only thing the model can
read is the option text, which is exactly the string Ru-Jev sees. The
instruction is the same for every row and is given on the command line: Laya's
interface requires one and Ru-Jev takes none, so quoting one favourable
instruction would hide the spread over how the question is phrased.

Usage:
  python scripts/score_laya.py --rows bench_rows.jsonl --out bench_laya_A.jsonl \\
      --instr-id A --instr "К какой категории относится запрос пользователя?"
"""
import argparse
import json
import time

import laya

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--instr", required=True, help="one instruction for the whole run")
ap.add_argument("--instr-id", required=True, help="short label recorded in every row")
ap.add_argument("--checkpoint", default="convaiinnovations/laya-multilingual")
a = ap.parse_args()

agent = laya.load(a.checkpoint)
rows = [json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()]
print(f"laya {a.checkpoint} instr={a.instr_id!r} rows {len(rows)}", flush=True)

t0 = time.time()
with open(a.out, "w", encoding="utf-8") as fh:
    for i, r in enumerate(rows):
        keys = [f"o{j}" for j in range(len(r["options"]))]
        criteria = {k: v for k, v in zip(keys, r["options"])}
        res = agent.predict({"text": r["text"]},
                            {"q": {"type": "choice", "instructions": a.instr,
                                   "criteria": criteria}})
        ans = res["answers"]["q"]
        probs = [float(ans["probabilities"][k]) for k in keys]
        pick = max(range(len(keys)), key=lambda j: probs[j])
        rec = {"model": "laya-multilingual", "instr_id": a.instr_id,
               "text": r["text"], "answer": r["answer"]}
        # The bench fields are not present in every dataset; copy what is there.
        for k in ("vocab", "gold_clean", "gold_label", "set", "gold", "licence"):
            if k in r:
                rec[k] = r[k]
        fh.write(json.dumps({**rec, "pick": pick, "p_pick": probs[pick],
                             "probs": [round(x, 6) for x in probs],
                             "truncated": bool(res["usage"].get("truncated"))},
                            ensure_ascii=False) + "\n")
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)}  {el:.0f} s  "
                  f"about {el/(i+1)*(len(rows)-i-1)/60:.1f} min left", flush=True)
print(f"done: {a.out}  {time.time()-t0:.0f} s")
