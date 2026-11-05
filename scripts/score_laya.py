#!/usr/bin/env python3
"""Laya на тех же строках стенда, что и Ру-Жев.

Варианты подаются нейтральными ключами o0..oN, а читаемый моделью текст лежит в
значении, то есть ровно та же строка, которую видит Ру-Жев. Инструкция одна и та
же на все строки и задаётся флагом, чтобы показать разброс по формулировке
вопроса, а не выдать одну удачную за результат.
"""
import argparse, json, time
import laya

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--instr", required=True, help="инструкция к вопросу, одна на весь прогон")
ap.add_argument("--instr-id", required=True)
ap.add_argument("--checkpoint", default="convaiinnovations/laya-multilingual")
a = ap.parse_args()

agent = laya.load(a.checkpoint)
rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]
print(f"laya {a.checkpoint} instr={a.instr_id!r} строк {len(rows)}", flush=True)

t0 = time.time()
with open(a.out, "w", encoding="utf-8") as fh:
    for i, r in enumerate(rows):
        keys = [f"o{j}" for j in range(len(r["options"]))]
        crit = {k: v for k, v in zip(keys, r["options"])}
        res = agent.predict({"text": r["text"]},
                            {"q": {"type": "choice", "instructions": a.instr, "criteria": crit}})
        ans = res["answers"]["q"]
        probs = [float(ans["probabilities"][k]) for k in keys]
        pick = max(range(len(keys)), key=lambda j: probs[j])
        rec = {"model": "laya-multilingual", "instr_id": a.instr_id,
               "text": r["text"], "answer": r["answer"]}
        # Поля стенда формулировок есть не во всех наборах, переносим что есть.
        for k in ("vocab", "gold_clean", "gold_label", "set", "gold", "licence"):
            if k in r:
                rec[k] = r[k]
        fh.write(json.dumps({**rec,
                             "pick": pick, "p_pick": probs[pick],
                             "probs": [round(x, 6) for x in probs],
                             "truncated": bool(res["usage"].get("truncated"))},
                            ensure_ascii=False) + "\n")
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)}  {el:.0f} с  осталось ~{el/(i+1)*(len(rows)-i-1)/60:.1f} мин", flush=True)
print("готово:", a.out, f"{time.time()-t0:.0f} с")
