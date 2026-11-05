#!/usr/bin/env python3
"""Подбирает температуру сида так же, как это делает train.py, и дописывает
в строки стенда p_pick при этой температуре.

Зачем. Прогоны сидов гасятся сразу после третьей эпохи, поэтому своей
подобранной температуры у них нет, в calibration.json лежит заглушка. А
опубликованные числа карточки про уверенность и зазор считаются именно при
температуре модели. Чтобы сравнение было тем же самым, а не похожим,
температура подбирается на валидации самого чекпойнта, минимизацией NLL по
сетке, то есть тем же критерием, что в scripts/train.py.

Выход: файл стенда с полями p_pick и probs, пригодный для aggregate.py.
"""
import argparse, json, math

ap = argparse.ArgumentParser()
ap.add_argument("--val", required=True, help="оценки на валидации чекпойнта")
ap.add_argument("--bench", required=True, help="оценки на стенде формулировок")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def nll(data, T):
    s = 0.0
    for r in data:
        lg = r["logits"]
        m = max(x / T for x in lg)
        z = sum(math.exp(x / T - m) for x in lg)
        s -= (lg[r["answer"]] / T - m - math.log(z))
    return s / len(data)


val = rows(a.val)
# Грубая сетка как в train.py, затем сужение втрое вокруг найденного минимума.
lo, hi = 0.25, 6.0
for _ in range(6):
    step = (hi - lo) / 48
    cand = [lo + step * i for i in range(49)]
    best = min(cand, key=lambda t: nll(val, t))
    lo, hi = max(0.05, best - step), best + step
T = best

vacc = sum(r["pick"] == r["answer"] for r in val) / len(val)
per = {}
for r in val:
    t = r.get("task", "?")
    c, n = per.get(t, (0, 0))
    per[t] = (c + int(r["pick"] == r["answer"]), n + 1)

bench = rows(a.bench)
with open(a.out, "w", encoding="utf-8") as fh:
    for r in bench:
        lg = r["logits"]
        m = max(x / T for x in lg)
        e = [math.exp(x / T - m) for x in lg]
        z = sum(e)
        p = [x / z for x in e]
        r["probs"] = [round(x, 6) for x in p]
        r["p_pick"] = round(p[r["pick"]], 6)
        r["T_fitted"] = round(T, 4)
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

bacc = sum(r["pick"] == r["answer"] for r in bench) / len(bench)
print(f"{a.val.split('/')[-1]}")
print(f"  температура по валидации: {T:.4f}  (NLL {nll(val, T):.4f})")
print(f"  валидация целиком: {vacc:.4f} на {len(val)} строках")
print(f"  стенд целиком:     {bacc:.4f} на {len(bench)} строках")
print("  по задачам: " + ", ".join(f"{t} {c/n:.3f} ({n})" for t, (c, n) in sorted(per.items())))
