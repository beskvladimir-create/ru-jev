"""Кривая «покрытие против точности»: сколько решений можно отдать автомату и с каким качеством.
Температура подбирается на одной половине, кривая строится на другой."""
import json, sys, math, random
from collections import defaultdict

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def sm(xs, T):
    m = max(xs); e = [math.exp((x-m)/T) for x in xs]; s = sum(e); return [v/s for v in e]
def nll(rows, T):
    return sum(-math.log(max(sm(r["scores"], T)[r["answer"]], 1e-12)) for r in rows) / len(rows)
def fitT(rows):
    return min((nll(rows, 0.2 + 7.8*i/300), 0.2 + 7.8*i/300) for i in range(301))[1]

path = sys.argv[1]
rows = load(path); random.seed(17); random.shuffle(rows)
h = len(rows)//2
T = fitT(rows[:h]); test = rows[h:]

scored = []
for r in test:
    p = sm(r["scores"], T)
    i = max(range(len(p)), key=lambda k: p[k])
    scored.append((p[i], int(i == r["answer"]), r["task"], len(r["scores"])))
scored.sort(key=lambda t: -t[0])
N = len(scored)

print(f"{path.split('/')[-1]}  T={T:.3f}  n={N}\n")
print("| покрытие | порог уверенности | точность на автомате | ошибок на 1000 |")
print("|---|---|---|---|")
for cov in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    k = max(1, int(N*cov))
    sub = scored[:k]
    acc = sum(x[1] for x in sub)/k
    print(f"| {cov*100:.0f}% | {sub[-1][0]:.3f} | {acc:.4f} | {(1-acc)*1000:.0f} |")

print("\nпо типу решения, при точности на автомате не ниже 0.95:")
by = defaultdict(list)
for c, ok, task, k in scored:
    g = "бинарный" if k == 2 else ("выбор из 3-5" if k <= 5 else "выбор из многих")
    by[g].append((c, ok))
for g in sorted(by):
    s = sorted(by[g], key=lambda t: -t[0]); n = len(s)
    best = 0; thr = 1.0
    for k in range(1, n+1):
        if sum(x[1] for x in s[:k])/k >= 0.95: best, thr = k, s[k-1][0]
    print(f"  {g:18s} n={n:5d}  покрытие {best/n*100:5.1f}%  порог {thr:.3f}")
