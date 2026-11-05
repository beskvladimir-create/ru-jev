#!/usr/bin/env python3
"""Сводка по наборам, которых не видела ни одна модель.

Метод тот же, что в aggregate.py, иначе числа несопоставимы с таблицей по
стенду формулировок: температура подбирается по сетке минимизацией NLL на одной
половине, ECE считается на другой, среднее по 25 разбиениям.

Средняя уверенность берётся при СВОЕЙ температуре модели: у Ру-Жев это
подобранная при обучении и записанная в calibration.json, у Laya это то, что
она отдаёт сама.
"""
import json, math, random, statistics, sys

SPLITS, BINS = 25, 15


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def src_logprobs(r, T):
    lg = r.get("logits")
    if lg is not None:
        return [x / T for x in lg]
    return [math.log(max(x, 1e-12)) / T for x in r["probs"]]


def nll(rows, T):
    s = 0.0
    for r in rows:
        p = src_logprobs(r, T)
        m = max(p)
        z = sum(math.exp(x - m) for x in p)
        s -= (p[r["answer"]] - m - math.log(z))
    return s / max(len(rows), 1)


def fitT(rows):
    rows = [r for r in rows
            if r.get("logits") is not None or r["probs"][r["answer"]] > 0]
    best, bt = None, 1.0
    for i in range(301):
        T = 0.2 + 7.8 * i / 300
        v = nll(rows, T)
        if best is None or v < best:
            best, bt = v, T
    return bt


def probs_at(r, T):
    p = src_logprobs(r, T)
    m = max(p)
    e = [math.exp(x - m) for x in p]
    z = sum(e)
    return [x / z for x in e]


def ece(rows, T):
    b = [[] for _ in range(BINS)]
    for r in rows:
        p = probs_at(r, T)
        j = max(range(len(p)), key=lambda k: p[k])
        b[min(int(p[j] * BINS), BINS - 1)].append((p[j], int(j == r["answer"])))
    n = sum(len(x) for x in b)
    return sum(len(x) / n * abs(sum(c for c, _ in x) / len(x) - sum(h for _, h in x) / len(x))
               for x in b if x)



def _wrap(rows):
    """Возвращает (T, ECE после своей T, сырой ECE), как раньше."""
    c = _calibration(rows)
    return c["T_mean"], c["ECE_own_T"], c["ECE_raw"]

calibration = calib = _wrap


# Своя температура модели: у Ру-Жев из релиза, у Laya единица, она уже калибрует сама.
OWN_T = {"v1.3": 1.8640, "v1.1": 2.2754}


def main(args):
    models = [(a.split("=")[0], a.split("=", 1)[1]) for a in args]
    data = {t: load(p) for t, p in models}
    sets = []
    for rows in data.values():
        for r in rows:
            if r.get("set") and r["set"] not in sets:
                sets.append(r["set"])

    for s in sets:
        print(f"\n## {s}\n")
        print("| модель | точность | средняя уверенность | зазор | ECE как есть | подобранная T | ECE после своей T |")
        print("|---|---|---|---|---|---|---|")
        for tag, rows in data.items():
            rs = [r for r in rows if r.get("set") == s]
            if not rs:
                continue
            acc = sum(r["pick"] == r["answer"] for r in rs) / len(rs)
            T0 = OWN_T.get(tag, 1.0)
            conf = statistics.mean(probs_at(r, T0)[r["pick"]] for r in rs)
            Tf, e_out, e_raw = calib(rs)
            print(f"| {tag} | {acc:.3f} | {conf:.3f} | {conf-acc:+.3f} | "
                  f"{e_raw:.4f} | {Tf:.3f} | {e_out:.4f} |")
        first = [r for r in next(iter(data.values())) if r.get("set") == s]
        n = len(first)
        # Число вариантов берём из длины оценок: сами варианты в выгрузку
        # оценок не пишутся, они лежат в файле строк набора.
        k = len(first[0].get("logits") or first[0]["probs"])
        print(f"\nСтрок {n}, вариантов {k}, случайное угадывание {1/k:.3f}.")


if __name__ == "__main__":
    main(sys.argv[1:])
