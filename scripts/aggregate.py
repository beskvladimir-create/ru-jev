#!/usr/bin/env python3
"""Сводка по стенду формулировок для всех моделей на одних и тех же строках.

Температура подбирается по сетке минимизацией NLL на одной половине и ECE
считается на другой, по 25 случайным разбиениям. Это тот же метод, которым
считается карточка, иначе числа между моделями несопоставимы.
"""
import json, math, random, statistics, sys, collections

VOCAB_ORDER = ["русское слово", "английский ключ", "описание запроса",
               "русская фраза", "названия служб", "жаргон"]
SPLITS = 25
# ВНИМАНИЕ. Прежняя реализация разбиений здесь перемешивала список
# индексов НА МЕСТЕ и не сбрасывала его между разбиениями, из-за чего
# разбиение s применялось к перестановке от s-1 и результат зависел от
# порядка строк в файле. Калибровка вынесена в calib.py, где индексы
# сбрасываются, сид каждого разбиения выводится из записанного
# SPLIT_SEED, а строки ставятся в канонический порядок.
from calib import calibration as _calibration  # noqa: E402

BINS = 15


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def nll(rows, T):
    s = 0.0
    for r in rows:
        lg = r.get("logits")
        if lg is not None:
            m = max(x / T for x in lg)
            z = sum(math.exp(x / T - m) for x in lg)
            s -= (lg[r["answer"]] / T - m - math.log(z))
        else:  # вероятности без логитов: берём log p, температуру применяем к log p
            p = [math.log(max(x, 1e-12)) / T for x in r["probs"]]
            m = max(p); z = sum(math.exp(x - m) for x in p)
            s -= (p[r["answer"]] - m - math.log(z))
    return s / max(len(rows), 1)


def fitT(rows):
    # Строки, где вероятность верного варианта пришла нулём из-за округления
    # выдачи до четырёх знаков, в подгонку не берём: log(0) утащил бы
    # температуру вверх. Их число печатается отдельно.
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
    lg = r.get("logits")
    src = [x / T for x in lg] if lg is not None else [math.log(max(x, 1e-12)) / T for x in r["probs"]]
    m = max(src); e = [math.exp(x - m) for x in src]; z = sum(e)
    return [x / z for x in e]


def ece(rows, T):
    buckets = [[] for _ in range(BINS)]
    for r in rows:
        p = probs_at(r, T)
        pick = max(range(len(p)), key=lambda j: p[j])
        conf = p[pick]
        b = min(int(conf * BINS), BINS - 1)
        buckets[b].append((conf, int(pick == r["answer"])))
    n = sum(len(b) for b in buckets)
    return sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b))
               for b in buckets if b)



def _wrap(rows):
    """Возвращает (T, ECE после своей T, сырой ECE), как раньше."""
    c = _calibration(rows)
    return c["T_mean"], c["ECE_own_T"], c["ECE_raw"]

calibration = calib = _wrap


def main(files):
    models = {}
    for tag, path in files:
        models[tag] = load(path)

    print("## Стенд формулировок, одни и те же 1000 вопросов и 18 категорий на каждую строку\n")
    head = "| словарь | " + " | ".join(models) + " |"
    print(head)
    print("|" + "---|" * (len(models) + 1))
    for v in VOCAB_ORDER:
        cells = []
        for tag, rows in models.items():
            rs = [r for r in rows if r["vocab"] == v]
            acc = sum(r["pick"] == r["answer"] for r in rs) / len(rs)
            cells.append(f"{acc:.3f}")
        print(f"| {v} | " + " | ".join(cells) + " |")

    print("\n### Жаргон подробно\n")
    print("| модель | 18 категорий | 15 чистых | средняя уверенность | зазор уверенность минус точность | уверенно неверно из 1000 |")
    print("|---|---|---|---|---|---|")
    for tag, rows in models.items():
        rs = [r for r in rows if r["vocab"] == "жаргон"]
        cl = [r for r in rs if r["gold_clean"]]
        acc = sum(r["pick"] == r["answer"] for r in rs) / len(rs)
        acc_cl = sum(r["pick"] == r["answer"] for r in cl) / len(cl)
        conf = sum(r["p_pick"] for r in rs) / len(rs)
        over = sum(1 for r in rs if r["pick"] != r["answer"] and r["p_pick"] > 0.8)
        print(f"| {tag} | {acc:.3f} | {acc_cl:.3f} | {conf:.3f} | {conf-acc:+.3f} | {over} |")

    print("\n### Калибровка на всех 6000 строках стенда\n")
    print("| модель | ECE как есть | подобранная T | ECE после своей T |")
    print("|---|---|---|---|")
    for tag, rows in models.items():
        T, e_out, e_raw = calibration(rows)
        dropped = sum(1 for r in rows
                      if r.get("logits") is None and r["probs"][r["answer"]] <= 0)
        note = f" (из подгонки исключено строк: {dropped})" if dropped else ""
        print(f"| {tag} | {e_raw:.4f} | {T:.3f} | {e_out:.4f} |{note}")


if __name__ == "__main__":
    main([(a.split("=")[0], a.split("=", 1)[1]) for a in sys.argv[1:]])
