#!/usr/bin/env python3
"""Подогнать температуру модели под свои данные.

Зачем. Температура из релиза подобрана на нашей валидации. На ваших решениях
распределение уверенности другое, и вероятности будут смещены. Точность от
температуры НЕ зависит вообще: софтмакс по вариантам монотонен, аргмакс не
меняется. Меняется только то, насколько честно число рядом с ответом.

Сколько строк нужно. Несколько сотен уже дают устойчивую оценку. Скрипт считает
температуру на половинных разбиениях и печатает разброс, чтобы было видно,
хватило ли данных: если край разброса далеко от середины, строк мало.

Формат входа, по строке на решение:
  {"input": "текст запроса", "options": ["вариант", ...], "answer": 0}
Поле "text" тоже принимается вместо "input".

Пример:
  python scripts/fit_temperature.py --model besktech/ru-jev \
      --rows my_decisions.jsonl --out my_calibration.json
"""
import argparse, json, math, os, random, statistics

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

SPLITS = 25
BINS = 15


class Scorer(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.head.load_state_dict(_load_head(path))

    def forward(self, ids, mask):
        cls = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


def _load_head(path):
    """Голова лежит в safetensors; .pt принимается для старых выгрузок."""
    from huggingface_hub import hf_hub_download
    for name, reader in (("head.safetensors", None), ("head.pt", torch.load)):
        local = os.path.join(path, name)
        if os.path.exists(local):
            src = local
        else:
            try:
                src = hf_hub_download(path, name)
            except Exception:
                continue
        if reader is None:
            from safetensors.torch import load_file
            return load_file(src)
        return reader(src, map_location="cpu")
    raise SystemExit("не найдена голова: ни head.safetensors, ни head.pt")


def nll(scored, T):
    s = 0.0
    for lg, ans in scored:
        p = [x / T for x in lg]
        m = max(p)
        s -= (p[ans] - m - math.log(sum(math.exp(x - m) for x in p)))
    return s / max(len(scored), 1)


def fit(scored, lo=0.2, hi=8.0, steps=300, rounds=4):
    """Сетка по NLL, затем сужение вокруг найденной точки."""
    best = 1.0
    for _ in range(rounds):
        cand = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
        best = min(cand, key=lambda t: nll(scored, t))
        step = (hi - lo) / steps
        lo, hi = max(0.05, best - step), best + step
    return best


def probs(lg, T):
    p = [x / T for x in lg]
    m = max(p)
    e = [math.exp(x - m) for x in p]
    z = sum(e)
    return [x / z for x in e]


def ece(scored, T):
    b = [[] for _ in range(BINS)]
    for lg, ans in scored:
        p = probs(lg, T)
        j = max(range(len(p)), key=lambda k: p[k])
        b[min(int(p[j] * BINS), BINS - 1)].append((p[j], int(j == ans)))
    n = sum(len(x) for x in b)
    return sum(len(x) / n * abs(sum(c for c, _ in x) / len(x)
                                - sum(h for _, h in x) / len(x))
               for x in b if x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="путь к папке или имя репозитория на хабе")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", help="куда записать calibration.json с новой температурой")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261105, help="сид разбиений, записывается в отчёт")
    a = ap.parse_args()

    max_len = a.max_len
    if max_len is None:
        try:
            cal = os.path.join(a.model, "calibration.json")
            if os.path.exists(cal):
                max_len = int(json.load(open(cal)).get("max_len", 256))
            else:
                from huggingface_hub import hf_hub_download
                max_len = int(json.load(open(hf_hub_download(a.model, "calibration.json")))
                              .get("max_len", 256))
        except Exception:
            max_len = 256

    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()]
    if len(rows) < 100:
        print(f"предупреждение: строк всего {len(rows)}, оценка будет шумной")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Scorer(a.model).eval()

    scored = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            text = r.get("input", r.get("text"))
            opts = [str(o) for o in r["options"]]
            enc = tok([text] * len(opts), opts, truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt")
            lg = model(enc["input_ids"], enc["attention_mask"]).float().tolist()
            scored.append((lg, int(r["answer"])))
            if (i + 1) % 200 == 0:
                print(f"  посчитано {i+1}/{len(rows)}", flush=True)

    acc = sum(lg.index(max(lg)) == ans for lg, ans in scored) / len(scored)
    T_all = fit(scored)

    idx = list(range(len(scored)))
    ts, es = [], []
    for s in range(SPLITS):
        order = list(idx)
        random.Random(a.seed + s).shuffle(order)
        h = len(order) // 2
        T = fit([scored[i] for i in order[:h]])
        ts.append(T)
        es.append(ece([scored[i] for i in order[h:]], T))

    print(f"\nстрок {len(scored)}, точность {acc:.4f} (от температуры не зависит)")
    print(f"температура на всех строках: {T_all:.4f}")
    print(f"по {SPLITS} половинным разбиениям, сид {a.seed}: "
          f"среднее {statistics.mean(ts):.4f}, от {min(ts):.3f} до {max(ts):.3f}")
    print(f"ECE при температуре 1: {ece(scored, 1.0):.4f}")
    print(f"ECE после подобранной:  {statistics.mean(es):.4f}")
    spread = max(ts) - min(ts)
    if spread > 0.5:
        print(f"разброс температуры {spread:.2f}, это много: добавьте строк")

    if a.out:
        json.dump({"temperature": T_all, "max_len": max_len, "calibrated": True,
                   "fitted_on": os.path.basename(a.rows), "n_rows": len(scored),
                   "split_seed": a.seed, "accuracy": round(acc, 4)},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nзаписано: {a.out}")


if __name__ == "__main__":
    main()
