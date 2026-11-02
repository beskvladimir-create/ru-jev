#!/usr/bin/env python3
"""Capability probe: run a checkpoint on the decisions people actually ask for.

The dev splits measure the training tasks. They do not say whether the model can
route a support ticket or answer a free-form yes/no question, which is what a
caller of the serving API will try first. This probe is a small hand-written set
of such cases with the answer a human would give. It is a smoke test of
behaviour, not a benchmark: the cases are written by the author, and the point
is to compare two checkpoints on the same cases.

Usage: python scripts/probe.py --model out/ru-jev-v1.1 [--model-b model/ru-jev-v1]
"""
import json, os, argparse
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

HERE = os.path.dirname(__file__)

ROUTING = ["оплата, счета и возвраты", "техническая проблема", "продажи"]
BILLING, TECH, SALES = ROUTING

CASES = [
    # group, context, options, expected
    ("routing", "Списали деньги дважды за один заказ.", ROUTING, BILLING),
    ("routing", "Не приходит счёт на почту уже неделю.", ROUTING, BILLING),
    ("routing", "Прошу вернуть деньги за отменённый заказ.", ROUTING, BILLING),
    ("routing", "Приложение падает при входе.", ROUTING, TECH),
    ("routing", "Не открывается личный кабинет, ошибка 500.", ROUTING, TECH),
    ("routing", "Забыл пароль, восстановление не приходит.", ROUTING, TECH),
    ("routing", "Хочу узнать условия для корпоративных клиентов.", ROUTING, SALES),
    ("routing", "Сколько стоит тариф на год для десяти сотрудников?", ROUTING, SALES),

    ("boolean", "Клиента списали дважды за один заказ, он просит вернуть лишний платёж.\nВопрос: клиент просит возврат денег?", ["да", "нет"], "да"),
    ("boolean", "Клиент пишет, что всё работает, претензий нет.\nВопрос: клиент просит возврат денег?", ["да", "нет"], "нет"),
    ("boolean", "Заказ доставлен вовремя, упаковка целая.\nВопрос: есть ли проблема с доставкой?", ["да", "нет"], "нет"),
    ("boolean", "Курьер не приехал, посылка потерялась на складе.\nВопрос: есть ли проблема с доставкой?", ["да", "нет"], "да"),

    ("sentiment", "Отличный сервис, всё быстро, рекомендую.", ["негативный", "нейтральный", "позитивный"], "позитивный"),
    ("sentiment", "Третий раз пишу, никто не отвечает, ужасно.", ["негативный", "нейтральный", "позитивный"], "негативный"),
    ("sentiment", "Заказ получен, вопросов нет.", ["негативный", "нейтральный", "позитивный"], "нейтральный"),

    ("toxicity", "Ты вообще читать умеешь, бестолочь?", ["токсичный", "нетоксичный"], "токсичный"),
    ("toxicity", "Добрый день, подскажите статус заявки, пожалуйста.", ["токсичный", "нетоксичный"], "нетоксичный"),

    # the same decisions with English option wording, which callers do use
    ("en labels", "Списали деньги дважды за один заказ.", ["billing", "support", "sales"], "billing"),
    ("en labels", "Приложение падает при входе.", ["billing", "support", "sales"], "support"),
    ("en labels", "Отличный сервис, всё быстро, рекомендую.", ["negative", "neutral", "positive"], "positive"),
]


class Scorer(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.head.load_state_dict(torch.load(os.path.join(path, "head.pt"), map_location="cpu"))

    def forward(self, ids, am):
        return self.head(self.backbone(input_ids=ids, attention_mask=am).last_hidden_state[:, 0]).squeeze(-1)


def run(path, verbose=True):
    calib = json.load(open(os.path.join(path, "calibration.json")))
    T = float(calib.get("temperature", 1.0))
    max_len = int(calib.get("max_len", 256))
    tok = AutoTokenizer.from_pretrained(path)
    model = Scorer(path).eval()
    per_group = defaultdict(lambda: [0, 0, 0.0])
    with torch.no_grad():
        for group, ctx, opts, want in CASES:
            enc = tok([ctx] * len(opts), opts, truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"]) / max(T, 1e-2), dim=0)
            i = int(torch.argmax(probs))
            ok = opts[i] == want
            per_group[group][0] += int(ok)
            per_group[group][1] += 1
            per_group[group][2] += float(probs[i])
            if verbose:
                print(f"  {'OK ' if ok else 'MISS'} [{group:9s}] -> {opts[i]!r} {float(probs[i]):.3f}"
                      f"{'' if ok else '  want ' + want!r}")
    return per_group


def report(name, per_group):
    print(f"\n== {name} ==")
    tc = tn = 0
    for g in sorted(per_group):
        c, n, conf = per_group[g]
        tc += c; tn += n
        print(f"  {g:10s} {c}/{n}  mean confidence {conf/n:.3f}")
    print(f"  {'TOTAL':10s} {tc}/{tn} = {tc/max(tn,1):.3f}")
    return tc, tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "..", "out", "ru-jev-v1.1"))
    ap.add_argument("--model-b", default=None, help="second checkpoint to compare against")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    a = run(args.model, verbose=not args.quiet)
    report(args.model, a)
    if args.model_b:
        b = run(args.model_b, verbose=not args.quiet)
        report(args.model_b, b)


if __name__ == "__main__":
    main()
