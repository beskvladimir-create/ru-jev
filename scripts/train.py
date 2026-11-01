#!/usr/bin/env python3
"""Train ru-jev (Жевский): a calibrated typed-decision scorer on mmBERT.

Model: encoder backbone + linear scoring head over (context [SEP] option) pairs.
For each example the options are scored and a softmax gives the decision
distribution. Loss is cross-entropy over the options. After training we fit a
single temperature on the validation set so the probabilities are calibrated,
which is what makes selective routing (auto vs human) possible.

Usage: python scripts/train.py --config configs/base.yaml
"""
import json, os, argparse, math, yaml, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

HERE = os.path.dirname(__file__)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class DecisionData(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collate(tok, max_len):
    def collate(batch):
        max_opts = max(len(b["options"]) for b in batch)
        contexts, options, mask = [], [], []
        answers = []
        for b in batch:
            opts = b["options"]
            answers.append(b["answer"])
            row_mask = []
            for j in range(max_opts):
                if j < len(opts):
                    contexts.append(b["input"]); options.append(str(opts[j])); row_mask.append(1)
                else:
                    contexts.append(b["input"]); options.append(str(opts[0])); row_mask.append(0)
            mask.append(row_mask)
        enc = tok(contexts, options, truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "opt_mask": torch.tensor(mask, dtype=torch.float),   # [E, M]
            "answers": torch.tensor(answers, dtype=torch.long),  # [E]
            "n_opts": max_opts,
        }
    return collate


class Scorer(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base)
        h = self.backbone.config.hidden_size
        self.head = nn.Linear(h, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]           # first-token pooling
        return self.head(cls).squeeze(-1)           # [pairs]


def grouped_logits(scores, n_opts, opt_mask):
    E = opt_mask.shape[0]
    logits = scores.view(E, n_opts)
    logits = logits.masked_fill(opt_mask == 0, float("-inf"))
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "..", "configs", "base.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(HERE, "..", cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = Scorer(cfg["base_model"]).to(device)

    train_rows = load_jsonl(os.path.join(HERE, "..", "data", "train.jsonl"))
    val_rows = load_jsonl(os.path.join(HERE, "..", "data", "val.jsonl"))
    collate = make_collate(tok, cfg["max_len"])
    dl = DataLoader(DecisionData(train_rows), batch_size=cfg["batch_examples"],
                    shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    total = len(dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(total * cfg["warmup_ratio"]), total)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    lossf = nn.CrossEntropyLoss()

    model.train()
    for ep in range(cfg["epochs"]):
        running = 0.0
        for step, batch in enumerate(dl):
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            opt_mask = batch["opt_mask"].to(device)
            ans = batch["answers"].to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                scores = model(ids, am)
                logits = grouped_logits(scores, batch["n_opts"], opt_mask)
                loss = lossf(logits, ans)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"ep{ep} step{step}/{len(dl)} loss {running/(step+1):.4f}")
        print(f"== epoch {ep} done, mean loss {running/len(dl):.4f}")

    # ---- temperature scaling on val ----
    model.eval()
    val_dl = DataLoader(DecisionData(val_rows), batch_size=cfg["batch_examples"],
                        shuffle=False, collate_fn=collate)
    all_logits, all_ans = [], []
    with torch.no_grad():
        for batch in val_dl:
            ids = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
            scores = model(ids, am)
            logits = grouped_logits(scores, batch["n_opts"], batch["opt_mask"].to(device))
            # pad to a common width with -inf so we can stack mixed option counts
            all_logits.append(logits.cpu()); all_ans.append(batch["answers"])
    T = fit_temperature(all_logits, all_ans)
    print(f"fitted temperature T = {T:.4f}")

    model.backbone.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    torch.save(model.head.state_dict(), os.path.join(out_dir, "head.pt"))
    json.dump({"temperature": T, "base_model": cfg["base_model"], "max_len": cfg["max_len"]},
              open(os.path.join(out_dir, "calibration.json"), "w"))
    print(f"saved to {out_dir}")


def fit_temperature(logit_batches, ans_batches):
    """Fit a single scalar T minimizing NLL over variable-width option logits."""
    T = torch.nn.Parameter(torch.ones(1))
    optim = torch.optim.LBFGS([T], lr=0.05, max_iter=100)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optim.zero_grad()
        loss = torch.tensor(0.0)
        n = 0
        for lb, ab in zip(logit_batches, ans_batches):
            scaled = lb / T.clamp(min=1e-2)
            loss = loss + nll(scaled, ab) * lb.shape[0]
            n += lb.shape[0]
        loss = loss / max(n, 1)
        loss.backward()
        return loss

    optim.step(closure)
    return float(T.detach().clamp(min=1e-2).item())


if __name__ == "__main__":
    main()
