#!/usr/bin/env python3
"""Train Ru-Jev: a calibrated typed-decision scorer on mmBERT.

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


def pick_device(requested=None):
    """cuda, then Apple MPS, then cpu. MPS lets the whole run happen on a Mac."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def make_collate(tok, max_len, shuffle_options=False):
    """shuffle_options re-orders the candidates of every example on every epoch,
    so the model cannot lean on the position of the correct answer."""
    def collate(batch):
        max_opts = max(len(b["options"]) for b in batch)
        contexts, options, mask = [], [], []
        answers = []
        for b in batch:
            opts = b["options"]
            ans = b["answer"]
            if shuffle_options and len(opts) > 1:
                order = list(range(len(opts)))
                random.shuffle(order)
                opts = [opts[i] for i in order]
                ans = order.index(ans)
            answers.append(ans)
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


def save_checkpoint(model, tok, out_dir, cfg, temperature=None, epoch=None):
    """Write the backbone, tokenizer, head and calibration file to out_dir.

    temperature None means it has not been fitted yet: the file then carries 1.0
    and calibrated false, so a checkpoint can never be mistaken for a calibrated
    one just because it exists.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.backbone.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    torch.save(model.head.state_dict(), os.path.join(out_dir, "head.pt"))
    meta = {"temperature": float(temperature) if temperature is not None else 1.0,
            "base_model": cfg["base_model"], "max_len": cfg["max_len"],
            "calibrated": temperature is not None}
    if epoch is not None:
        meta["epoch"] = epoch
    json.dump(meta, open(os.path.join(out_dir, "calibration.json"), "w"))
    state = "calibrated" if temperature is not None else "temperature not fitted yet"
    print(f"checkpoint saved to {out_dir} after epoch {epoch} ({state})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "..", "configs", "base.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = pick_device(cfg.get("device"))
    print(f"device: {device}")
    out_dir = os.path.join(HERE, "..", cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = Scorer(cfg["base_model"]).to(device)

    data_dir = cfg.get("data_dir") or os.path.join(HERE, "..", "data")
    train_rows = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val_rows = load_jsonl(os.path.join(data_dir, "val.jsonl"))
    print(f"data: {len(train_rows)} train / {len(val_rows)} val from {data_dir}")
    collate = make_collate(tok, cfg["max_len"],
                           shuffle_options=bool(cfg.get("shuffle_options", True)))
    eval_collate = make_collate(tok, cfg["max_len"], shuffle_options=False)
    dl = DataLoader(DecisionData(train_rows), batch_size=cfg["batch_examples"],
                    shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    total = len(dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(total * cfg["warmup_ratio"]), total)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
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
            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                scores = model(ids, am)
                logits = grouped_logits(scores, batch["n_opts"], opt_mask)
                loss = lossf(logits, ans)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"ep{ep} step{step}/{len(dl)} loss {running/(step+1):.4f}")
        print(f"== epoch {ep} done, mean loss {running/len(dl):.4f}")
        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for b in DataLoader(DecisionData(val_rows), batch_size=cfg["batch_examples"],
                                shuffle=False, collate_fn=eval_collate):
                ids = b["input_ids"].to(device); am = b["attention_mask"].to(device)
                lg = grouped_logits(model(ids, am), b["n_opts"], b["opt_mask"].to(device))
                a = b["answers"].to(device)
                vl += float(lossf(lg, a)) * a.shape[0]
                vc += int((lg.argmax(dim=1) == a).sum()); vn += a.shape[0]
        print(f"   val loss {vl/max(vn,1):.4f}  val acc {vc/max(vn,1):.4f}  (n={vn})")
        # Save after every epoch. A run can die for reasons that have nothing to
        # do with the code, a spent balance among them, and an epoch-old
        # checkpoint is worth incomparably more than none.
        save_checkpoint(model, tok, out_dir, cfg, temperature=None, epoch=ep)
        model.train()

    # Calibration runs after this on purpose: a checkpoint with a placeholder
    # temperature is recoverable, a run that dies before saving is not.
    save_checkpoint(model, tok, out_dir, cfg, temperature=None, epoch="final")

    # ---- temperature scaling on val ----
    model.eval()
    val_dl = DataLoader(DecisionData(val_rows), batch_size=cfg["batch_examples"],
                        shuffle=False, collate_fn=eval_collate)
    all_logits, all_masks, all_ans = [], [], []
    with torch.no_grad():
        for batch in val_dl:
            ids = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
            scores = model(ids, am)
            # keep the raw grid and the mask apart: dividing -inf by T gives a NaN
            # gradient, which silently produced T = nan on mixed option counts
            raw = scores.view(batch["opt_mask"].shape[0], batch["n_opts"])
            all_logits.append(raw.float().cpu())
            all_masks.append(batch["opt_mask"].cpu())
            all_ans.append(batch["answers"])
    T = fit_temperature(all_logits, all_masks, all_ans)
    print(f"fitted temperature T = {T:.4f}")

    save_checkpoint(model, tok, out_dir, cfg, temperature=T, epoch="final")


def fit_temperature(logit_batches, mask_batches, ans_batches):
    """Fit a single scalar T minimizing NLL over variable-width option logits.

    Everything is stacked into one tensor first. An earlier version looped over
    the per-batch tensors inside the optimiser closure, which rebuilt a graph
    over every validation batch on each of the ~125 evaluations and turned a
    two-second fit into more than an hour on CPU.

    The padded slots are masked out *after* the division by T. Masking before it
    would put -inf into the arithmetic and hand the optimiser a NaN gradient.
    """
    width = max(lb.shape[1] for lb in logit_batches)
    logits, masks, answers = [], [], []
    for lb, mb, ab in zip(logit_batches, mask_batches, ans_batches):
        pad = width - lb.shape[1]
        if pad:
            lb = torch.nn.functional.pad(lb, (0, pad))
            mb = torch.nn.functional.pad(mb, (0, pad))
        logits.append(lb.float()); masks.append(mb); answers.append(ab)
    L = torch.cat(logits).float()
    M = torch.cat(masks)
    A = torch.cat(answers)
    nll = torch.nn.CrossEntropyLoss()

    def loss_at(t):
        scaled = (L / t).masked_fill(M == 0, float("-inf"))
        return nll(scaled, A)

    # coarse scan first, so the optimiser starts near the answer and a bad
    # starting point cannot send it somewhere silly
    with torch.no_grad():
        grid = torch.linspace(0.25, 6.0, 48)
        best = min(grid, key=lambda g: float(loss_at(float(g))))
    T = torch.nn.Parameter(torch.tensor([float(best)]))
    optim = torch.optim.LBFGS([T], lr=0.05, max_iter=60)

    def closure():
        optim.zero_grad()
        loss = loss_at(T.clamp(min=1e-2))
        loss.backward()
        return loss

    optim.step(closure)
    t = float(T.detach().clamp(min=1e-2).item())
    if not (t == t) or t <= 0:          # NaN or nonsense, fall back to the scan
        print(f"[warn] temperature fit failed, falling back to the scan value {float(best):.3f}")
        return float(best)
    with torch.no_grad():
        print(f"    nll at T=1: {float(loss_at(1.0)):.4f}  at T={t:.3f}: {float(loss_at(t)):.4f}")
    return t


if __name__ == "__main__":
    main()
