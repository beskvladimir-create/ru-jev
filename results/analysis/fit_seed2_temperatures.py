#!/usr/bin/env python3
"""Fit the temperature of a saved epoch, with the instrument the release used.

    python3 results/analysis/fit_seed2_temperatures.py \
        --config configs/v1_3_seed2.yaml --rows data/val.jsonl \
        out/snapshots/ep0 out/snapshots/ep1 out/snapshots/ep2 out/snapshots/epfinal

Why this is not `scripts/fit_temperature.py`. That script is the one shipped for
a reader to calibrate a release on their own decisions: it runs on the CPU, it
reports a spread over half-splits, and it can fit per-option biases. It is not
the path a released temperature came from. The number in a release is fitted by
`scripts/train.py` at the end of training, on the validation split of the
mixture, with a single scalar over the whole set.

This points that same path at a checkpoint on disk instead of the model that
happens to be in memory when a run ends: the validation pass and the
`fit_temperature` of `scripts/train.py`, the same collate with the option order
left alone, the batch size and the maximum length out of the same config, on the
same rows. It uses the card when there is one, as the training did.

It is useful because training fits a temperature once, for the epoch it stops
at, and a run that keeps all of its epochs has the others uncalibrated. The
fourth epoch is worth fitting again as a control: it has to return the figure the
training already wrote into the release.

Writes nothing into the checkpoints. It prints a row per epoch and a json
summary, and the caller records them.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch                                                       # noqa: E402
import yaml                                                        # noqa: E402
from torch.utils.data import DataLoader                            # noqa: E402
from transformers import AutoTokenizer                             # noqa: E402

import train as TR                                                 # noqa: E402
import fit_temperature as FT                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the config the run was trained with")
    ap.add_argument("--rows", required=True, help="the validation split of that mixture")
    ap.add_argument("snapshots", nargs="+", help="the checkpoint directories to fit")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    rows = TR.load_jsonl(a.rows)
    coll = TR.make_collate(tok, cfg["max_len"], shuffle_options=False)
    print("device %s, %d val rows, batch %d, max_len %d"
          % (dev, len(rows), cfg["batch_examples"], cfg["max_len"]), flush=True)

    lossf = torch.nn.CrossEntropyLoss()
    out = {}
    for snap in a.snapshots:
        model = FT.Scorer(snap).to(dev).eval()
        dl = DataLoader(TR.DecisionData(rows), batch_size=cfg["batch_examples"],
                        shuffle=False, collate_fn=coll)
        lg, mk, an = [], [], []
        hits, n, loss = 0, 0, 0.0
        with torch.no_grad():
            for b in dl:
                ids = b["input_ids"].to(dev)
                am = b["attention_mask"].to(dev)
                scores = model(ids, am)
                # the raw grid and the mask are kept apart on purpose: dividing
                # -inf by T gives a NaN gradient, which is how a temperature fit
                # on mixed option counts once came back as nan
                raw = scores.view(b["opt_mask"].shape[0], b["n_opts"])
                lg.append(raw.float().cpu())
                mk.append(b["opt_mask"].cpu())
                an.append(b["answers"])
                grouped = TR.grouped_logits(scores, b["n_opts"], b["opt_mask"].to(dev))
                ans = b["answers"].to(dev)
                loss += float(lossf(grouped, ans)) * ans.shape[0]
                hits += int((grouped.argmax(dim=1) == ans).sum())
                n += ans.shape[0]
        T = TR.fit_temperature(lg, mk, an)
        out[os.path.basename(os.path.normpath(snap))] = {
            "temperature": round(float(T), 4), "val_loss": round(loss / n, 4),
            "val_acc": round(hits / n, 4), "n": n}
        print("%s  T = %.4f  val loss %.4f  val acc %.4f  (n=%d)"
              % (snap, T, loss / n, hits / n, n), flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
