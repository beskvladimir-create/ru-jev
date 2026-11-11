#!/usr/bin/env python3
"""Temperature and ECE over half-splits, reproducibly.

What was wrong before. In aggregate.py and unseen_table.py the index list was
shuffled IN PLACE and never reset between splits:

    idx = list(range(len(rows)))
    for s in range(SPLITS):
        random.Random(s).shuffle(idx)     # idx is already shuffled by the previous step

Split s was therefore Random(s) applied to the permutation left by s-1: the
splits ran as a chain rather than as independent draws. The figures were still
deterministic, but they depended on the order of rows in the file and changed
whenever that order did, which is why they could not be reproduced.

Here the indices are reset on every split, each split's seed is derived from a
recorded SPLIT_SEED, and the rows are first put in a canonical order by their
own content, so the order of rows in the file cannot affect the result at all.

The canonical key of a row is (slice, text, index of the correct answer). It is
present in every score dump and is the same across models, so two models being
compared get the same splits row for row and the difference in their fitted
temperatures carries no split noise.
"""
import argparse, hashlib, json, math, os, random, statistics, sys

SPLIT_SEED = 20261105        # recorded; change it only together with a new set of figures
SPLITS = 25
BINS = 15


def row_key(r):
    s = r.get("set") or r.get("vocab") or r.get("task") or ""
    t = r.get("text") or r.get("input") or ""
    return f"{s}\x00{t}\x00{r.get('row_id', '')}\x00{r['answer']}"


def canonical(rows):
    """Order by the sha256 of the key: independent of the order in the file."""
    return sorted(rows, key=lambda r: hashlib.sha256(
        row_key(r).encode("utf-8")).hexdigest())


def logprobs(r, T):
    lg = r.get("logits")
    if lg is not None:
        return [x / T for x in lg]
    return [math.log(max(x, 1e-12)) / T for x in r["probs"]]


def nll(rows, T):
    s = 0.0
    for r in rows:
        p = logprobs(r, T)
        m = max(p)
        s -= (p[r["answer"]] - m - math.log(sum(math.exp(x - m) for x in p)))
    return s / max(len(rows), 1)


def fitT(rows, lo=0.2, hi=8.0, steps=300):
    best, bt = None, 1.0
    for i in range(steps + 1):
        T = lo + (hi - lo) * i / steps
        v = nll(rows, T)
        if best is None or v < best:
            best, bt = v, T
    return bt


def probs_at(r, T):
    p = logprobs(r, T)
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
    return sum(len(x) / n * abs(sum(c for c, _ in x) / len(x)
                                - sum(h for _, h in x) / len(x))
               for x in b if x)


def calibration(rows):
    rows = canonical(rows)
    n = len(rows)
    ts, es = [], []
    for s in range(SPLITS):
        idx = list(range(n))                      # reset on every split
        random.Random(SPLIT_SEED + s).shuffle(idx)
        h = n // 2
        T = fitT([rows[i] for i in idx[:h]])
        ts.append(T)
        es.append(ece([rows[i] for i in idx[h:]], T))
    return {"T_mean": statistics.mean(ts), "T_min": min(ts), "T_max": max(ts),
            "ECE_own_T": statistics.mean(es), "ECE_raw": ece(rows, 1.0),
            "n": n, "splits": SPLITS, "split_seed": SPLIT_SEED}


# ---------------------------------------------------------------------------
# Per-option biases.
#
# The temperature cannot change which option is picked: dividing every score by
# the same number leaves the argmax where it was. All it changes is how
# confident the number next to the answer looks. So when a model is wrong in a
# systematic way, by leaning towards one option whatever the input says, no
# temperature will repair it.
#
# One additive bias per option does change the argmax:
#
#     p_i = softmax(s_i / T + b[option_i])
#
# The biases are fitted together with the temperature by maximum likelihood on
# labelled rows. Only differences between biases matter, so one of them is held
# at zero; the one held is the first option in sorted order, which makes the
# reported numbers reproducible. The optimiser is L-BFGS over (log T, b), with
# log T as the parameter so the temperature stays positive.
#
# This is a correction of the caller's label set, not of the model. It is
# fitted on the caller's own labelled rows and it belongs with them: carrying
# biases fitted on one label vocabulary over to another is meaningless.

LBFGS_MEMORY = 10
LBFGS_MAX_ITER = 500
LBFGS_GTOL = 1e-9


def lbfgs(fun_grad, x0, memory=LBFGS_MEMORY, max_iter=LBFGS_MAX_ITER,
          gtol=LBFGS_GTOL, ftol=1e-12, bounds=None):
    """Limited-memory BFGS with a backtracking Armijo line search.

    Written out here rather than taken from scipy so that the calibration and
    the scripts that reproduce its figures depend on nothing but the standard
    library and run identically anywhere.

    bounds, when given, is one (low, high) pair per coordinate and every trial
    point is projected into the box. It is there to stop a run away rather than
    to shape the answer: the box is set so wide in fit_calibration that no fit
    which converges can reach it, and every temperature this repository
    publishes sits in the middle of it.
    """
    def project(v):
        if bounds is None:
            return v
        return [min(max(c, lo), hi) for c, (lo, hi) in zip(v, bounds)]

    x = project(list(x0))
    f, g = fun_grad(x)
    S, Y, RHO = [], [], []
    it = 0
    for it in range(1, max_iter + 1):
        if max(abs(v) for v in g) < gtol:
            break
        q = [-v for v in g]
        alpha = []
        for s, y, rho in zip(reversed(S), reversed(Y), reversed(RHO)):
            a = rho * sum(si * qi for si, qi in zip(s, q))
            alpha.append(a)
            q = [qi - a * yi for qi, yi in zip(q, y)]
        if S:
            s, y = S[-1], Y[-1]
            yy = sum(v * v for v in y)
            if yy > 0:
                gamma = sum(si * yi for si, yi in zip(s, y)) / yy
                q = [gamma * qi for qi in q]
        for (s, y, rho), a in zip(zip(S, Y, RHO), reversed(alpha)):
            beta = rho * sum(yi * qi for yi, qi in zip(y, q))
            q = [qi + (a - beta) * si for qi, si in zip(q, s)]
        d = q
        slope = sum(di * gi for di, gi in zip(d, g))
        if slope >= 0:                      # lost curvature, restart on -gradient
            d = [-v for v in g]
            slope = -sum(v * v for v in g)
            S, Y, RHO = [], [], []
        step, fn, gn, xn = 1.0, None, None, None
        for _ in range(60):
            xn = project([xi + step * di for xi, di in zip(x, d)])
            fn, gn = fun_grad(xn)
            if fn <= f + 1e-4 * step * slope:
                break
            step *= 0.5
        else:
            break                           # line search failed, keep the best point
        s = [step * di for di in d]
        y = [b - a for a, b in zip(g, gn)]
        ys = sum(si * yi for si, yi in zip(s, y))
        if ys > 1e-12:
            S.append(s)
            Y.append(y)
            RHO.append(1.0 / ys)
            if len(S) > memory:
                S.pop(0)
                Y.pop(0)
                RHO.pop(0)
        done = abs(f - fn) <= ftol * max(1.0, abs(f))
        x, f, g = xn, fn, gn
        if done:
            break
    return x, f, it


def option_vocab(samples):
    """Every option string seen, in sorted order. The first is held at zero."""
    seen = set()
    for _, opts, _ in samples:
        seen.update(opts)
    return sorted(seen)


def _prepare(samples, vocab):
    """(scores, option indices, index of the correct option) per row."""
    at = {o: i for i, o in enumerate(vocab)}
    out = []
    for lg, opts, ans in samples:
        if len(lg) != len(opts):
            raise ValueError("a row has %d scores and %d options" % (len(lg), len(opts)))
        out.append(([float(x) for x in lg], [at[o] for o in opts], int(ans)))
    return out


def _nll_grad(theta, data, k, fit_T):
    """Mean negative log-likelihood of softmax(s / T + b) and its gradient.

    theta is [log T, b_1 .. b_{k-1}] with b_0 held at zero, or just the biases
    when the temperature is not being fitted.
    """
    if fit_T:
        logT = theta[0]
        biases = [0.0] + list(theta[1:])
    else:
        logT = 0.0
        biases = [0.0] + list(theta)
    inv_T = math.exp(-logT)
    total = 0.0
    g_logT = 0.0
    g_b = [0.0] * k
    n = max(len(data), 1)
    for scores, idx, ans in data:
        u = [s * inv_T for s in scores]
        z = [ui + biases[j] for ui, j in zip(u, idx)]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        tot = sum(e)
        p = [v / tot for v in e]
        total -= z[ans] - m - math.log(tot)
        # d/d logT: u_a - sum_i p_i u_i   (because d u_i / d logT = -u_i)
        g_logT += u[ans] - sum(pi * ui for pi, ui in zip(p, u))
        for pos, j in enumerate(idx):
            g_b[j] += p[pos]
        g_b[idx[ans]] -= 1.0
    grad = [v / n for v in g_b[1:]]          # b_0 is held at zero
    if fit_T:
        return total / n, [g_logT / n] + grad
    return total / n, grad


def fit_calibration(samples, fit_temperature=True, start_T=1.0):
    """Fit a temperature and one bias per option on labelled rows.

    samples is an iterable of (scores, options, index of the correct option).
    Returns a dict in the shape written to calibration.json.
    """
    samples = [(lg, [str(o) for o in opts], int(ans)) for lg, opts, ans in samples]
    if not samples:
        raise ValueError("no rows to fit on")
    vocab = option_vocab(samples)
    data = _prepare(samples, vocab)
    k = len(vocab)
    x0 = ([math.log(start_T)] if fit_temperature else []) + [0.0] * (k - 1)
    # A box far outside anything meaningful: a temperature between 1e-3 and 1e3
    # and biases within 50 of each other, which on logits of order ten is total
    # certainty either way. It only catches the case where the likelihood keeps
    # improving as the temperature goes to zero, which happens on a slice the
    # model separates perfectly and which would otherwise overflow.
    bounds = ([(-7.0, 7.0)] if fit_temperature else []) + [(-50.0, 50.0)] * (k - 1)
    x, f, iters = lbfgs(lambda th: _nll_grad(th, data, k, fit_temperature), x0,
                        bounds=bounds)
    if fit_temperature:
        T = math.exp(x[0])
        b = [0.0] + list(x[1:])
    else:
        T = 1.0
        b = [0.0] + list(x)
    return {"temperature": T, "biases": dict(zip(vocab, b)), "options": vocab,
            "n_rows": len(samples), "held_at_zero": vocab[0],
            "nll": f, "iterations": iters, "fitted_temperature": bool(fit_temperature)}


def fit_T_only(samples, start_T=1.0):
    """The same likelihood with every bias held at zero, for comparison."""
    samples = [(lg, [str(o) for o in opts], int(ans)) for lg, opts, ans in samples]
    vocab = option_vocab(samples)
    data = _prepare(samples, vocab)

    def fg(th):
        f, g = _nll_grad([th[0]] + [0.0] * (len(vocab) - 1), data, len(vocab), True)
        return f, [g[0]]

    x, f, _ = lbfgs(fg, [math.log(start_T)], bounds=[(-7.0, 7.0)])
    return math.exp(x[0])


def load_calibration(path):
    """Read a calibration.json, from a file or from a model directory."""
    if os.path.isdir(path):
        path = os.path.join(path, "calibration.json")
    cal = json.load(open(path, encoding="utf-8"))
    return {"temperature": float(cal.get("temperature", 1.0)),
            "biases": {str(k): float(v) for k, v in (cal.get("biases") or {}).items()},
            "max_len": int(cal.get("max_len", 256))}


def apply_calibration(scores, options, temperature=1.0, biases=None, warn=True):
    """softmax(s / T + b[option]). Options without a fitted bias get zero.

    An option the biases do not cover is not an error: a caller may add a label
    after fitting. It is reported once per option, because a missing bias means
    that label is being scored on a different footing from the rest.
    """
    T = max(float(temperature), 1e-2)
    biases = biases or {}
    z = []
    for s, o in zip(scores, options):
        o = str(o)
        if biases and o not in biases:
            if warn and o not in _WARNED:
                _WARNED.add(o)
                print("warning: no fitted bias for option %r, using 0" % o,
                      file=sys.stderr)
            z.append(float(s) / T)
        else:
            z.append(float(s) / T + biases.get(o, 0.0))
    m = max(z)
    e = [math.exp(v - m) for v in z]
    tot = sum(e)
    return [v / tot for v in e]


_WARNED = set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="append", required=True, help="name=path")
    ap.add_argument("--by-set", action="store_true",
                    help="report separately per set or vocab field")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    data = {}
    for spec in a.scores:
        name, path = spec.split("=", 1)
        data[name] = [json.loads(l) for l in open(path, encoding="utf-8")]
    out = {}
    groups = [None]
    if a.by_set:
        groups = []
        for rows in data.values():
            for r in rows:
                g = r.get("set") or r.get("vocab")
                if g and g not in groups:
                    groups.append(g)
    print(f"SPLIT_SEED = {SPLIT_SEED}, splits {SPLITS}, ECE bins {BINS}\n")
    for g in groups:
        if g:
            print(f"## {g}\n")
        print("| Model | n | Fitted T | T min | T max | ECE as is | ECE after its own T |")
        print("|---|---|---|---|---|---|---|")
        for name, rows in data.items():
            rs = rows if g is None else [r for r in rows
                                         if (r.get("set") or r.get("vocab")) == g]
            if not rs:
                continue
            c = calibration(rs)
            out[f"{g or 'all'}|{name}"] = c
            print(f"| {name} | {c['n']} | {c['T_mean']:.4f} | {c['T_min']:.3f} | "
                  f"{c['T_max']:.3f} | {c['ECE_raw']:.4f} | {c['ECE_own_T']:.4f} |")
        print()
    if a.json_out:
        json.dump(out, open(a.json_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"figures: {a.json_out}")


if __name__ == "__main__":
    main()
