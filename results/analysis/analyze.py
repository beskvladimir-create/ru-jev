import json, sys, numpy as np
from scipy.stats import binomtest
R = sys.argv[1]; rng = np.random.default_rng(20261105)
T = {'v11': 2.2754, 'v13': 1.8640}
def load(p): return [json.loads(l) for l in open(p)]
def sm(lg, t): z = np.array(lg)/t; z -= z.max(); e = np.exp(z); return e/e.sum()
def conf_ok(rows, key):
    out = []
    for r in rows:
        if 'logits' in r: p = sm(r['logits'], T[key])
        else: p = np.array(r['probs'])
        out.append((p.max(), int(np.argmax(p)) == r['answer']))
    return np.array(out)
def aurc(co):
    o = np.argsort(-co[:,0]); err = 1-co[o,1]; return float(np.mean(np.cumsum(err)/np.arange(1,len(err)+1)))
cells = []
def cell(name, k, labels, a, b, c=None, keep=None):
    idx = np.arange(len(labels)) if keep is None else np.array([i for i,l in enumerate(labels) if l in keep])
    L = np.array(labels)[idx]; A = a[idx]; B = b[idx]
    x, y = A[:,1].astype(bool), B[:,1].astype(bool)
    bb, cc = int((x&~y).sum()), int((~x&y).sum())
    p = binomtest(min(bb,cc), bb+cc, 0.5).pvalue if bb+cc else 1.0
    d = y.mean()-x.mean()
    boots = [ (y[s].mean()-x[s].mean()) for s in (rng.integers(0,len(x),len(x)) for _ in range(5000))]
    ul = np.unique(L); byl = {l: np.where(L==l)[0] for l in ul}; cb=[]
    for _ in range(3000):
        pick = rng.choice(ul, len(ul)); ii = np.concatenate([rng.choice(byl[l], len(byl[l])) for l in pick])
        cb.append(y[ii].mean()-x[ii].mean())
    per = {l:(y[byl[l]].mean()-x[byl[l]].mean()) for l in ul}
    up = sum(v>0 for v in per.values()); dn = sum(v<0 for v in per.values())
    r = dict(cell=name, k=k, n=len(x), acc11=x.mean(), acc13=y.mean(), diff=d, b=bb, c=cc, p=p,
             ci=np.percentile(boots,[2.5,97.5]), cci=np.percentile(cb,[2.5,97.5]), up=up, dn=dn,
             macro=np.mean(list(per.values())), aurc11=aurc(A), aurc13=aurc(B),
             churn=(bb+cc)/len(x))
    if c is not None: r['accL']=c[idx][:,1].mean(); r['aurcL']=aurc(c[idx])
    cells.append(r)
# main bench: labels from vocab2 style A rows (same question order)
v2 = load(f'{R}/vocab2/rows.jsonl'); labA = [r['gold_label'] for r in v2 if r['set']=='стиль_A']; txtA=[r['text'] for r in v2 if r['set']=='стиль_A']
b11, b13, bL = load(f'{R}/bench_rujev_v11.jsonl'), load(f'{R}/bench_rujev_v13.jsonl'), load(f'{R}/bench_laya_instrA.jsonl')
vocs = list(dict.fromkeys(r['vocab'] for r in b11))
for v in vocs:
    s11=[r for r in b11 if r['vocab']==v]; s13=[r for r in b13 if r['vocab']==v]; sL=[r for r in bL if r['vocab']==v]
    assert [r['text'] for r in s11]==txtA, v
    keep = None if v!='жаргон' else set(labA)-{'audio','news','weather'}
    cell(f'bench5:{v}', 5, labA, conf_ok(s11,'v11'), conf_ok(s13,'v13'), conf_ok(sL,'L'))
    if keep: cell(f'bench5:{v}:clean15', 5, labA, conf_ok(s11,'v11'), conf_ok(s13,'v13'), conf_ok(sL,'L'), keep)
for vd,k,sl in [('vocab2',5,{'стиль_A':[None],'стиль_B':[None,{'recommendation'}]}),
                ('vocab3',18,{'стиль_A':[None,'cleanA','strictA'],'стиль_B':[None,'strictB']})]:
    rows=load(f'{R}/{vd}/rows.jsonl'); s11=load(f'{R}/{vd}/scores_v11.jsonl'); s13=load(f'{R}/{vd}/scores_v13.jsonl'); sL=load(f'{R}/{vd}/scores_laya_instrA.jsonl')
    allL = sorted(set(r['gold_label'] for r in rows))
    defs={'cleanA':set(allL)-{'play'},'strictA':{'alarm','audio','calendar','datetime','email','iot','lists','transport','weather'},
          'strictB':set(allL)-{'email','weather','datetime','iot'}}
    for st,variants in sl.items():
        f=lambda S:[r for r in S if r['set']==st]
        labs=[r['gold_label'] for r in f(rows)]
        for var in variants:
            keep = None if var is None else (set(allL)-var if isinstance(var,set) else defs[var])
            nm = f'{vd}:{st}' + ('' if var is None else (':clean' if isinstance(var,set) else ':'+var))
            cell(nm, k, labs, conf_ok(f(s11),'v11'), conf_ok(f(s13),'v13'), conf_ok(f(sL),'L'), keep)
ur=load(f'{R}/unseen/unseen_rows.jsonl'); u11=load(f'{R}/unseen/unseen_v11.jsonl'); u13=load(f'{R}/unseen/unseen_v13.jsonl')
uL={'рубрики новостей':load(f'{R}/unseen/unseen_laya_headline.jsonl'),'эмоции':load(f'{R}/unseen/unseen_laya_cedr.jsonl')}
for st in ['рубрики новостей','эмоции']:
    f=lambda S:[r for r in S if r['set']==st]
    labs=[r['gold'] for r in f(ur)]; k=len(f(ur)[0]['options'])
    cell(f'unseen:{st}', k, labs, conf_ok(f(u11),'v11'), conf_ok(f(u13),'v13'), conf_ok(uL[st],'L'))
# Holm over primary cells (exclude duplicate slices? include all)
ps=np.array([c['p'] for c in cells]); o=np.argsort(ps); m=len(ps); holm=np.empty(m); run=0
for rank,i in enumerate(o): run=max(run,(m-rank)*ps[i]); holm[i]=min(1,run)
print('cell|k|n|v1.1|v1.3|diff|95%CI q|95%CI labels|b|c|p|p_Holm|labels up/down|churn|AURC v1.1|AURC v1.3|Laya acc|Laya AURC')
for c,h in zip(cells,holm):
    print(f"{c['cell']}|{c['k']}|{c['n']}|{c['acc11']:.3f}|{c['acc13']:.3f}|{c['diff']:+.3f}|[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]|[{c['cci'][0]:+.3f},{c['cci'][1]:+.3f}]|{c['b']}|{c['c']}|{c['p']:.2g}|{h:.2g}|{c['up']}/{c['dn']}|{c['churn']:.2f}|{c['aurc11']:.3f}|{c['aurc13']:.3f}|{c.get('accL',float('nan')):.3f}|{c.get('aurcL',float('nan')):.3f}")
