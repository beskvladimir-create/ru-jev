import json, sys, numpy as np
from scipy.stats import binomtest
R=sys.argv[1]; rng=np.random.default_rng(7)
L=lambda p:[json.loads(l) for l in open(p)]
v3=json.load(open(f'{R}/vocab3/vocab.json')); rows3=L(f'{R}/vocab3/rows.jsonl')
v2rows=[r for r in L(f'{R}/vocab2/rows.jsonl') if r['set']=='стиль_A']
cand5=[[o[:-4].lower() for o in r['options']] for r in v2rows]   # AUDIO_SCN -> audio
gold=[r['gold_label'] for r in v2rows]
assert all(g==c[r['answer']] for g,c,r in zip(gold,cand5,v2rows))
T={'v11':2.2754,'v13':1.8640}
def run(style, keep=None):
    inv={s:l for l,s in v3[style]['метки'].items()}
    R3=[r for r in rows3 if r['set']==style]; assert [r['gold_label'] for r in R3]==gold
    out={}
    for m in ['v11','v13']:
        S=[r for r in L(f'{R}/vocab3/scores_{m}.jsonl') if r['set']==style]
        ok=[];cf=[]
        for r3,s,c,g in zip(R3,S,cand5,gold):
            lab=[inv[o] for o in r3['options']]; idx=[lab.index(x) for x in c]
            z=np.array(s['logits'])[idx]/T[m]; p=np.exp(z-z.max()); p/=p.sum()
            ok.append(c[int(p.argmax())]==g); cf.append(p.max())
        out[m]=(np.array(ok),np.array(cf))
    sel=np.array([keep is None or g in keep for g in gold])
    x,y=out['v11'][0][sel],out['v13'][0][sel]; G=np.array(gold)[sel]
    b,c=int((x&~y).sum()),int((~x&y).sum()); p=binomtest(min(b,c),b+c,.5).pvalue
    ul=np.unique(G); byl={l:np.where(G==l)[0] for l in ul}; cb=[]
    for _ in range(3000):
        ii=np.concatenate([rng.choice(byl[l],len(byl[l])) for l in rng.choice(ul,len(ul))]); cb.append(y[ii].mean()-x[ii].mean())
    per=[y[byl[l]].mean()-x[byl[l]].mean() for l in ul]
    ce=lambda m:1000*((~out[m][0][sel])&(out[m][1][sel]>0.8)).mean()
    print(f"{style} {'all' if keep is None else 'subset'} n={sel.sum()} v11={x.mean():.3f} v13={y.mean():.3f} d={y.mean()-x.mean():+.3f} b={b} c={c} p={p:.2g} labelCI=[{np.percentile(cb,2.5):+.3f},{np.percentile(cb,97.5):+.3f}] up/dn={sum(d>0 for d in per)}/{sum(d<0 for d in per)} confErr/1000 {ce('v11'):.0f}->{ce('v13'):.0f}")
allL=set(gold)
run('стиль_B'); run('стиль_B', allL-{'email','weather','datetime','iot'})
run('стиль_A'); run('стиль_A', {'alarm','audio','calendar','datetime','email','iot','lists','transport','weather'})
