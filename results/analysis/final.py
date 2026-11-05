import json, sys, numpy as np
from scipy.stats import binomtest
R=sys.argv[1]; rng=np.random.default_rng(20261105)
L=lambda p:[json.loads(l) for l in open(p)]
T={'v11':2.2754,'v13':1.8640}
def probs(r,m):
    if 'logits' in r: z=np.array(r['logits'])/T[m]; p=np.exp(z-z.max()); return p/p.sum()
    return np.array(r['probs'])
v2A=[r for r in L(f'{R}/vocab2/rows.jsonl') if r['set']=='стиль_A']
cand5=[[o[:-4].lower() for o in r['options']] for r in v2A]; gold=[r['gold_label'] for r in v2A]
cells=[]
def stats(name,k,G,ok11,ok13,cf11,cf13,okL=None):
    x,y=ok11,ok13; b,c=int((x&~y).sum()),int((~x&y).sum())
    p=binomtest(min(b,c),b+c,.5).pvalue if b+c else 1.0
    ul=np.unique(G); byl={l:np.where(G==l)[0] for l in ul}; cb=[]
    for _ in range(3000):
        ii=np.concatenate([rng.choice(byl[l],len(byl[l])) for l in rng.choice(ul,len(ul))]); cb.append(y[ii].mean()-x[ii].mean())
    per=[y[byl[l]].mean()-x[byl[l]].mean() for l in ul]
    ce=lambda ok,cf:1000*((~ok)&(cf>0.8)).mean()
    def aurc(ok,cf):
        o=np.argsort(-cf); e=1-ok[o].astype(float); return float(np.mean(np.cumsum(e)/np.arange(1,len(e)+1)))
    cells.append(dict(name=name,k=k,n=len(x),nl=len(ul),a=x.mean(),b_=y.mean(),d=y.mean()-x.mean(),B=b,C=c,p=p,
        ci=np.percentile(cb,[2.5,97.5]),up=sum(v>0 for v in per),dn=sum(v<0 for v in per),
        ce=(ce(x,cf11),ce(y,cf13)),au=(aurc(x,cf11),aurc(y,cf13)),gap=(cf11.mean()-x.mean(),cf13.mean()-y.mean()),L=(okL.mean() if okL is not None else np.nan)))
def evalset(rows,S,keep,cand=None):
    """rows: option lists; S: dict model->score rows; cand: restrict to these label sets (per row) using row label map"""
    out={}
    for m,sc in S.items():
        ok=[];cf=[]
        for i,(r,s) in enumerate(zip(rows,sc)):
            p=probs(s,m)
            if cand is not None:
                labs=r['_labels']; idx=[labs.index(x) for x in cand[i]]; p=p[idx]; p=p/p.sum()
                ok.append(cand[i][int(p.argmax())]==r['gold_label'])
            else: ok.append(int(p.argmax())==r['answer'])
            cf.append(p.max())
        out[m]=(np.array(ok),np.array(cf))
    sel=np.array([keep is None or r['gold_label'] in keep for r in rows])
    return {m:(a[sel],c[sel]) for m,(a,c) in out.items()}, np.array([r['gold_label'] for r in rows])[sel]
def add(name,k,rows,S,keep=None,cand=None):
    o,G=evalset(rows,S,keep,cand)
    stats(name,k,G,o['v11'][0],o['v13'][0],o['v11'][1],o['v13'][1],o['L'][0] if 'L' in o else None)
# --- card bench, 18-way (fix18) and 5-way derived from 18-way logits
fr=L(f'{R}/fix18/rows.jsonl'); fs={m:L(f'{R}/fix18/scores_{n}.jsonl') for m,n in [('v11','v11'),('v13','v13'),('L','laya_instrA')]}
ov=json.load(open(f'{R}/fix18/overlap_bench.json'))
fvoc=json.load(open(f'{R}/fix18/vocab.json'))
for v in list(dict.fromkeys(r['vocab'] for r in fr)):
    idx=[i for i,r in enumerate(fr) if r['vocab']==v]; rows=[fr[i] for i in idx]
    assert [r['gold_label'] for r in rows]==gold
    # map option strings to labels via position of gold? use vocab.json
    vm=fvoc[v] if isinstance(fvoc.get(v),dict) else None
    inv={s:l for l,s in (vm.get('метки',vm)).items()} if vm else None
    for r in rows: r['_labels']=[inv[o] for o in r['options']]
    S={m:[fs[m][i] for i in idx] for m in fs}
    variants=[('all',None)]
    if v=='жаргон':
        variants+=[('clean15',set(gold)-{'audio','news','weather'}),('strict',{l for l,d in ov[v].items() if d['чистая']})]
    for sl,keep in variants:
        add(f'bench {v} {sl}',18,rows,S,keep)
        add(f'bench {v} {sl}',5,rows,S,keep,cand5)
# --- vocab3 independent: 18-way native, 5-way derived
r3=L(f'{R}/vocab3/rows.jsonl'); v3=json.load(open(f'{R}/vocab3/vocab.json'))
s3={m:L(f'{R}/vocab3/scores_{n}.jsonl') for m,n in [('v11','v11'),('v13','v13'),('L','laya_instrA')]}
allL=set(gold)
for st,nm,strict in [('стиль_B','slang',allL-{'email','weather','datetime','iot'}),('стиль_A','opaque codes',{'alarm','audio','calendar','datetime','email','iot','lists','transport','weather'})]:
    idx=[i for i,r in enumerate(r3) if r['set']==st]; rows=[r3[i] for i in idx]
    inv={s:l for l,s in v3[st]['метки'].items()}
    for r in rows: r['_labels']=[inv[o] for o in r['options']]
    S={m:[s3[m][i] for i in idx] for m in s3}
    for sl,keep in [('all',None),('strict',strict)]:
        add(f'indep {nm} {sl}',18,rows,S,keep)
        add(f'indep {nm} {sl}',5,rows,S,keep,cand5)
# --- vocab2 rule-derived: 5-way native
r2=L(f'{R}/vocab2/rows.jsonl'); s2={m:L(f'{R}/vocab2/scores_{n}.jsonl') for m,n in [('v11','v11'),('v13','v13'),('L','laya_instrA')]}
for st,nm,clean in [('стиль_A','code identifiers',None),('стиль_B','abbreviations',allL-{'recommendation'})]:
    idx=[i for i,r in enumerate(r2) if r['set']==st]; rows=[r2[i] for i in idx]; S={m:[s2[m][i] for i in idx] for m in s2}
    add(f'rule {nm} all',5,rows,S)
    if clean: add(f'rule {nm} clean',5,rows,S,clean)
# --- unseen
ur=L(f'{R}/unseen/unseen_rows.jsonl'); u={'v11':L(f'{R}/unseen/unseen_v11.jsonl'),'v13':L(f'{R}/unseen/unseen_v13.jsonl')}
uL={'рубрики новостей':L(f'{R}/unseen/unseen_laya_headline.jsonl'),'эмоции':L(f'{R}/unseen/unseen_laya_cedr.jsonl')}
for st,nm in [('рубрики новостей','unseen headlines'),('эмоции','unseen emotions')]:
    idx=[i for i,r in enumerate(ur) if r['set']==st]; rows=[dict(ur[i],gold_label=ur[i]['gold']) for i in idx]
    S={'v11':[u['v11'][i] for i in idx],'v13':[u['v13'][i] for i in idx],'L':uL[st]}
    add(nm,len(rows[0]['options']),rows,S)
# Holm over held-out cells (exclude trained controls)
ctrl=lambda n: any(t in n for t in ['английский ключ','русское слово','русская фраза'])
H=[i for i,c in enumerate(cells) if not ctrl(c['name'])]
ps=np.array([cells[i]['p'] for i in H]); o=np.argsort(ps); m=len(ps); run=0
for rank,j in enumerate(o):
    run=max(run,(m-rank)*ps[j]); cells[H[j]]['holm']=min(1,run)
print(f"Holm family size {m}: all cells except the training-style controls (English key, Russian word, Russian phrase); includes the two unseen-dataset cells")
print('probe|k|n|labels|v1.1|v1.3|diff|labelCI|b|c|p|pHolm|up/dn|confErr v11->v13 per1000|Laya|AURC v11->v13|gap v11->v13')
for c in cells:
    print(f"{c['name']}|{c['k']}|{c['n']}|{c['nl']}|{c['a']:.3f}|{c['b_']:.3f}|{c['d']:+.3f}|[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]|{c['B']}|{c['C']}|{c['p']:.2g}|{c.get('holm',float('nan')):.2g}|{c['up']}/{c['dn']}|{c['ce'][0]:.0f}->{c['ce'][1]:.0f}|{c['L']:.3f}|{c['au'][0]:.3f}->{c['au'][1]:.3f}|{c['gap'][0]:+.3f}->{c['gap'][1]:+.3f}")
