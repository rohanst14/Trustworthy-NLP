"""Held-out probability calibration and selective classification."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, logsumexp

def validate(logits, labels):
    x=np.asarray(logits,dtype=float)
    y=np.asarray(labels)
    if x.ndim!=2 or not len(x) or x.shape[1]<2 or y.shape!=(len(x),) or not np.isfinite(x).all():
        raise ValueError("Invalid logits or labels")
    if y.dtype.kind not in "iu" or np.any(y<0) or np.any(y>=x.shape[1]):
        raise ValueError("Invalid label indices")
    return x,y

def nll(logits,labels):
    x,y=validate(logits,labels)
    return float(np.mean(logsumexp(x,axis=1)-x[np.arange(len(y)),y]))

def fit_temperature(logits,labels):
    x,y=validate(logits,labels)
    opt=minimize_scalar(lambda t:nll(x/np.exp(t),y),bounds=(-4,4),method="bounded")
    if not opt.success: raise RuntimeError("Calibration failed")
    return float(np.exp(opt.x))

def metrics(logits,labels,temperature=1.,threshold=0.,bins=15):
    x,y=validate(logits,labels)
    if not np.isfinite(temperature) or temperature<=0 or not 0<=threshold<=1 or bins<1:
        raise ValueError("Invalid metric parameters")
    p=softmax(x/temperature,axis=1)
    conf=p.max(axis=1)
    correct=p.argmax(axis=1)==y
    kept=conf>=threshold
    groups=np.minimum((conf*bins).astype(int),bins-1)
    reliability=[]
    for b in range(bins):
        mask=groups==b
        if mask.any():
            reliability.append({"bin":b,"count":int(mask.sum()),"confidence":float(conf[mask].mean()),"accuracy":float(correct[mask].mean())})
    ece=sum(r["count"]/len(y)*abs(r["confidence"]-r["accuracy"]) for r in reliability)
    return {"count":len(y),"accuracy":float(correct.mean()),"nll":nll(x/temperature,y),
            "brier":float(np.mean(np.sum((p-np.eye(x.shape[1])[y])**2,axis=1))),
            "ece":float(ece),"coverage":float(kept.mean()),"selective_risk":float(1-correct[kept].mean()) if kept.any() else None,
            "reliability":reliability}

def select_threshold(logits,labels,temperature,target_risk=.1):
    if not 0<=target_risk<=1: raise ValueError("Invalid target risk")
    best=None
    for t in np.linspace(0,1,101):
        m=metrics(logits,labels,temperature,float(t))
        if m["selective_risk"] is not None and m["selective_risk"]<=target_risk:
            if best is None or m["coverage"]>best[1]: best=(float(t),m["coverage"])
    return best[0] if best else None
