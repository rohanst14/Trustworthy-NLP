import math,re
from collections import Counter
class NaiveBayes:
    def fit(self,rows):
        self.counts={0:Counter(),1:Counter()}; self.docs=Counter(); self.vocab=set()
        for text,label in rows:
            if label not in (0,1): raise ValueError('binary labels required')
            words=re.findall(r'\w+',text.lower()); self.counts[label].update(words); self.vocab.update(words); self.docs[label]+=1
        if not self.docs[0] or not self.docs[1] or not self.vocab: raise ValueError('both classes and nonempty vocabulary required')
        return self
    def predict(self,text,threshold=.7):
        if not .5<=threshold<=1: raise ValueError('threshold must be in [.5,1]')
        scores=[math.log(self.docs[c]/sum(self.docs.values())) for c in (0,1)]; explanation={}
        for w in re.findall(r'\w+',text.lower()):
            if w not in self.vocab: continue
            terms=[math.log((self.counts[c][w]+1)/(sum(self.counts[c].values())+len(self.vocab))) for c in (0,1)]
            for c in (0,1): scores[c]+=terms[c]
            explanation[w]=terms[1]-terms[0]
        peak=max(scores); z=[math.exp(s-peak) for s in scores]; p=z[1]/sum(z); confidence=max(p,1-p)
        return {'probability':p,'label':int(p>=.5) if explanation and confidence>=threshold else None,'confidence':confidence,'token_log_odds':explanation}
def brier(probabilities,labels):
    if len(probabilities)!=len(labels) or not labels: raise ValueError('aligned nonempty inputs required')
    if any(not 0<=p<=1 for p in probabilities) or any(y not in (0,1) for y in labels): raise ValueError('invalid probabilities or labels')
    return sum((p-y)**2 for p,y in zip(probabilities,labels))/len(labels)
