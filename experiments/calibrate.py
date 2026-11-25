"""Train local text classifier; reserve calibration and policy splits."""
import argparse,json,random,hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from trustworthy_nlp.calibration import fit_temperature,metrics,select_threshold

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/calibration.json")
    p.add_argument("--epochs",type=int,default=5)
    a=p.parse_args()
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    train=fetch_20newsgroups(subset="train",remove=("headers","footers","quotes"),random_state=42)
    test=fetch_20newsgroups(subset="test",remove=("headers","footers","quotes"),random_state=42)
    ids=np.arange(len(train.target))
    fit,hold=train_test_split(ids,test_size=.30,stratify=train.target,random_state=42)
    cal,policy=train_test_split(hold,test_size=.5,stratify=train.target[hold],random_state=42)
    vec=TfidfVectorizer(max_features=5000,min_df=2,sublinear_tf=True,dtype=np.float32)
    x=vec.fit_transform([train.data[i] for i in fit]).toarray()
    model=tf.keras.Sequential([tf.keras.Input((x.shape[1],)),tf.keras.layers.Dense(128,activation="relu"),tf.keras.layers.Dropout(.2),tf.keras.layers.Dense(len(train.target_names))])
    model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(x,train.target[fit],epochs=a.epochs,batch_size=64,verbose=2)
    def predict(text): return model.predict(vec.transform(text).toarray(),batch_size=128,verbose=0)
    temp=fit_temperature(predict([train.data[i] for i in cal]),train.target[cal])
    threshold=select_threshold(predict([train.data[i] for i in policy]),train.target[policy],temp)
    z=predict(test.data)
    report={"dataset":"20 Newsgroups","seed":42,"epochs":a.epochs,"train_count":len(fit),"calibration_count":len(cal),"policy_count":len(policy),"test_count":len(test.target),"temperature":temp,"threshold":threshold,"uncalibrated":metrics(z,test.target),"calibrated":metrics(z,test.target,temp),"selective":metrics(z,test.target,temp,threshold) if threshold is not None else {"coverage":0,"selective_risk":None},"risk_coverage":[{"threshold":float(t),**{k:v for k,v in metrics(z,test.target,temp,float(t)).items() if k in ("coverage","selective_risk")}} for t in np.linspace(0,1,21)],"tensorflow":tf.__version__}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ("temperature","threshold","test_count","uncalibrated","calibrated","selective")},indent=2))
if __name__=="__main__": main()
