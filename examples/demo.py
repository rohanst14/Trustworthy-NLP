from trustworthy_nlp import NaiveBayes
import json
print(json.dumps(NaiveBayes().fit([('normal safe',0),('danger alarm',1)]).predict('danger alarm',.6),indent=2))
