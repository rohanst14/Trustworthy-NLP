from trustworthy_nlp import NaiveBayes,brier
import unittest
class Tests(unittest.TestCase):
    def setUp(self): self.model=NaiveBayes().fit([('normal safe',0),('danger alarm',1)])
    def test_class(self): self.assertEqual(self.model.predict('danger alarm',.6)['label'],1)
    def test_unknown(self): self.assertIsNone(self.model.predict('unseen')['label'])
    def test_brier(self): self.assertAlmostEqual(brier([.8,.2],[1,0]),.04)
