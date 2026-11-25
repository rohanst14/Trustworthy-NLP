import unittest
import numpy as np
from trustworthy_nlp.calibration import metrics,fit_temperature,nll,select_threshold
class CalibrationTests(unittest.TestCase):
    def test_perfect(self):
        m=metrics([[10,0],[0,10]],[0,1])
        self.assertEqual(m["accuracy"],1)
        self.assertLess(m["ece"],.001)
    def test_abstention(self):
        m=metrics([[0,0],[4,0]],[1,0],threshold=.8)
        self.assertEqual(m["coverage"],.5)
        self.assertEqual(m["selective_risk"],0)
    def test_empty_coverage(self):
        self.assertIsNone(metrics([[0,0]],[1],threshold=1)["selective_risk"])
    def test_calibration(self):
        x=np.array([[8,0],[8,0],[0,8],[0,8]])
        y=np.array([0,1,1,0])
        t=fit_temperature(x,y)
        self.assertGreater(t,1)
        self.assertLess(nll(x/t,y),nll(x,y))
    def test_bad_input(self):
        with self.assertRaises(ValueError): metrics([[0,1]],[2])
    def test_threshold(self):
        self.assertIsNotNone(select_threshold([[8,0],[0,8]],[0,1],1))
        self.assertIsNone(select_threshold([[8,0]],[1],1,0))
