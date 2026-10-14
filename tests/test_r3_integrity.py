"""Regression checks for missing inputs, identity errors and withdrawn outputs."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'r3_analysis'))
from _common import expected_units, load_reports, infer


class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs = self.root/'runs'; self.runs.mkdir()
        self.man = self.root/'manifest.json'
        self.man.write_text(json.dumps({'cells':[{'site':'A','fold':0,'feasible':True}, {'site':'B','fold':0,'feasible':True}], 'seeds':{'training':[666,2025,2026]}}))
        for tag, identity in expected_units(self.man).items():
            report = dict(identity, tag=tag, test_auc=1.)
            (self.runs/(tag+'_report.json')).write_text(json.dumps(report))
            np.savez_compressed(self.runs/(tag+'_oof.npz'), oof_role=np.array(['T']*4),
                                oof_label=np.array([0,0,1,1]), oof_logits=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]]), subject_ids=np.array(['a','b','c','d']))

    def tearDown(self):
        self.tmp.cleanup()

    def aggregate(self):
        out=self.root/'output.json'
        result=subprocess.run([sys.executable,str(ROOT/'r3_analysis/major1_aggregate.py'),'--manifest',str(self.man),'--runs',str(self.runs),'--boot','20','--out-json',str(out),'--out-csv',str(self.root/'out.csv')], capture_output=True,text=True)
        return result,out

    def test_complete_and_no_withdrawn_fields(self):
        p,out=self.aggregate();self.assertEqual(p.returncode,0,p.stderr)
        obj=json.loads(out.read_text());self.assertEqual(obj['fits_on_disk'],12)
        self.assertNotIn('p_two_sided',(self.root/'out.csv').read_text())
        self.assertNotIn('p_holm',(self.root/'out.csv').read_text())
        self.assertNotIn('bootstrap_two_sided_p',out.read_text())

    def test_missing_whole_cell(self):
        for f in self.runs.glob('cell001*'):f.unlink()
        p,out=self.aggregate();self.assertNotEqual(p.returncode,0);self.assertFalse(out.exists())

    def test_empty_directory(self):
        for f in self.runs.iterdir():f.unlink()
        p,out=self.aggregate();self.assertNotEqual(p.returncode,0);self.assertFalse(out.exists())

    def test_missing_single_run(self):
        next(self.runs.glob('*report.json')).unlink()
        with self.assertRaisesRegex(ValueError,'missing 1 official'):load_reports(self.man,self.runs)

    def test_stray_smoke_is_excluded(self):
        (self.runs/'smoke_report.json').write_text('{}')
        reports,expected=load_reports(self.man,self.runs);self.assertEqual(len(reports),12)
        p,out=self.aggregate();self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(out.read_text())['fits_on_disk'],12)

    def test_identity_corruption(self):
        f=next(self.runs.glob('*report.json'));obj=json.loads(f.read_text());obj['seed']=999;f.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'identity'):load_reports(self.man,self.runs)

    def test_missing_predictions(self):
        next(self.runs.glob('*npz')).unlink()
        with self.assertRaisesRegex(ValueError,'missing prediction'):load_reports(self.man,self.runs)

    def test_report_auc_corruption(self):
        f=next(self.runs.glob('*report.json'));obj=json.loads(f.read_text());obj['test_auc']=.4;f.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'AUC differs'):load_reports(self.man,self.runs)

    def test_sign_tails_and_zeros(self):
        a=infer({str(i):1. if i<8 else -1. for i in range(18)},b=100)
        self.assertAlmostEqual(a['sign_p'],.8145294189453125)
        self.assertTrue(0 <= a['null_centred_p'] <= 1)
        self.assertEqual(infer({'a':0.,'b':0.},b=10)['sign_p'],1.)


if __name__=='__main__':unittest.main()
