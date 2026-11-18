import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fit_award_only import KEY, TARGET, classify, baseline, award_features, reporting_totals, validate, training_offsets

class AwardOnlyTests(unittest.TestCase):
    def sample(self):
        rows=[]
        for p,n,y in [("a",100,20),("b",200,60),("c",50,25)]:
            for z in ["0–<1","1–<4"]:
                r=dict(player_key=p,Type="RS",zone=z,**{"Season Code":2020,"FGA Total":n})
                r.update({c:0 for c in TARGET});r[TARGET[0]]=y;r["target"]=y
                r.update(current_MVP=0,current_ALL_NBA=1,current_ALL_STAR=1,
                         current_scorer=1,other_FTA=999)
                rows.append(r)
        return pd.DataFrame(rows)
    def test_features_exclude_scorer_other_and_fg(self):
        x=award_features(self.sample(),"current")
        self.assertEqual(list(x),["intercept","current_MVP","current_ALL_NBA","current_ALL_STAR"])
    def test_independent_baseline(self):
        d=self.sample();train=d[d.player_key!="a"];test=d[d.player_key=="a"].copy()
        a=baseline(train,test);test.target=999999
        np.testing.assert_allclose(a,baseline(train,test))
    def test_volume_scaling(self):
        d=self.sample();train=d[d.player_key!="a"];test=d[d.player_key=="a"].copy()
        a=baseline(train,test);test["FGA Total"]*=2
        np.testing.assert_allclose(2*a,baseline(train,test))
    def test_fast_training_offsets_match_explicit_exclusion(self):
        d=self.sample();fast=training_offsets(d)
        for player,g in d.groupby("player_key"):
            slow=baseline(d[d.player_key!=player],g)
            np.testing.assert_allclose(fast.reindex(slow.index),slow)
    def test_legacy_bins_rejected(self):
        d=self.sample();d.loc[0,"zone"]="20 to <21"
        with self.assertRaises(ValueError):validate(d)
    def test_shot_categories(self):
        for text,expected in [("Driving Dunk Shot","dunk"),("Driving Layup Shot","driving_non_dunk"),
          ("Reverse Layup Shot","other_layup"),("Hook Shot","hook"),("Jump Shot","jumper"),
          (None,"unknown"),("Tip Shot","other")]:
            self.assertEqual(classify(text),expected)
    def test_other_fta_reporting_join_does_not_change_predictions(self):
        d=self.sample()
        p=d.groupby(KEY).target.sum().reset_index()
        p["award_only"]=p.target*1.1
        p["shot_baseline"]=p.target
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/"totals.csv"
            raw=p[KEY].copy();raw["FTA_Other"]=8;raw["games_played"]=4
            raw.to_csv(f,index=False)
            q=reporting_totals(p,d,str(f))
            np.testing.assert_allclose(q.other_FTA_per_game,2)
            np.testing.assert_allclose(q.award_only,p.award_only)
            np.testing.assert_allclose(q.target,p.target)
    def test_rs_totals_not_joined_to_playoffs(self):
        d=self.sample();p=d.groupby(KEY).target.sum().reset_index();p["Type"]="PO"
        p["award_only"]=p.target;p["shot_baseline"]=p.target
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/"totals.csv"
            raw=p[["player_key"]].copy();raw["season"]=2021;raw["FTA_Other"]=8;raw["GP"]=4
            raw.to_csv(f,index=False)
            q=reporting_totals(p,d,str(f))
            self.assertTrue(q.other_FTA_per_game.isna().all())

if __name__=="__main__":unittest.main()
