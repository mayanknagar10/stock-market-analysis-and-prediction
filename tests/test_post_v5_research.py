"""Leakage and immutability tests for the actual experiment path."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

class ResearchTests(unittest.TestCase):
    def api(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('research.post_v5.experiments'), 'bounded experiment API missing')
        from research.post_v5 import experiments
        return experiments

    def fixture(self):
        dates=pd.bdate_range('2023-01-02','2025-11-01')
        rng=np.random.default_rng(42)
        frames={}
        for ticker in ['A.NS','B.NS','C.NS']:
            close=100*np.exp(np.cumsum(rng.normal(.0002,.01,len(dates))))
            frames[ticker]=pd.DataFrame({'Open':close,'High':close*1.01,'Low':close*.99,'Close':close,'Volume':1e6},index=dates)
        universe={'members':[{'ticker':t,'industry':'Group'} for t in frames]}
        return frames,universe

    def test_future_sentinel_excluded_through_actual_experiment(self):
        e=self.api();frames,u=self.fixture()
        p,_=e.build_panel(frames,{}, {},u,5)
        modified={k:v.copy() for k,v in frames.items()}
        for v in modified.values():v.loc[v.index>=pd.Timestamp('2025-10-01'),'Close']=1e15
        q,_=e.build_panel(modified,{}, {},u,5)
        pd.testing.assert_frame_equal(p,q)
        a=e.evaluate_panel(p,[['2025-07-01','2025-09-30']],5,models=['ridge'])
        b=e.evaluate_panel(q,[['2025-07-01','2025-09-30']],5,models=['ridge'])
        pd.testing.assert_frame_equal(a['predictions'],b['predictions'])
        self.assertLess(pd.to_datetime(p.label_end,utc=True).max(),pd.Timestamp('2025-10-01',tz='UTC'))

    def test_all_preprocessing_and_pruning_fit_training_only(self):
        e=self.api()
        train=pd.DataFrame({'x':[1.,2.,3.,4.], 'duplicate':[1.,2.,3.,4.], 'missing':[np.nan,1.,2.,3.]})
        test=pd.DataFrame({'x':[1e12], 'duplicate':[-1e12], 'missing':[1e15]})
        fit=e.fit_transform(train,test,prune=True)
        self.assertEqual(fit['fit_rows'],4)
        self.assertEqual(fit['medians']['x'],2.5)
        self.assertNotIn('duplicate',fit['columns'])
        self.assertEqual(fit['columns'],e.fit_transform(train,test*1e6,prune=True)['columns'])
        np.testing.assert_allclose(fit['train'],e.fit_transform(train,test*1e6,prune=True)['train'])

    def test_reference_labels_are_never_predictors_or_oracle_reconstruction(self):
        e=self.api()
        frame=pd.DataFrame({'market_future':[100.], 'sector_future':[500.], 'actual':[600.], 'x':[1.]})
        self.assertEqual(e.predictor_columns(frame,{'technical':['x'],'market':['market_future']}),['x'])
        pred=e.reconstruct('market_excess',np.array([.01]),np.array([.02]),np.array([.03]),np.array([.04]))
        np.testing.assert_allclose(pred,[.03])
        pred=e.reconstruct('sector_excess',np.array([.01]),np.array([.02]),np.array([.03]),np.array([.04]))
        np.testing.assert_allclose(pred,[.06])

    def test_validation_future_reference_changes_cannot_change_predictions(self):
        e=self.api();frames,u=self.fixture()
        reference=frames['B.NS'].copy()
        p,_=e.build_panel(frames,{'market_india':reference},{},u,5)
        q=p.copy();q.attrs=p.attrs.copy()
        validation=pd.to_datetime(q.feature_time,utc=True)>=pd.Timestamp('2025-07-01',tz='UTC')
        q.loc[validation,['market_future','sector_future']]=1e9
        a=e.evaluate_panel(p,[['2025-07-01','2025-09-30']],5,models=['market_excess','sector_excess','decomposition'])
        b=e.evaluate_panel(q,[['2025-07-01','2025-09-30']],5,models=['market_excess','sector_excess','decomposition'])
        np.testing.assert_allclose(a['predictions'].prediction,b['predictions'].prediction,equal_nan=True)
        self.assertTrue(np.isfinite(a['predictions'].prediction).all())
        for audit in a['audits']:
            self.assertLess(pd.Timestamp(audit['fit_feature_max']),pd.Timestamp('2025-07-01',tz='UTC'))
            self.assertLess(pd.Timestamp(audit['fit_label_max']),pd.Timestamp('2025-07-01',tz='UTC'))
            self.assertFalse(set(audit['columns'])&{'actual','market_future','sector_future'})

    def test_market_only_and_market_sector_baselines_have_no_stock_relative_features(self):
        e=self.api();frames,u=self.fixture()
        p,_=e.build_panel(frames,{'market_india':frames['B.NS'].copy()},{},u,5)
        result=e.evaluate_panel(p,[['2025-07-01','2025-09-30']],5,models=['market_only','market_sector'])
        for audit in result['audits']:
            self.assertFalse(any(c.startswith(('stock_minus','rolling_market')) for c in audit['columns']))
            self.assertFalse(any(c.startswith('ret_') for c in audit['columns']))

    def test_sector_target_reconstruction_restricts_to_available_sector_labels(self):
        e=self.api();frames,u=self.fixture()
        frames['D.NS']=frames['A.NS'].copy();frames['E.NS']=frames['B.NS'].copy()
        u['members'] += [{'ticker':'D.NS','industry':'Small'},{'ticker':'E.NS','industry':'Small'}]
        p,_=e.build_panel(frames,{'market_india':frames['B.NS'].copy()},{},u,5)
        result=e.evaluate_panel(p,[['2025-07-01','2025-09-30']],5,models=['sector_excess','decomposition'])
        r=result['predictions']
        self.assertTrue(r.loc[r.industry=='Small','prediction'].isna().all())
        self.assertTrue(r.loc[r.industry=='Group','prediction'].notna().all())

    def test_zero_volume_and_missing_intervening_sessions_excluded_in_actual_panel(self):
        e=self.api();frames,u=self.fixture()
        volume_day=pd.Timestamp('2024-07-01');missing_day=pd.Timestamp('2024-08-01')
        frames['A.NS'].loc[volume_day,'Volume']=0.
        frames['B.NS']=frames['B.NS'].drop(missing_day)
        calendar=frames['C.NS'].index
        p,_=e.build_panel(frames,{}, {},u,5)
        for ticker,bad_day in [('A.NS',volume_day),('B.NS',missing_day)]:
            location=calendar.get_loc(bad_day)
            forbidden=calendar[location-5:location+200]
            origins=pd.to_datetime(p.loc[p.ticker==ticker,'origin_session'])
            self.assertFalse(origins.isin(forbidden).any())
            self.assertTrue((origins<forbidden[0]).any())
            self.assertTrue((origins>forbidden[-1]).any())
        out=e.evaluate_panel(p,[['2025-07-01','2025-09-30']],5,models=['ridge'])
        self.assertTrue(np.isfinite(out['predictions'].prediction).all())
        self.assertLess(pd.to_datetime(out['predictions'].label_end,utc=True).max(),pd.Timestamp('2025-10-01',tz='UTC'))

    def test_weekend_maturity_session_and_information_cutoff_remain_distinct(self):
        e=self.api();frames,u=self.fixture()
        special=pd.Timestamp('2025-02-01')
        for ticker,frame in list(frames.items()):
            frame.loc[special]=frame.loc[pd.Timestamp('2025-01-31')]
            frames[ticker]=frame.sort_index()
        panel,_=e.build_panel(frames,{}, {},u,1)
        row=panel.loc[(panel.ticker=='A.NS')&(panel.origin_session=='2025-01-31')].iloc[0]
        self.assertEqual(row.maturity_session,'2025-02-01')
        self.assertEqual(pd.Timestamp(row.label_end),pd.Timestamp('2025-02-02T00:20:00Z'))
        self.assertEqual(float(row.actual),0.)

    def test_rank_target_only_same_origin(self):
        e=self.api()
        p=pd.DataFrame({'origin_session':['2025-01-01']*3+['2025-01-02']*2,'actual':[1.,2.,3.,100.,200.]})
        ranks=e.rank_target(p)
        np.testing.assert_allclose(ranks,[1/3,2/3,1.,.5,1.])

    def test_folds_disjoint_and_labels_purged(self):
        from research.post_v5.development import fixed_folds
        p=pd.DataFrame({'feature_time':pd.to_datetime(['2024-12-20','2024-12-23','2025-01-02'],utc=True),'label_end':pd.to_datetime(['2024-12-31','2025-01-03','2025-01-03'],utc=True)})
        tr,va=list(fixed_folds(p,[['2025-01-01','2025-03-31']]))[0]
        np.testing.assert_array_equal(tr,[0]);np.testing.assert_array_equal(va,[2])

    def test_result_write_is_exclusive(self):
        e=self.api()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json';e.exclusive_json(p,{'a':1})
            with self.assertRaises(FileExistsError):e.exclusive_json(p,{'a':2})
            self.assertEqual(p.read_text().strip(),'{"a": 1}')

if __name__=='__main__':unittest.main()
