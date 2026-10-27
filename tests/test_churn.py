import unittest

import numpy as np
import pandas as pd

from churn import config
from churn.data import clean
from churn.preprocessing import build_preprocessor
from churn.risk import risk_tier, segment
from churn.sample_data import make_sample

try:
    from fastapi.testclient import TestClient
    from churn.api import app
    HAS_API = True
except Exception:  # fastapi/httpx not installed
    HAS_API = False

RAW = make_sample(n=800, seed=1)
DF = clean(RAW)
VALID = DF[config.FEATURES].iloc[0].to_dict()


class TestData(unittest.TestCase):
    def test_clean_types_and_target(self):
        self.assertTrue(pd.api.types.is_numeric_dtype(DF["TotalCharges"]))
        self.assertTrue(set(DF[config.TARGET].unique()) <= {0, 1})
        self.assertTrue(set(DF["SeniorCitizen"].unique()) <= {"Yes", "No"})

    def test_blank_total_charges_become_nan(self):
        self.assertEqual(DF["TotalCharges"].isna().sum(), (RAW["tenure"] == 0).sum())

    def test_duplicates_dropped(self):
        dup = pd.concat([RAW, RAW.iloc[:5]])
        self.assertEqual(len(clean(dup)), len(RAW))

    def test_bad_target_raises(self):
        bad = RAW.copy(); bad.loc[0, "Churn"] = "Maybe"
        with self.assertRaises(ValueError):
            clean(bad)


class TestPipeline(unittest.TestCase):
    def test_preprocessor_handles_nan_and_unseen_category(self):
        prep = build_preprocessor().fit(DF[config.FEATURES])
        row = DF[config.FEATURES].iloc[[0]].copy()
        row["TotalCharges"] = np.nan
        row["PaymentMethod"] = "Crypto"  # unseen
        out = prep.transform(row)
        self.assertFalse(np.isnan(out.toarray() if hasattr(out, "toarray") else out).any())

    def test_scaler_fit_on_train_only(self):
        train, test = DF.iloc[:600], DF.iloc[600:]
        prep = build_preprocessor().fit(train[config.FEATURES])
        scaler = prep.named_transformers_["num"].named_steps["scale"]
        self.assertAlmostEqual(scaler.mean_[0], train["tenure"].mean(), places=6)
        self.assertNotAlmostEqual(scaler.mean_[0], DF["tenure"].mean(), places=6)


class TestRisk(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(risk_tier(0.1), "Low")
        self.assertEqual(risk_tier(0.45), "Medium")
        self.assertEqual(risk_tier(0.9), "High")

    def test_segment_counts(self):
        d = pd.DataFrame({"churn_probability": [.1, .5, .9, .95], "Churn": [0, 0, 1, 1],
                          "MonthlyCharges": [10, 20, 30, 40]})
        s = segment(d, months=12)
        self.assertEqual(s.loc["High", "customers"], 2)
        self.assertEqual(s.loc["High", "assumed_revenue_exposure"], 70 * 12)


@unittest.skipUnless(HAS_API, "fastapi/httpx not installed")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not config.MODEL_PATH.exists():
            raise unittest.SkipTest("Train a model first: python -m churn.train")
        cls.client = TestClient(app)

    def test_health(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_predict_ok(self):
        r = self.client.post("/predict", json={k: (None if pd.isna(v) else v) for k, v in VALID.items()})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(0 <= body["churn_probability"] <= 1)
        self.assertIn(body["risk_tier"], {"Low", "Medium", "High"})

    def test_invalid_category_422(self):
        bad = {k: (None if pd.isna(v) else v) for k, v in VALID.items()}
        bad["Contract"] = "Weekly"
        self.assertEqual(self.client.post("/predict", json=bad).status_code, 422)

    def test_missing_field_422(self):
        self.assertEqual(self.client.post("/predict", json={"tenure": 3}).status_code, 422)


class TestPredict(unittest.TestCase):
    def test_predict_one_matches_pipeline(self):
        from churn import predict
        if not config.MODEL_PATH.exists():
            self.skipTest("Train a model first")
        rec = {k: (None if pd.isna(v) else v) for k, v in VALID.items()}
        out = predict.predict_one(rec)
        self.assertTrue(0 <= out["churn_probability"] <= 1)
        with self.assertRaises(ValueError):
            predict.predict_dataframe(pd.DataFrame([{"tenure": 1}]))


if __name__ == "__main__":
    unittest.main()
