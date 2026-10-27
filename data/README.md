# Data

The project expects the **IBM Telco Customer Churn** dataset (one CSV, 7,043 rows × 21 columns in the commonly distributed version).

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` (IBM sample data, widely mirrored on Kaggle: search "Telco Customer Churn" by *blastchar*).
2. Place it at: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Verify the license/terms on the page you download it from; the CSV is intentionally **not** committed (see `.gitignore`).

No dataset? Use the **synthetic demo data** (same schema, simulated churn, *not real*):

```bash
python -m churn.train --use-sample
```
