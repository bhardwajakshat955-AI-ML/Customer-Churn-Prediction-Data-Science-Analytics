# Data Card

**Dataset:** IBM Telco Customer Churn (sample dataset published by IBM; distributed via Kaggle and other mirrors). Fictional telecom customers; treat as a public **sample/synthetic-style** dataset, not real customer records. Confirm license terms where you download it.

**Unit of analysis:** one customer. **Target:** `Churn` (Yes → 1 = left within the last month, No → 0).

| Column | Type | Meaning |
|---|---|---|
| customerID | id | Unique ID (dropped, never a feature) |
| gender, SeniorCitizen, Partner, Dependents | categorical | Demographics (SeniorCitizen recoded 0/1 → No/Yes) |
| tenure | numeric | Months as customer |
| PhoneService, MultipleLines | categorical | Phone services |
| InternetService | categorical | DSL / Fiber optic / No |
| OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | categorical | Add-on services (Yes / No / No internet service) |
| Contract | categorical | Month-to-month / One year / Two year |
| PaperlessBilling, PaymentMethod | categorical | Billing |
| MonthlyCharges, TotalCharges | numeric | Current monthly bill; lifetime billed |
| Churn | target | Left the company |

**Known data-quality points handled in code**
- `TotalCharges` contains blank strings for customers with tenure 0 → converted to NaN, median-imputed inside the pipeline (fit on training data only).
- Duplicate `customerID`s are dropped if present.
- Whitespace in text columns is stripped.

**Exact counts** (rows, churn rate, missing values, per-segment churn) are computed by `python -m churn.eda` and written to `reports/eda_summary.md`; they are deliberately not hard-coded here.

**Limitations / bias:** single snapshot (no time dimension), no usage or support-ticket history, class imbalance (minority = churners), demographic fields (gender, senior status) present — see the model card for fairness cautions.
