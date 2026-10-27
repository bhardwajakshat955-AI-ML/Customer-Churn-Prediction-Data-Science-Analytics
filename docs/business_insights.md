# Business Insights & Use Cases

> Findings below are *what to look for*; the actual numbers come from your run of `python -m churn.eda` (`reports/eda_summary.md`, `reports/figures/`). Update this file with the verified figures once generated.

## Questions the EDA answers
1. Does churn differ by **contract type** (month-to-month vs. 1/2-year)?
2. How does churn change with **tenure** (early-life churn)?
3. Are customers with higher **monthly charges**, particular **internet services**, or **payment methods** more likely to leave?
4. Do add-ons such as **TechSupport** or **OnlineSecurity** relate to lower churn?

## Use cases
| Use case | How the model helps |
|---|---|
| Find at-risk customers | Score the base weekly; list High-tier customers |
| Prioritise retention campaigns | Rank by probability × monthly revenue |
| Support CRM / customer success | Surface risk tier next to the account |
| Segment-level churn analysis | Compare churn across contract/service segments |
| Retention planning | Tier counts and assumption-based revenue exposure (`reports/risk_segments.csv`) |

## Risk tiers (configurable in `churn/config.py`)
Low < 0.30 ≤ Medium < 0.60 ≤ High. Suggested treatment: High → personal outreach/offer test; Medium → automated nudges; Low → standard service.

## Measuring value honestly
Predicted churn is not saved revenue. Run a controlled test (offer vs. holdout among High-tier customers), track retention rate, offer cost and incremental revenue. `assumed_revenue_exposure` in the risk report = monthly revenue × `ASSUMED_RETENTION_MONTHS` — an illustrative, configurable assumption.
