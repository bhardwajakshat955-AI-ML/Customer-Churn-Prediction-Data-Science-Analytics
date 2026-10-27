# Model Card

**Purpose:** estimate the probability that a telecom customer churns, to help prioritise retention outreach. **Intended users:** analysts, CRM/customer-success teams. **Not intended for:** automated, consequential decisions about individuals (pricing, service denial, credit) or any use without human review.

**Data / features:** IBM Telco dataset, 19 features (3 numeric, 16 categorical); see [data_card.md](data_card.md). `customerID` excluded.

**Preprocessing:** numeric → median impute (+ standard scaling for Logistic Regression); categorical → most-frequent impute + one-hot (`handle_unknown="ignore"`). All fitted inside the Pipeline on training data only.

**Candidates:** Logistic Regression (interpretable baseline), Decision Tree, Random Forest, Gradient Boosting. Class imbalance handled with `class_weight="balanced"` where supported.

**Evaluation:** stratified 80/20 split (seed 42); 5-fold stratified CV ROC-AUC on the training split for selection; held-out test metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC and PR curves) at threshold 0.5. **Verified metrics for your data are in `reports/metrics.md` and `models/model_metadata.json` after you run training.**

**Selection rationale:** highest CV ROC-AUC (threshold-independent, robust to imbalance), not accuracy. Recall matters because a missed churner is usually costlier than a wasted offer, but precision determines campaign cost, so choose the operating threshold from your own cost assumptions.

**Limitations / failure modes**
- Probabilities from `class_weight="balanced"` models are *not calibrated* (they overstate churn risk); use them for ranking/tiering, or calibrate before quoting absolute probabilities.
- Trained on one snapshot; performance will drift as products, pricing and customers change.
- Correlation ≠ causation: high-risk customers are not necessarily saved by an offer. Test interventions (A/B).
- `gender` and `SeniorCitizen` are model inputs; review error rates across groups before any deployment and consider removing them.
- Unusual inputs (e.g., values far outside training range) may give unreliable scores.

**Ethics:** comply with privacy law (e.g., GDPR/India DPDP) for real customer data; keep a human in the loop.
