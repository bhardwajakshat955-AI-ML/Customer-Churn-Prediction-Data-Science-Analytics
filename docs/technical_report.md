# Technical Report

1. **Problem:** binary classification of churn (≈ minority class), evaluated mainly by ROC-AUC, recall/precision/F1.
2. **Cleaning:** see [data_card.md](data_card.md). Cleaning is stateless; all learned statistics live in the Pipeline.
3. **Features:** raw Telco columns, one-hot encoded; no manual feature engineering was added (kept simple and leakage-safe). Possible extensions: tenure bands, service-count, charge-per-month ratio.
4. **Models:** 4 classifiers compared with identical split/CV (seed 42). Hyperparameters are modest defaults, not exhaustively tuned.
5. **Selection:** max mean 5-fold CV ROC-AUC on training split; test set reported once.
6. **Reproducibility:** fixed seeds, pinned minimum versions in `requirements.txt`, `models/model_metadata.json` records data file, versions, split sizes and metrics.
7. **Deployment:** FastAPI loads the saved pipeline; `/predict` validates input with Pydantic and returns probability, binary prediction and risk tier.
8. **Next steps:** hyperparameter search, probability calibration, threshold optimisation from a cost matrix, SHAP explanations, drift monitoring.
