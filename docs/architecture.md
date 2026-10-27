# Architecture

```mermaid
flowchart LR
    A[Raw CSV<br/>data/raw] --> B[churn.data<br/>validate + clean]
    B --> C[Stratified train/test split]
    C --> D[sklearn Pipeline<br/>impute, scale/encode, model]
    D --> E[5-fold CV on train only<br/>model selection]
    E --> F[Final test evaluation<br/>metrics, plots, risk tiers]
    D --> G[(models/churn_model.joblib)]
    G --> H[churn.predict]
    H --> I[FastAPI /predict]
    B --> J[churn.eda<br/>charts + summary]
```

| Module | Responsibility |
|---|---|
| `churn/config.py` | Paths, feature lists, seed, risk thresholds |
| `churn/data.py` | Load, validate schema, clean (stateless → no leakage) |
| `churn/preprocessing.py` | ColumnTransformer (imputer, scaler, one-hot) |
| `churn/train.py` | Compare 4 models, select, evaluate, save artifacts/reports |
| `churn/predict.py` | Load pipeline, enforce column order, score |
| `churn/api.py` | FastAPI app with Pydantic validation |
| `churn/risk.py` | High/Medium/Low segmentation, assumption-based revenue exposure |
| `churn/eda.py` | Charts and statistics computed from the data |

**Leakage controls:** the split happens before any fitting; imputation, scaling and encoding live inside the `Pipeline`, so they are refit on each CV training fold; model selection uses CV on the training split only; the test set is used once for final reporting. `customerID` is dropped from features.

The saved artifact is the *whole pipeline*, so training and serving use identical transformations and column order.
