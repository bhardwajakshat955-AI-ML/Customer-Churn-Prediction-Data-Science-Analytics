# Setup & Troubleshooting

## Setup
```bash
git clone https://github.com/bhardwajakshat955-AI-ML/Customer-Churn-Prediction-Data-Science-Analytics.git
cd Customer-Churn-Prediction-Data-Science-Analytics
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

## Common errors
| Problem | Fix |
|---|---|
| `FileNotFoundError: Dataset not found` | Put the CSV at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (see `data/README.md`) or add `--use-sample` |
| `ModuleNotFoundError: churn` | Run commands from the repository root |
| API returns 503 "Model artifact not found" | Run `python -m churn.train` first |
| `ValueError: Dataset is missing required columns` | You are using a different/modified CSV; columns must match `data/README.md` |
| `InconsistentVersionWarning` on load | Retrain with your installed scikit-learn: `python -m churn.train` |
| Windows: `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `422 Unprocessable Entity` from API | Input value not in allowed set; see `/docs` for valid values |
