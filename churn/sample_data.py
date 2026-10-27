"""Generate a SYNTHETIC dataset with the same schema as the IBM Telco Churn data.

For demos/tests only. Churn is simulated from a hand-written logistic rule, so
metrics on it say nothing about real customers. Run: python -m churn.sample_data
"""
import numpy as np
import pandas as pd

from . import config


def make_sample(n: int = 7043, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yn = lambda p: np.where(rng.random(n) < p, "Yes", "No")
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    tenure = np.clip(rng.gamma(2.0, 18, n).astype(int), 0, 72)
    tenure = np.where(contract == "Two year", np.maximum(tenure, 12), tenure)
    phone = yn(0.9)
    base = np.where(internet == "Fiber optic", 75, np.where(internet == "DSL", 52, 20))
    monthly = np.round(base + rng.normal(0, 8, n) + np.where(phone == "Yes", 5, 0), 2).clip(18.25, 119)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2)
    pay = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"], n, p=[0.34, 0.23, 0.22, 0.21])
    net = internet != "No"
    addon = lambda: np.where(~net, "No internet service", yn(0.4))
    df = pd.DataFrame({
        "customerID": [f"SYN-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": yn(0.48), "Dependents": yn(0.3), "tenure": tenure, "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", yn(0.42)),
        "InternetService": internet, "OnlineSecurity": addon(), "OnlineBackup": addon(),
        "DeviceProtection": addon(), "TechSupport": addon(), "StreamingTV": addon(),
        "StreamingMovies": addon(), "Contract": contract, "PaperlessBilling": yn(0.59),
        "PaymentMethod": pay, "MonthlyCharges": monthly, "TotalCharges": total,
    })
    z = (-0.6
         + 1.3 * (contract == "Month-to-month") - 1.0 * (contract == "Two year")
         - 0.035 * tenure + 0.6 * (internet == "Fiber optic")
         + 0.5 * (pay == "Electronic check") + 0.012 * (monthly - 65)
         - 0.5 * (df["TechSupport"] == "Yes") + 0.3 * (df["SeniorCitizen"] == 1)
         + rng.normal(0, 0.6, n))
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")
    # mimic the real file: blank TotalCharges for brand-new customers
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    return df


def write_sample(path=config.SAMPLE_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    make_sample().to_csv(path, index=False)


if __name__ == "__main__":
    write_sample()
    print(f"Wrote {config.SAMPLE_DATA_PATH}")
