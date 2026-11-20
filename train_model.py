"""Train, compare, and save house-price regression models."""
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "house_prices.csv"
MODEL_FILE = BASE_DIR / "best_model.pkl"
PLOT_FILE = BASE_DIR / "model_comparison.png"


def main():
    # Read the small example dataset into a table.
    data = pd.read_csv(DATA_FILE)
    print("Missing values before cleanup:\n", data.isna().sum())

    # Fill numeric gaps with each column's median and location gaps with its most common value.
    numeric_columns = ["area_sqft", "bedrooms"]
    numeric_imputer = SimpleImputer(strategy="median")
    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
    data["location"] = data["location"].fillna(data["location"].mode()[0])

    # The target price cannot be imputed: remove rows where it is unknown.
    data = data.dropna(subset=["price"])

    # One-hot encoding converts each location into a 0/1 feature column.
    features = pd.get_dummies(data[["area_sqft", "bedrooms", "location"]],
                              columns=["location"], dtype=int)
    target = data["price"]

    # Keep some records aside so model quality is measured on unseen data.
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            "model": model,
            "r2": r2_score(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
        }
        print(f"{name}: R² = {results[name]['r2']:.3f}, "
              f"MAE = ${results[name]['mae']:,.0f}")

    # Prefer higher R²; use lower MAE to break a tie.
    best_name = max(results, key=lambda name: (results[name]["r2"], -results[name]["mae"]))
    best_model = results[best_name]["model"]
    print(f"\nSelected model: {best_name}")

    # Save the model and the feature-column names for the Streamlit app.
    with MODEL_FILE.open("wb") as file:
        pickle.dump({"model": best_model, "feature_columns": list(features.columns),
                     "model_name": best_name}, file)

    # A small bar chart makes the evaluation easy to compare visually.
    scores = pd.DataFrame({name: {"R² score": row["r2"], "MAE ($)": row["mae"]}
                           for name, row in results.items()})
    axes = scores.plot(kind="bar", subplots=True, layout=(1, 2),
                       figsize=(9, 4), legend=False, color=["#3973ac", "#e49b45"])
    plt.suptitle("House price model comparison")
    for axis in axes.flat:
        axis.set_xticklabels(axis.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Saved model to {MODEL_FILE.name} and chart to {PLOT_FILE.name}")


if __name__ == "__main__":
    main()
