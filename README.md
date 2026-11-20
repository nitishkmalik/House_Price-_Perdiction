# House Price Prediction

A small, runnable regression project that estimates a house price from its floor area, bedroom count, and location. It compares Linear Regression with Random Forest, then serves the better-scoring model in a simple Streamlit form. This is for learning, not for real estate decisions.

## Folder structure

```text
house-price-prediction/
├── app.py                 # Streamlit form and prediction display
├── train_model.py         # Cleaning, encoding, training, comparison, chart, model save
├── house_prices.csv       # Small example dataset (includes a few missing fields)
├── requirements.txt       # Python packages needed
├── README.md              # Instructions and interview preparation
├── best_model.pkl         # Created by train_model.py
└── model_comparison.png   # Created by train_model.py
```

The last two output files are generated locally and are intentionally not included in the starter folder.

## Run in VS Code

1. Install Python 3.10 or newer and the VS Code Python extension.
2. Open this folder in VS Code, then open **Terminal → New Terminal**.
3. Create and activate a virtual environment:
   - Windows PowerShell: `python -m venv .venv` then `.venv\Scripts\Activate.ps1`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install packages: `python -m pip install -r requirements.txt`
5. Train and compare the models: `python train_model.py`
6. Start the interface: `python -m streamlit run app.py`
7. Streamlit opens the app in a browser. Select an area, bedroom count, and location, then click **Predict price**.

## How the project works

1. `pandas.read_csv` loads the CSV into a DataFrame (a table).
2. `data.isna().sum()` counts missing values in each column.
3. `SimpleImputer(strategy="median")` fills missing numeric inputs with the median; the most common location fills missing locations. Rows without the answer (`price`) are dropped because there is no correct target value to learn from.
4. `pd.get_dummies` one-hot encodes location. For example, `location_Suburb` is 1 for a Suburb row and 0 otherwise. This avoids pretending location names have a numeric order.
5. `train_test_split` holds out 25% of rows for evaluation. `random_state=42` makes the split repeatable.
6. Both models are fitted using the training rows. R² is better when higher (1 is a perfect fit; it can be negative). MAE is better when lower and is reported in dollars.
7. The model with the higher R² is selected; if scores tie, lower MAE wins. The fitted model and exact feature-column order are saved in `best_model.pkl`.
8. Streamlit creates a form and assembles the selected location into the same one-hot feature format before asking the saved model for a prediction.

## File guide and important lines

- **`house_prices.csv`** — one house per row. `area_sqft`, `bedrooms`, and `location` are the inputs; `price` is the value to predict. Four records intentionally have missing fields so the cleanup steps can be demonstrated. Values are illustrative sample data, not a real market feed.
- **`train_model.py`** — `pd.read_csv(DATA_FILE)` loads data; `SimpleImputer` and `fillna` handle missing inputs; `pd.get_dummies(...)` encodes locations; `train_test_split(...)` creates a fair test set; `model.fit(...)` trains; `r2_score` and `mean_absolute_error` compare; `pickle.dump(...)` saves the winning estimator and feature names. The final section saves a Matplotlib comparison chart.
- **`app.py`** — loads the saved model; the Streamlit widgets collect user inputs; a dictionary creates a row of zeros, then the chosen `location_*` column becomes 1; `model.predict(...)` returns the estimated price. Train once before launching the app.
- **`requirements.txt`** — lists the libraries so pip can install the same dependencies on another machine.

## Example use and output

Example input: **1,500 sq ft**, **3 bedrooms**, **Suburb**.

Example output: **Estimated price: about $320,000–$380,000**. Exact output depends on the test split and which model wins on this small dataset; it is a model estimate, not a guaranteed value.

After running training, the terminal prints both model metrics and the selected model, for example:

```text
Linear Regression: R² = 0.8xx, MAE = $xx,xxx
Random Forest: R² = 0.7xx, MAE = $xx,xxx
Selected model: Linear Regression
```

Actual scores vary with package versions and the small hold-out sample. Check the printed metrics rather than copying the illustrative numbers above.

lf; do not load pickle files from untrusted sources.
