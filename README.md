# Beginner House Price Prediction

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

## Placement interview: 15 questions and answers

1. **What is supervised learning?** Learning from examples that include both input features and a known answer. Here, house details are features and price is the answer.
2. **Is this classification or regression?** Regression, because price is a continuous numeric value.
3. **What are the features and target?** Area, bedrooms, and location are features; price is the target.
4. **Why check missing values?** Most estimators cannot safely learn from empty values, and missing data can distort results if ignored.
5. **Why use the median for numeric values?** It is simple and is less affected by unusually high or low values than the mean.
6. **Why fill missing locations with the mode?** Location is categorical, so its most frequent category is a straightforward replacement.
7. **Why drop a row with missing price?** Price is the answer the model must learn. Filling it with a guess would teach the model fabricated labels.
8. **What is One-Hot Encoding?** It creates a separate 0/1 column for each category, allowing a model to use labels such as Downtown without inventing a numeric ranking.
9. **What does `train_test_split` do?** It separates data for fitting from unseen rows used to evaluate how well the model generalizes.
10. **Why set `random_state`?** It makes the random split reproducible so results are easier to compare during development.
11. **How does Linear Regression work?** It learns weights for input features and combines them to estimate a numeric target.
12. **How does Random Forest Regressor work?** It combines predictions from many decision trees trained on varied samples/features to produce a regression estimate.
13. **What does R² mean?** It measures how much target variation is explained relative to a simple average-based baseline. Higher is generally better; a negative score can occur on poor predictions.
14. **What is MAE?** Mean Absolute Error is the average absolute gap between predicted and actual prices, in the same units as price. Lower is better.
15. **Why choose one model using a held-out test set, and what are the limitations?** The example uses the held-out scores to demonstrate selection, but this tiny synthetic dataset makes scores unstable and can favor a model by chance. A real project should use more representative data and ideally a separate validation set or cross-validation, then keep a final test set untouched.

## Notes

This is intentionally small and beginner-friendly. It does not use an API, database, deep learning, or production data. For a portfolio, explain that location and market coverage are limited, small-data metrics are noisy, and a prediction is not an appraisal. The pickle file is loaded only from this project after you create it yourself; do not load pickle files from untrusted sources.
