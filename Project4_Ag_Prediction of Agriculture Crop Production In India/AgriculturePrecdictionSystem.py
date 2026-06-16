# ==========================================================
# AGRICULTURAL ANALYTICS SYSTEM
# Model 1 : Crop Production Prediction
# Model 2 : Crop Yield Prediction
# ==========================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# ==========================================================
# MODEL 1 : CROP PRODUCTION PREDICTION
# ==========================================================

print("\n" + "="*60)
print("MODEL 1 : CROP PRODUCTION PREDICTION")
print("="*60)

# Load Dataset 2
df_prod = pd.read_csv("datafile (2).csv")

# Clean column names
df_prod.columns = df_prod.columns.str.strip()

# Encode Crop
crop_encoder_prod = LabelEncoder()
df_prod["Crop"] = crop_encoder_prod.fit_transform(
    df_prod["Crop"]
)

# Features
X_prod = df_prod.drop(
    [
        "Production 2010-11",
        "Area 2010-11",
        "Yield 2010-11"
    ],
    axis=1
)

# Target
y_prod = df_prod["Production 2010-11"]

# Split
X_train_prod, X_test_prod, y_train_prod, y_test_prod = train_test_split(
    X_prod,
    y_prod,
    test_size=0.2,
    random_state=42
)

# Scaling for Linear Regression
scaler_prod = StandardScaler()

X_train_prod_scaled = scaler_prod.fit_transform(
    X_train_prod
)

X_test_prod_scaled = scaler_prod.transform(
    X_test_prod
)

# ---------------------------
# Linear Regression
# ---------------------------

lr_prod = LinearRegression()

lr_prod.fit(
    X_train_prod_scaled,
    y_train_prod
)

pred_lr_prod = lr_prod.predict(
    X_test_prod_scaled
)

# ---------------------------
# Decision Tree
# ---------------------------

dt_prod = DecisionTreeRegressor(
    random_state=42
)

dt_prod.fit(
    X_train_prod,
    y_train_prod
)

pred_dt_prod = dt_prod.predict(
    X_test_prod
)

# ---------------------------
# Random Forest
# ---------------------------

rf_prod = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_prod.fit(
    X_train_prod,
    y_train_prod
)

pred_rf_prod = rf_prod.predict(
    X_test_prod
)

# ---------------------------
# Evaluation
# ---------------------------

print("\nProduction Prediction Results")

print("\nLinear Regression")
print("R2:", r2_score(y_test_prod, pred_lr_prod))
print("MAE:", mean_absolute_error(y_test_prod, pred_lr_prod))

print("\nDecision Tree")
print("R2:", r2_score(y_test_prod, pred_dt_prod))
print("MAE:", mean_absolute_error(y_test_prod, pred_dt_prod))

print("\nRandom Forest")
print("R2:", r2_score(y_test_prod, pred_rf_prod))
print("MAE:", mean_absolute_error(y_test_prod, pred_rf_prod))

# Save Best Model
joblib.dump(
    rf_prod,
    "crop_production_model.pkl"
)

print("\nProduction Model Saved")


# ==========================================================
# MODEL 2 : CROP YIELD PREDICTION
# ==========================================================

print("\n" + "="*60)
print("MODEL 2 : CROP YIELD PREDICTION")
print("="*60)

# Load Dataset 1
df_yield = pd.read_csv("datafile (1).csv")

# Clean Columns
df_yield.columns = df_yield.columns.str.strip()

# Encode Crop
crop_encoder_yield = LabelEncoder()

df_yield["Crop"] = crop_encoder_yield.fit_transform(
    df_yield["Crop"]
)

# Encode State
state_encoder = LabelEncoder()

df_yield["State"] = state_encoder.fit_transform(
    df_yield["State"]
)

# Target
y_yield = df_yield["Yield (Quintal/ Hectare)"]

# Features
X_yield = df_yield.drop(
    "Yield (Quintal/ Hectare)",
    axis=1
)

# Split
X_train_yield, X_test_yield, y_train_yield, y_test_yield = train_test_split(
    X_yield,
    y_yield,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler_yield = StandardScaler()

X_train_yield_scaled = scaler_yield.fit_transform(
    X_train_yield
)

X_test_yield_scaled = scaler_yield.transform(
    X_test_yield
)

# ---------------------------
# Linear Regression
# ---------------------------

lr_yield = LinearRegression()

lr_yield.fit(
    X_train_yield_scaled,
    y_train_yield
)

pred_lr_yield = lr_yield.predict(
    X_test_yield_scaled
)

# ---------------------------
# Decision Tree
# ---------------------------

dt_yield = DecisionTreeRegressor(
    random_state=42
)

dt_yield.fit(
    X_train_yield,
    y_train_yield
)

pred_dt_yield = dt_yield.predict(
    X_test_yield
)

# ---------------------------
# Random Forest
# ---------------------------

rf_yield = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_yield.fit(
    X_train_yield,
    y_train_yield
)

pred_rf_yield = rf_yield.predict(
    X_test_yield
)

# ---------------------------
# Evaluation
# ---------------------------

print("\nYield Prediction Results")

print("\nLinear Regression")
print("R2:", r2_score(y_test_yield, pred_lr_yield))
print("MAE:", mean_absolute_error(y_test_yield, pred_lr_yield))

print("\nDecision Tree")
print("R2:", r2_score(y_test_yield, pred_dt_yield))
print("MAE:", mean_absolute_error(y_test_yield, pred_dt_yield))

print("\nRandom Forest")
print("R2:", r2_score(y_test_yield, pred_rf_yield))
print("MAE:", mean_absolute_error(y_test_yield, pred_rf_yield))

# Save Best Model
joblib.dump(
    rf_yield,
    "yield_prediction_model.pkl"
)

print("\nYield Model Saved")


# ==========================================================
# FEATURE IMPORTANCE FOR YIELD MODEL
# ==========================================================

importance = pd.Series(
    rf_yield.feature_importances_,
    index=X_yield.columns
)

importance.sort_values().plot(
    kind="barh",
    figsize=(8,5)
)

plt.title("Yield Prediction Feature Importance")
plt.tight_layout()
plt.show()


# ==========================================================
# FEATURE IMPORTANCE FOR PRODUCTION MODEL
# ==========================================================

importance = pd.Series(
    rf_prod.feature_importances_,
    index=X_prod.columns
)

importance.sort_values().plot(
    kind="barh",
    figsize=(8,5)
)

plt.title("Production Prediction Feature Importance")
plt.tight_layout()
plt.show()


print("\n" + "="*60)
print("PROJECT EXECUTION COMPLETED")
print("="*60)

# ==========================================================
# PRODUCTION MODEL TESTING
# ==========================================================

print("\n" + "="*60)
print("PRODUCTION MODEL TESTING")
print("="*60)

sample_prod = X_prod.iloc[[0]]

actual_prod = y_prod.iloc[0]

predicted_prod = rf_prod.predict(sample_prod)[0]

print("\nSample Production Prediction")

print("Actual Production:", actual_prod)
print("Predicted Production:", round(predicted_prod, 2))
print("Error:", round(abs(actual_prod - predicted_prod), 2))


# ==========================================================
# PRODUCTION ACTUAL VS PREDICTED TABLE
# ==========================================================

results_prod = pd.DataFrame({
    "Actual Production": y_test_prod,
    "Predicted Production": pred_rf_prod
})

results_prod["Error"] = abs(
    results_prod["Actual Production"]
    - results_prod["Predicted Production"]
)

print("\nTop 10 Production Predictions")

print(results_prod.head(10))


# ==========================================================
# CUSTOM PRODUCTION PREDICTION
# ==========================================================

custom_prod = pd.DataFrame(
    [X_prod.iloc[5]],
    columns=X_prod.columns
)

custom_prediction_prod = rf_prod.predict(
    custom_prod
)

print("\nCustom Production Prediction")

print(
    "Predicted Production:",
    round(custom_prediction_prod[0], 2)
)


# ==========================================================
# YIELD MODEL TESTING
# ==========================================================

print("\n" + "="*60)
print("YIELD MODEL TESTING")
print("="*60)

sample_yield = X_yield.iloc[[0]]

actual_yield = y_yield.iloc[0]

predicted_yield = rf_yield.predict(
    sample_yield
)[0]

print("\nSample Yield Prediction")

print("Actual Yield:", actual_yield)
print("Predicted Yield:", round(predicted_yield, 2))
print("Error:", round(abs(actual_yield - predicted_yield), 2))


# ==========================================================
# YIELD ACTUAL VS PREDICTED TABLE
# ==========================================================

results_yield = pd.DataFrame({
    "Actual Yield": y_test_yield,
    "Predicted Yield": pred_rf_yield
})

results_yield["Error"] = abs(
    results_yield["Actual Yield"]
    - results_yield["Predicted Yield"]
)

print("\nTop 10 Yield Predictions")

print(results_yield.head(10))


# ==========================================================
# CUSTOM YIELD PREDICTION
# ==========================================================

custom_yield = pd.DataFrame(
    [X_yield.iloc[5]],
    columns=X_yield.columns
)

custom_prediction_yield = rf_yield.predict(
    custom_yield
)

print("\nCustom Yield Prediction")

print(
    "Predicted Yield:",
    round(custom_prediction_yield[0], 2)
)


# ==========================================================
# ACCURACY SUMMARY
# ==========================================================

print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)

production_accuracy = (
    r2_score(y_test_prod, pred_rf_prod)
    * 100
)

yield_accuracy = (
    r2_score(y_test_yield, pred_rf_yield)
    * 100
)

print(
    f"Production Model Accuracy: "
    f"{production_accuracy:.2f}%"
)

print(
    f"Yield Model Accuracy: "
    f"{yield_accuracy:.2f}%"
)

# ==========================================================
# MODULE 3 : EXPLORATORY DATA ANALYSIS & VISUALIZATION
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# LOAD DATASETS
# ==========================================================

df_cost = pd.read_csv("datafile (1).csv")
df_prod = pd.read_csv("datafile (2).csv")
df_season = pd.read_csv("produce.csv")

# Clean column names
df_cost.columns = df_cost.columns.str.strip()
df_prod.columns = df_prod.columns.str.strip()
df_season.columns = df_season.columns.str.strip()

# ==========================================================
# CHART 1 : COST OF CULTIVATION VS YIELD
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    df_cost["Cost of Cultivation (`/Hectare) C2"],
    df_cost["Yield (Quintal/ Hectare)"]
)

plt.xlabel("Cost of Cultivation (C2)")
plt.ylabel("Yield (Quintal/Hectare)")
plt.title("Cost of Cultivation vs Yield")

plt.grid(True)

plt.show()

# ==========================================================
# CHART 2 : TOP STATES BY AVERAGE YIELD
# ==========================================================

state_yield = df_cost.groupby(
    "State"
)["Yield (Quintal/ Hectare)"].mean()

state_yield = state_yield.sort_values(
    ascending=False
)

plt.figure(figsize=(10,6))

state_yield.plot(kind="bar")

plt.title("Average Yield by State")

plt.ylabel("Yield (Quintal/Hectare)")
plt.xlabel("State")

plt.tight_layout()

plt.show()

# ==========================================================
# CHART 3 : TOP CROPS BY YIELD
# ==========================================================

crop_yield = df_cost.groupby(
    "Crop"
)["Yield (Quintal/ Hectare)"].mean()

crop_yield = crop_yield.sort_values(
    ascending=False
)

plt.figure(figsize=(10,6))

crop_yield.plot(kind="bar")

plt.title("Average Yield by Crop")

plt.ylabel("Yield")

plt.xlabel("Crop")

plt.tight_layout()

plt.show()

# ==========================================================
# PRODUCTION TREND
# ==========================================================

production_cols = [col for col in df_prod.columns if "Production" in col]

years = []
production_means = []

for col in production_cols:

    df_prod[col] = pd.to_numeric(
        df_prod[col],
        errors='coerce'
    )

    production_means.append(
        df_prod[col].mean()
    )

    years.append(
        col.replace("Production ", "")
    )

print(production_means)

plt.figure(figsize=(10,6))

plt.plot(
    years,
    production_means,
    marker='o'
)

plt.title("Average Crop Production Trend")
plt.xlabel("Year")
plt.ylabel("Average Production")
plt.grid(True)

plt.show()

# ==========================================================
# CULTIVATED AREA TREND
# ==========================================================

area_cols = [col for col in df_prod.columns if "Area" in col]

years_area = []
area_means = []

for col in area_cols:

    df_prod[col] = pd.to_numeric(
        df_prod[col],
        errors='coerce'
    )

    area_means.append(
        df_prod[col].mean()
    )

    years_area.append(
        col.replace("Area ", "")
    )

print(area_means)

plt.figure(figsize=(10,6))

plt.plot(
    years_area,
    area_means,
    marker='o'
)

plt.title("Average Cultivated Area Trend")
plt.xlabel("Year")
plt.ylabel("Average Area")
plt.grid(True)

plt.show()

# ==========================================================
# YIELD TREND
# ==========================================================

yield_cols = [col for col in df_prod.columns if "Yield" in col]

years_yield = []
yield_means = []

for col in yield_cols:

    df_prod[col] = pd.to_numeric(
        df_prod[col],
        errors='coerce'
    )

    yield_means.append(
        df_prod[col].mean()
    )

    years_yield.append(
        col.replace("Yield ", "")
    )

print(yield_means)

plt.figure(figsize=(10,6))

plt.plot(
    years_yield,
    yield_means,
    marker='o'
)

plt.title("Average Yield Trend")
plt.xlabel("Year")
plt.ylabel("Average Yield")
plt.grid(True)

plt.show()

# ==========================================================
# CHART 7 : CORRELATION HEATMAP
# ==========================================================

numeric_df = df_cost.select_dtypes(
    include=np.number
)

corr = numeric_df.corr()

plt.figure(figsize=(8,6))

plt.imshow(corr)

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()

plt.show()

# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nDataset 1 Shape:")
print(df_cost.shape)

print("\nDataset 2 Shape:")
print(df_prod.shape)

print("\nTop 5 States by Yield")

print(
    state_yield.head()
)

print("\nTop 5 Crops by Yield")

print(
    crop_yield.head()
)

print("\nEDA COMPLETED SUCCESSFULLY")

