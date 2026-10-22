# New York City Taxi Trip Duration Prediction

A deep learning (ANN) regression model that predicts the total duration of NYC taxi trips, built for the Kaggle challenge **[New York City Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview)**.


![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-ANN-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipelines-F7931E?logo=scikitlearn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)

---

## 📌 Table of Contents

1. [Why I Built This](#-why-i-built-this)
2. [Dataset](#-dataset)
3. [Key Findings from EDA](#-key-findings-from-eda)
4. [Key Decisions and Approach](#-key-decisions-and-approach)
5. [Model Architecture](#-model-architecture)
6. [Evaluation Metrics](#-evaluation-metrics)
7. [Results](#-results)
8. [Repository Structure](#-repository-structure)
9. [How to Run](#-how-to-run)
10. [Limitations and Next Steps](#-limitations-and-next-steps)
11. [Training Curves](#-training-curves)

---

## Why I Built This

I wanted to work through a complete deep learning workflow on a **large, messy, real-world dataset**. This project was my way to practice:

- Letting **EDA drive decisions** (outlier handling, which features to keep or drop, which to engineer)
- Writing **custom scikit-learn transformers** and plugging them into a `ColumnTransformer` pipeline
- Building and regularising an **ANN for regression** in Keras (BatchNorm, Dropout, L1/L2, RMSprop)
- Evaluating with the **competition's own metric (RMSLE)** and testing against Kaggle's leaderboard

---

## Dataset

- **Source:** [NYC Taxi Trip Duration, Kaggle competition data](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data)
- **Size:** 1,458,644 training trips, all from **1 Jan 2016 to 30 Jun 2016**
- **Target:** `trip_duration` (seconds)
- **Data quality:** no null values, no duplicate rows

| Column | Description |
|---|---|
| `id` | Unique trip identifier |
| `vendor_id` | Code for the provider associated with the trip record |
| `pickup_datetime` | Date and time when the meter was engaged |
| `dropoff_datetime` | Date and time when the meter was disengaged (**train only**) |
| `passenger_count` | Number of passengers (driver-entered) |
| `pickup_longitude` / `pickup_latitude` | Where the meter was engaged |
| `dropoff_longitude` / `dropoff_latitude` | Where the meter was disengaged |
| `store_and_fwd_flag` | Whether the record was held in vehicle memory before being sent to the vendor |
| `trip_duration` | Trip duration in seconds (**target**) |

---

## Key Findings from EDA

**Target (`trip_duration`)**
- Heavily **right-skewed**: mean ≈ 959 s, median = 662 s, but the max is 3,526,282 s (≈ **979 hours**).

- Using the IQR rule, the upper bound is **2,092 s**. About **5.09% of rows (~74k)** lie above it.

**Passenger count**

- Only 60 trips have 0 passengers and just 5 trips have 7 to 9. These are essentially junk records (the median duration for 0 passengers is 20 s).
- Median duration is nearly flat (≈ 654 to 698 s) for 1 to 6 passengers, so **passenger count barely affects trip duration**.

**Vendor**
- Vendor 2 handles 53.5% of trips, vendor 1 handles 46.5%.
- Average duration is higher for vendor 2 (≈ 1,059 s) than vendor 1 (≈ 845 s), measured on the data *before* outlier removal.

**Time**
- Data covers 6 months of 2016 only
- Average duration by weekday (before outlier removal) ranges from ≈ 898 s (Monday) to ≈ 1,007 s (Thursday). Monday and Sunday are the shortest, Thursday the longest.

**`store_and_fwd_flag`**
- 99.45% of trips are `N`, so this column carries very little signal.

**Coordinates**
- Coordinates themselves do not have good correlation with target.
- We calculated an approx distance based on coordinates which showed a better correlation to the target

**Correlation with trip duration**
- Raw features are almost uncorrelated with the target 
- After feature engineering, **`distance_km` has r ≈ 0.59**, by far the strongest signal. Date features (month, day, hour, minute) all have pearsons coeff < 0.04.

---

## 🧠 Key Decisions and Approach

| Decision | Reason |
|---|---|
| **Dropped `id`** | Unique identifier with no predictive value |
| **Dropped `dropoff_datetime`** | Not available in the test set (and it would leak the target) |
| **Dropped `store_and_fwd_flag`** | 99.45% of values are `N`, so it adds little information |
| **Removed duration outliers with the IQR rule** | Extreme values (up to ~979 h) would dominate an MSE loss. This removed ~5.09% of rows (1,458,644 → 1,384,320) |
| **Engineered `distance_km`** | Raw coordinates correlate weakly with duration, whereas the geodesic distance correlates at ≈ 0.59 |
| **Extracted month, day, hour, minute** from `pickup_datetime` | The raw timestamp string can't be fed to a network |
| **One-hot encoded `vendor_id`** | Vendor is a category |
| **Wrapped everything in a `ColumnTransformer`** | The same transformations apply identically to train, holdout and test data |
| **Clipped negative predictions** | RMSLE is undefined for negative values, so negative outputs are mapped to their absolute value |

### Custom Transformers

Both are scikit-learn compatible (`BaseEstimator` + `TransformerMixin`, with `fit`, `transform` and `get_feature_names_out`), so they slot directly into the pipeline.

- **`DateExtractor`** parses `pickup_datetime` into `pickup_month`, `pickup_day`, `pickup_hour` and `pickup_minute`, then drops the original column.
- **`CoordinatesToDistance`** computes the **geodesic distance in km** (via `geopy`) between pickup and dropoff coordinates and appends it as `distance_km`.

Final feature vector: **12 features** (4 date parts, 4 coordinates, `distance_km`, 2 one-hot vendor columns, `passenger_count`).

---

## Model Architecture

A fully connected network (**3,905 parameters**) with regularisation at every layer to reduce overfitting:

```
Input (12 features)
 ├─ Dense(64, ReLU, L1/L2 = 0.05) → Dropout(0.2) → BatchNormalization
 ├─ Dense(32, ReLU, L1/L2 = 0.05) → Dropout(0.2) → BatchNormalization
 ├─ Dense(16, ReLU, L1/L2 = 0.05) → Dropout(0.1) → BatchNormalization
 └─ Dense(1, linear, L1/L2 = 0.05)  → predicted trip_duration (seconds)
```

**Overfitting controls and training setup**

| Component | Setting |
|---|---|
| **Optimizer** | **RMSprop**, learning rate `0.0001` |
| **Loss** | Mean Squared Error |
| **Batch Normalization** | After every hidden layer, to stabilise and speed up training |
| **Dropout** | 0.2 → 0.2 → 0.1 across the hidden layers |
| **Weight regularisation** | Combined **L1 + L2** (`l1_l2(0.05)`) on all Dense layers |
| **Early stopping** | Monitors `val_loss`, `patience=10`, `min_delta=10000`, `restore_best_weights=True` |
| **Batch size / epochs** | 2048 / up to 300 |
| **Validation** | 20% of the training portion (`validation_split=0.2`), plus a separate 20% holdout test split |
| **Tracked metrics** | R² score and MAE during training |

**Data split:** 1,384,320 rows after outlier removal → 1,107,456 for training / 276,864 for the holdout evaluation (80/20).

---

##  Evaluation Metrics

| Metric | What it tells us |
|---|---|
| **RMSLE**| The **official Kaggle metric**. It measures error on a log scale, so it cares about *relative* error (being 100 s off on a 200 s trip matters more than on a 2,000 s trip), is less dominated by very long trips, and penalises under-prediction more than over-prediction. Lower is better. |
| **RMSE** | Average error in the **same unit as the target (seconds)**. Squaring makes it punish large errors heavily, so it is sensitive to outliers. Lower is better. |
| **R² score** | The **share of variance in trip duration explained by the model**. 1.0 is perfect, 0 means no better than predicting the mean, and negative means worse than the mean. Higher is better. |

---

## Results


| Metric | Score |
|---|---|
| RMSE | **273.22 s** |
| RMSLE | **0.4638** |
| R² | **0.6276** |


**Kaggle leaderboard** (late submission :>, RMSLE, lower is better):

| Submission | Private score | Public score |
|---|---|---|
| Latest | **0.50076** | **0.50122** |

---

## 📈 Training Curves

### Loss vs Validation Loss

<img src="images\losses.png" alt="Training loss and validation loss curves" width="550">

Loss and validation loss fall together and track each other closely, so there is no significant sign of overfitting.

### R² Score vs Validation R² Score

<img src="images\r2_scores.png" alt="Training R2 score and validation R2 score curves" width="550">

R² climbs steadily before plateauing, where early stopping ends training.
