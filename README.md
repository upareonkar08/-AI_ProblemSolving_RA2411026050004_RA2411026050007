# -AI_ProblemSolving_RA2411026050004_RA2411026050007
# Python GUI Applications — TSP Travel Planner & Student Score Predictor

Two standalone Python desktop applications built with **Tkinter**, covering a classic combinatorial optimisation problem and a supervised machine learning pipeline.

---
##  Project Structure

```
.
├── tsp_travel_planner.py       # Travelling Salesman Problem GUI
├── student_score_predictor.py  # ML Regression Score Predictor GUI
└── README.md                   # This file
```

---

##  Application 1 — TSP Tourist Travel Planner

### Problem Description

The **Travelling Salesman Problem (TSP)** is one of the most famous problems in computer science and combinatorial optimisation.

> Given a list of cities and the distances between each pair of cities, find the **shortest possible route** that visits every city exactly once and returns to the starting city.

**Why is it hard?**  
The number of possible routes grows factorially with the number of cities — for *n* cities there are *(n−1)!* unique routes (fixing the start). This makes brute-force impractical for large inputs, but perfectly usable for small tourist itineraries (≤ 10 cities).

| Cities | Permutations |
|--------|-------------|
| 4      | 6           |
| 6      | 120         |
| 8      | 5,040       |
| 10     | 362,880     |

### Algorithm Used

**Brute-Force Permutation Search**

1. Fix the **starting city** to eliminate rotationally equivalent routes.
2. Generate **all permutations** of the remaining cities using Python's `itertools.permutations`.
3. For each permutation, compute the **total round-trip distance** by summing each consecutive leg (including the return to start).
4. Track and return the permutation with the **minimum total distance**.

```
Best = ∞
For each permutation P of (cities − start):
    route = [start] + P + [start]
    dist  = Σ distance(route[i], route[i+1])
    If dist < Best:
        Best  = dist
        Route = route
```

**Time Complexity:** O((n−1)!) — feasible up to ~10 cities.  
**Space Complexity:** O(n) — only the current best route is stored.

Distances are stored in a **bidirectional dictionary** `{(A, B): float}` for O(1) lookup of any city pair in either direction.

### Execution Steps

#### Prerequisites

Python 3.9+ is required. No third-party packages are needed — only the standard library.

```bash
# Verify Python version
python3 --version
```

#### Running the Application

```bash
python3 tsp_travel_planner.py
```

#### Step-by-Step Usage

**Step 1 — Add Cities**
- Type a city name in the input field (e.g. `Paris`) and press **Enter** or click **Add**.
- Add between 2 and 10 cities. City names are auto title-cased.
- Select a city in the list and click **✕ Remove Selected** to delete it.

**Step 2 — Enter Distances**
- The centre panel auto-generates one entry field per unique city pair.
- Type the distance in kilometres for each pair (e.g. `344` for Paris → Amsterdam).
- Distances are symmetric — entering A→B automatically covers B→A.

**Step 3 — Compute Optimal Route**
- Click **🗺 Find Optimal Route** in the right panel.
- The app validates all distances, runs the brute-force solver, and displays the result.

**Step 4 — Reset**
- Click **↺ Reset All** to clear everything and start a new problem.

### Sample Output

**Input cities:** Paris, Amsterdam, Brussels, London, Berlin

**Distances entered (km):**

| From / To   | Amsterdam | Brussels | London | Berlin |
|-------------|-----------|----------|--------|--------|
| Paris       | 430       | 265      | 340    | 878    |
| Amsterdam   | —         | 173      | 356    | 577    |
| Brussels    | —         | —        | 318    | 651    |
| London      | —         | —        | —      | 930    |

**Result displayed in the app:**

```
ROUTE
Paris → Brussels → Amsterdam → Berlin → London → Paris

TOTAL DISTANCE
2,962.0 km
```

**Error handling examples:**
```
⚠ Missing distances for: Paris ↔ Berlin, London ↔ Amsterdam
  → Solver blocked until all fields are filled

⚠ "London" is already in the list.
  → Duplicate city rejected

⚠ Maximum 10 cities supported (brute-force is O(n!)).
  → Hard cap enforced
```

---

##  Application 2 — Student Exam Score Predictor

### Problem Description

Predicting student academic performance is a classic **supervised regression** problem. Given measurable study-related features for a student, the model estimates their likely exam score on a 0–100 scale.

**Input Features (X):**

| Feature            | Description                              | Typical Range |
|--------------------|------------------------------------------|---------------|
| Hours Studied      | Weekly study hours before exam           | 1 – 12 hrs    |
| Attendance %       | Percentage of classes attended           | 40 – 100 %    |
| Previous Score     | Score from the most recent prior exam    | 30 – 95 pts   |
| Assignments Done   | Number of assignments submitted (of 10)  | 0 – 10        |

**Target Variable (y):**

| Variable   | Description             | Range     |
|------------|-------------------------|-----------|
| Exam Score | Final exam result       | 0 – 100   |

### Algorithm Used

**Linear Regression** (`sklearn.linear_model.LinearRegression`)

Linear Regression models the target as a weighted linear combination of the input features:

```
Exam Score = w₁·(Hours) + w₂·(Attendance) + w₃·(Prev Score) + w₄·(Assignments) + b
```

Where *w₁…w₄* are learned weights (coefficients) and *b* is the bias (intercept), determined by minimising the **Mean Squared Error** over the training set using the closed-form **Ordinary Least Squares** solution:

```
w = (XᵀX)⁻¹ Xᵀy
```

**Full ML Pipeline:**

```
Raw Data
   │
   ▼
Handle Missing Values  ← fill NaN with column median; drop residual NaN rows
   │
   ▼
Train / Test Split     ← default 80% train, 20% test (configurable)
   │
   ▼
StandardScaler         ← zero mean, unit variance (fit on train only)
   │
   ▼
LinearRegression.fit() ← learns weights on scaled training data
   │
   ▼
Predict on Test Set    ← clip predictions to [0, 100]
   │
   ▼
Evaluate Metrics       ← R², MAE, RMSE
   │
   ▼
Predict New Students   ← scale input → model.predict() → display
```

**Evaluation Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| R² Score | 1 − SS_res/SS_tot | 1.0 = perfect fit; 0 = predicts mean |
| MAE | mean(\|y − ŷ\|) | Average error in exam points |
| RMSE | √mean((y − ŷ)²) | Penalises large errors more than MAE |

### Execution Steps

#### Prerequisites

```bash
pip install scikit-learn pandas numpy
```

Or with the break-system flag if needed:

```bash
pip install scikit-learn pandas numpy --break-system-packages
```

#### Running the Application

```bash
python3 student_score_predictor.py
```

#### Step-by-Step Usage

The application has **three tabs**:

---

**Tab ① — Dataset**

Choose one of three data sources:

*Option A — Upload CSV*
- Click **Browse** and select a `.csv` file.
- The file must contain columns: `Hours Studied`, `Attendance %`, `Previous Score`, `Assignments Done`, `Exam Score`.
- Missing values are auto-filled with column medians.

*Option B — Synthetic Dataset*
- Set the number of rows (default: 120, minimum: 10).
- Click **Generate** — a realistic dataset is created instantly using a weighted formula with Gaussian noise.

*Option C — Manual Entry*
- Fill in all five fields for one student row.
- Click **+ Add Row** to accumulate rows (minimum 5 required).
- Click **Use Manual Data** to load the accumulated rows.

After loading, the **Data Preview** table on the right populates with up to 200 rows.

---

**Tab ② — Train Model**

- Set the **test split %** (5–50, default 20%).
- Click ** Train Model**.
- The **Metrics panel** updates with R², MAE, RMSE, and individual feature coefficients.

---

**Tab ③ — Predict**

- Enter values for all four features.
- Click ** Predict Score**.
- The app displays the predicted score, a grade label, and appends the prediction to the session history log.

---

*CSV Format Reference:*

```csv
Hours Studied,Attendance %,Previous Score,Assignments Done,Exam Score
7.5,88.0,74.0,9,81.3
3.2,61.0,55.0,4,52.7
11.0,97.0,91.0,10,95.1
```

### Sample Output

#### Model Metrics (trained on 120-row synthetic dataset)

```
┌─────────────┬────────────┬─────────────────────────────────┐
│  R² Score   │    MAE     │             RMSE                │
│   0.9293    │   2.93 pts │            3.94 pts             │
└─────────────┴────────────┴─────────────────────────────────┘

Training rows: 96    Test rows: 24

Feature Coefficients (scaled):
  Hours Studied      +14.2341
  Attendance %       + 2.8812
  Previous Score     +13.0457
  Assignments Done   + 7.6123
```

#### Single Student Prediction

**Input:**
```
Hours Studied    : 8.0
Attendance %     : 90.0
Previous Score   : 76.0
Assignments Done : 8
```

**Output:**
```
Predicted Exam Score
        83.6

  Good

Prediction History:
Score=83.6  |  Hour=8.0  Atte=90.0  Prev=76.0  Assi=8.0
```

#### Grade Bands

```
90 – 100  →   Excellent
75 –  89  →   Good
60 –  74  →   Average
40 –  59  →   Below Average
 0 –  39  →   At Risk
```

#### Error Handling Examples

```
⚠ Invalid value for "Attendance %": "abc"
  → Entry rejected; prediction blocked

⚠ Not Enough Data — Enter at least 5 rows (currently 3).
  → Manual data load blocked

⚠ Train the model first (Tab ②).
  → Prediction blocked until model is trained

⚠ Missing columns: Assignments Done, Exam Score
  → CSV rejected with specific column list
```

---

##  Dependencies Summary

| Package        | Used In                    | Install             |
|----------------|----------------------------|---------------------|
| `tkinter`      | Both apps (GUI)            | Built into Python   |
| `itertools`    | TSP (permutations)         | Built into Python   |
| `scikit-learn` | Score Predictor (ML)       | `pip install scikit-learn` |
| `pandas`       | Score Predictor (data)     | `pip install pandas` |
| `numpy`        | Score Predictor (numerics) | `pip install numpy` |

---

##  Design Decisions

### TSP Planner
- **Bidirectional dict** `{(A,B): d, (B,A): d}` gives O(1) distance lookup regardless of argument order.
- **Start city is pinned** to index 0, reducing search space from n! to (n−1)! without affecting correctness.
- **Live distance sync** — entries update the dict on every keystroke via `StringVar.trace_add`.

### Score Predictor
- **StandardScaler** is fit only on training data and then applied to test/prediction data, preventing data leakage.
- **Predictions are clipped** to [0, 100] because linear models can extrapolate beyond physical bounds.
- **Modular `StudentModel` class** separates all ML logic from the GUI — the regression algorithm can be swapped in one line.
