# Learning Notes: Telco Churn Case Study (Q&A)

Notes from working through `pre-reporting.ipynb`. They explain *why* each step works the way it does. For the syntax, see `CHEATSHEET.md`.

## Contents

1. [Debugging: `could not convert string to float: ' '`](#1-debugging-could-not-convert-string-to-float--)
2. [`pd.to_numeric(errors='coerce')`](#2-pdto_numericerrorscoerce)
3. [Rows, Train/Test Split, `stratify` and `random_state`](#3-rows-traintest-split-stratify-and-random_state)
4. [Preprocessing: Outliers, Skewness, Encoding, Scaling](#4-preprocessing-outliers-skewness-encoding-scaling)
5. [Why KNN Cares About Scale, Skew and Outliers (and Trees Don't)](#5-why-knn-cares-about-scale-skew-and-outliers-and-trees-dont)
6. [How a Decision Tree Is Built](#6-how-a-decision-tree-is-built)
7. [Random Forest](#7-random-forest)
8. [Hyperparameters Explained](#8-hyperparameters-explained)
9. [`fit`, `transform`, `fit_transform` and the Pipeline](#9-fit-transform-fit_transform-and-the-pipeline)
10. [Cross Validation with `StratifiedKFold`](#10-cross-validation-with-stratifiedkfold)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Reading `classification_report`](#12-reading-classification_report)
13. [Multi-class (3+ classes)](#13-multi-class-3-classes)

---

## 1. Debugging: `could not convert string to float: ' '`

```python
X = df[feature]                                                          # 1. X is a COPY
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # 2. fixes df only
df['TotalCharges'] = df['TotalCharges'].fillna(0)                        # 3. fixes df only
pipe.fit(X_train, y_train)   # ValueError: X still has ' ' strings
```

- `X = df[feature]` makes a **copy**. Changes made to `df` afterwards don't reach `X`.
- The blank `' '` values were still in `X`, so `StandardScaler` couldn't turn them into floats.
- `X.isnull().sum()` showed 0, because `' '` is a string, not NaN.

**Fix:** clean `df` first, then create `X`. You can also apply the same fix to `X` directly.

> 💡 Always check with **Restart Kernel → Run All**. Cells run in a different order can show results that can't be reproduced.

---

## 2. `pd.to_numeric(errors='coerce')`

| Option | What happens with `' '` |
|---|---|
| `errors='raise'` (default) | error |
| `errors='coerce'` | becomes **NaN**; everything else is converted normally |

"Coerce" = force. The column is forced to be numeric, and any value that can't be converted becomes NaN.

`coerce` only **creates** NaN. You still need to fill it:

```python
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)
```

Filling with 0 is correct here: all 11 rows have `tenure == 0` (new customers who haven't been charged yet).

---

## 3. Rows, Train/Test Split, `stratify` and `random_state`

### Row

**One row = one customer.** Columns are that customer's information. The number on the left is the **index** (the row ID). Splitting the data means dividing the **customers** into two groups. Each customer goes to train or test as a whole, with all their columns and their `y` label.

### `stratify=y`

Keeps the **class ratio** the same in train and test.

- Full data: ~26.5% churn → train ~26.5%, test ~26.5% (374 / 1409).
- It needs no training or "understanding". It only looks at the labels in `y`:
  1. group the rows by label (all 0s, all 1s)
  2. shuffle each group
  3. take 20% of each group for test, the rest goes to train
- **Which** rows are picked is random; only **how many from each class** is controlled.
- Most important for **imbalanced** targets (churn, fraud, disease). It keeps the ratio the same but does **not** fix the imbalance (use `class_weight` for that).

### `random_state=42`

A **seed**, the starting point for the pseudo-random number generator.

```
same starting point → same sequence of numbers → same shuffle → same split
```

- Same seed → the same split every run, on any computer.
- Only if **everything else stays the same**: the same data in the same order, the same `test_size` / `stratify`, and (usually) the same scikit-learn version.
- 42 has no special meaning (it's a *Hitchhiker's Guide* joke). Any fixed number works.

**Each `random_state` controls a different random process.** Setting one does not fix the others:

| Where | What it controls |
|---|---|
| `train_test_split` | which customers go to train or test |
| `StratifiedKFold(shuffle=True)` | how the CV folds are shuffled |
| `DecisionTreeClassifier` | breaking ties between equally good questions |
| `RandomForestClassifier` | which rows and columns each tree gets |
| KNN | none, it has no randomness |

If you fix the split but not the forest, you get the same data but different trees, so F1 changes slightly each run.

---

## 4. Preprocessing: Outliers, Skewness, Encoding, Scaling

### What the notebook does

| Step | In notebook? | How |
|---|---|---|
| Missing values | ✅ | `to_numeric(errors='coerce')` + `fillna(0)` |
| Outliers | ❌ (none exist) | — |
| Encoding | ✅ | `y.map({'Yes':1,'No':0})`; `OneHotEncoder` on categories |
| Scaling | ✅ | `StandardScaler` on numeric columns |

Encoding and scaling happen **inside the Pipeline**, which is why `X_train` still looks like raw data when you display it.

### Outliers: check, but don't remove here

IQR method (`CHEATSHEET.md` section 8):

```python
for c in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    q1, q3 = X[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((X[c] < q1 - 1.5*iqr) | (X[c] > q3 + 1.5*iqr)).sum()
    print(c, 'outliers:', n)
```

| Column | Min | Max | Skewness | IQR outliers |
|---|---|---|---|---|
| tenure | 0 | 72 | 0.24 | 0 |
| MonthlyCharges | 18.25 | 118.75 | -0.22 | 0 |
| TotalCharges | 0 | 8684.8 | **0.96** | 0 |

- **IQR is the right method here**: it uses quartiles, so it works on skewed data. The z-score assumes a bell-shaped distribution.
- **No outliers found → nothing removed.**
- Large values are **real** (a long-term customer on an expensive plan), not errors. Remove outliers only when they are **mistakes** (e.g. `tenure = 999` or negative charges).
- If you do remove outliers, decide using **`X_train` only**. Never touch the test set.

### Skewness ≠ outliers

| | Outlier | Skewness |
|---|---|---|
| About | **single values** far from the rest | the **shape of the whole column** (a long tail) |
| Check with | IQR / boxplot (dots past the whiskers) | `.skew()` / histogram |
| Fix | remove or cap (if it's an error) | transform (`np.log1p`, `np.sqrt`) |

```
A) 5, 5, 6, 6, 7, 7, 8, 100      → one outlier, the rest is symmetric
B) 1, 1, 2, 2, 3, 4, 6, 9, 14    → no outlier, but skewed
```

Skewed data can have no outliers, like `TotalCharges`. Outliers can create skew.

### Encoding: why one-hot?

| Encoder | Use for |
|---|---|
| Label / `map` | 2 categories (used for `y`) |
| Ordinal | categories with a **real order** (Fair < Good < Excellent) |
| **One-hot** | categories with **no order** |

Ordinal encoding on unordered categories creates **fake math** (e.g. "bank transfer is twice as far from electronic check as mailed check"). One-hot puts every category at the same distance from the others.

| Column | Has order? | Choice |
|---|---|---|
| gender, Partner, Dependents | No | one-hot (or a 0/1 map) |
| **Contract** | **Yes** (month < 1 year < 2 years) | one-hot **or** `OrdinalEncoder` → idea for Scenario 02 |

```python
OrdinalEncoder(categories=[['Month-to-month', 'One year', 'Two year']])
OneHotEncoder(handle_unknown='ignore', drop='if_binary')   # 1 column for yes/no
```

- `handle_unknown='ignore'`: a category not seen during training becomes all 0s instead of causing an error.
- `sparse_output=False`: stores the result as a normal (dense) array instead of storing only the 1s (sparse). Sparse saves memory when there are many categories. Dense works with **every** model (e.g. `GaussianNB` needs dense). The values are the same either way.
- `SeniorCitizen` is already 0/1, so scaling it is unnecessary (but harmless).

---

## 5. Why KNN Cares About Scale, Skew and Outliers (and Trees Don't)

**KNN uses distances**, so the actual size of the numbers matters.
- Skew: typical customers get squeezed together and the tail gets stretched out.
- Outliers: they pull the scaler's mean and std, and an outlier's "nearest" neighbors aren't really near.

**Trees use thresholds** ("tenure < 12?"), so only the **order** of values matters.

```
TotalCharges:  100   400   900   1500   8000
log1p:         4.6   6.0   6.8   7.3    9.0     ← same order → same split
```

An outlier of 1,000,000 just goes to the ">" side, exactly like 8,000 would.

| | KNN | Decision Tree / Random Forest |
|---|---|---|
| Uses | distance | thresholds (order) |
| Needs scaling? | **Yes** | No |
| Affected by skew? | **Yes** | No |
| Affected by outliers? | **Yes** | Barely |

---

## 6. How a Decision Tree Is Built

### Parts

```
                    [ Contract = Month-to-month? ]          ← root
                     /                         \
      [ tenure < 12? ]                     [ MonthlyCharges > 90? ]   ← nodes
        /         \                           /            \
   🍂 CHURN     🍂 STAY                  🍂 STAY         🍂 STAY      ← leaves
```

- **Root**: the first question. **Node**: a question. **Branch**: the yes/no path. **Leaf**: the end of a path (an answer, no more questions).
- Each question splits the data with a straight line, so the tree divides the space into **boxes**.

### Two phases

1. **`fit`**: look at **all training rows** → build the tree.
2. **`predict`**: the tree is finished; a new customer follows the questions down to a leaf.

### Building (greedy, one step at a time)

1. Start with all customers in one group.
2. Try **every column × every cut point** and pick the question that best separates churn from stay.
3. Repeat **inside each new group**.
4. Stop when a group is pure or a limit is reached (`max_depth`, `min_samples_leaf`). That group becomes a **leaf**.
5. **Leaf prediction = the majority class** of the training customers in that leaf (e.g. 28 churn vs 12 stay → CHURN, 70%).

- It does **not** try every combination of questions (that would be billions of trees). It picks the best question **at each step** and never goes back. This is fast but not guaranteed to find the best possible tree.
- "Trying combinations" happens in **GridSearchCV**, which tries combinations of **settings** and builds a whole tree for each.

```
tiny example: "tenure < 10?" → Yes: 4 churn, 0 stay | No: 0 churn, 6 stay → perfect split
```

### Depth ≠ number of columns

- Depth = **questions in a row**. A tree can reuse a column (`tenure < 24` → `tenure < 6` → `tenure < 2`).
- Useless columns (e.g. `gender`) are simply never picked.
- Deeper trees **memorize** individual customers (**overfitting**): ~100% on train, worse on new data.
- In the notebook: Decision Tree best `max_depth=3` (with 8 features); `None` (grow fully) lost for both models.

### Multi-class trees

The questions are always yes/no, but leaves can hold **any number of classes**. The leaf predicts the majority among 3 classes, for example.

---

## 7. Random Forest

Many different trees that **vote**.

Each tree is different because of two random choices:
1. **Different customers**: each tree gets a random sample of rows (some picked twice, some left out).
2. **Different questions**: at each node, only a random subset of columns is considered.

### Predicting

The new customer goes down **every tree** (200 trees → 200 leaves):

```
Tree 1: 80% churn   Tree 2: 75%   Tree 3: 40%   ...   Tree 200: 70%
average = 68% churn → CHURN
```

scikit-learn **averages the leaf probabilities** (soft voting) instead of counting votes, so a confident tree counts more.

```python
final_model.predict_proba(X_test.iloc[[0]])   # [[0.32, 0.68]] → 32% stay, 68% churn
```

Trees make different mistakes, and averaging cancels many of them out. That's why the forest can go deeper (best `max_depth=10`) than a single tree (3), and why it scored higher (F1 0.622 vs 0.606).

| | Decision Tree | Random Forest |
|---|---|---|
| Goes through | 1 tree | all trees |
| Lands in | 1 leaf | 1 leaf per tree |
| Answer | that leaf's majority | average of all leaves |

---

## 8. Hyperparameters Explained

### `max_depth`

The maximum number of questions from root to leaf. It limits overfitting.

### `min_samples_leaf`

The **minimum number of training customers in each leaf**. A split is only allowed if **both** new groups have at least that many.

```
group of 50, "tenure < 3?" → Yes: 8 | No: 42    with min_samples_leaf=20 → split NOT allowed
```

| Value | Effect |
|---|---|
| 1 | tiny leaves are possible → memorizing (overfitting) |
| 5 | middle ground |
| 20 | big general leaves → may miss real patterns (underfitting) |

The Decision Tree's best was 1, but `max_depth=3` already keeps the leaves large (~700 customers). The Random Forest's best was 5 with depth 10.

### `class_weight='balanced'`

Gives the smaller class a **bigger weight**:

```
weight = total ÷ (n_classes × class count)
STAY:  5634 ÷ (2 × 4139) = 0.68
CHURN: 5634 ÷ (2 × 1495) = 1.88   → each churner counts ~2.8×
```

```
Leaf 30 stay / 20 churn:  unweighted → STAY;  weighted 20.4 vs 37.6 → CHURN
```

| Random Forest | without balanced | with balanced |
|---|---|---|
| Recall (churn) | 0.44 | **0.80** |
| Precision (churn) | 0.57 | 0.51 |
| F1 (churn) | 0.49 | **0.62** |

KNN has no `class_weight`, which is one reason its recall is low. To balance data for KNN, use SMOTE or undersampling.

### KNN `weights`

**Both options use distance to find the neighbors.** The setting only changes **how much each neighbor's vote counts**:

| | Vote |
|---|---|
| `'uniform'` | every neighbor = 1 vote |
| `'distance'` | vote = 1 ÷ distance (closer neighbors count more) |

```
k=5: A(0.5,CHURN) B(1,STAY) C(2,STAY) D(4,STAY) E(5,CHURN)
uniform:  2 CHURN vs 3 STAY → STAY
distance: 2.0+0.2=2.2 CHURN vs 1+0.5+0.25=1.75 STAY → CHURN
```

These are two different models, and neither is "the" answer. Only the real outcome tells you which was right for this customer. You choose the setting that is right **most often**, using GridSearch (the best here was `k=15, uniform`). In KNN they're called **neighbors**; **nodes** belong to trees.

---

## 9. `fit`, `transform`, `fit_transform` and the Pipeline

| Object | Method | Does |
|---|---|---|
| Transformer (scaler, encoder) | `fit` | **learns** (mean/std, categories) |
| | `transform` | **applies** what it learned |
| | `fit_transform` | both |
| Model | `fit(X, y)` | learns the patterns |
| | `predict(X)` | answers |

Manual version:

```python
X_train_s = scaler.fit_transform(X_train)   # learn from TRAIN
X_test_s  = scaler.transform(X_test)        # never fit on test (data leakage)
```

The Pipeline does this for you:

```
pipe.fit(X_train, y_train)  = preprocessing.fit_transform + model.fit
pipe.predict(X_test)        = preprocessing.transform     + model.predict
```

Why use a Pipeline: less code, no accidental fitting on test data, correct re-fitting inside each CV fold, and a single `joblib` file that contains the preprocessing and the model.

---

## 10. Cross Validation with `StratifiedKFold`

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

```
Round 1:  [VALID] [train] [train] [train] [train]
...
Round 5:  [train] [train] [train] [train] [VALID]   → average the 5 scores
```

- It runs **only on `X_train`** (whatever you pass to `grid.fit`). `X_test` is never used and stays aside for the **final** check.
- The fold used for scoring inside CV is called the **validation** fold, to avoid confusing it with the test set.

```
7043 → 🎲 train_test_split → X_train 5634 | X_test 1409 (aside)
X_train → 🎲 StratifiedKFold → 5 folds of ~1127
```

- **Stratified**: each fold keeps ~26.5% churn, so comparisons are fair and F1 is stable.
- `shuffle=True` = mix the rows; `random_state=42` = mix them the **same way** every time.
- `cv=5` for a classifier already means StratifiedKFold, but **without shuffle**.

| Method | Use when |
|---|---|
| `KFold` | regression or balanced classes |
| **`StratifiedKFold`** | **classification, especially imbalanced** |
| `GroupKFold` | the same person appears in several rows |
| `TimeSeriesSplit` | data over time (train on past, test on future) |
| `LeaveOneOut` | very small datasets |

---

## 11. Evaluation Metrics

### Confusion matrix (Random Forest, 1409 test customers)

|  | Said STAY | Said CHURN |
|---|---|---|
| **Really STAYED** (1035) | TN **746** | FP **289** (false alarm) |
| **Really CHURNED** (374) | FN **75** (missed) | TP **299** (caught) |

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, final_model.predict(X_test))
```

| Metric | Question | Formula | RF |
|---|---|---|---|
| Accuracy | how much is correct overall? | (746+299)/1409 | 0.74 |
| Precision | when it says CHURN, is it right? | 299/(299+289) | 0.51 → wasted effort if low |
| Recall | of the real churners, how many were found? | 299/(299+75) | 0.80 → lost customers if low |
| F1 | is the balance good? | 2·P·R/(P+R) | 0.62 |

- **Accuracy is misleading with imbalance**: predicting "stay" for everyone already gives 73.5%. KNN had the **highest accuracy (0.77)** but the **worst recall (0.45)**.
- **F1 punishes imbalance**: P=1.0, R=0.1 → F1 ≈ 0.18 (not 0.55).
- **For churn, recall usually matters most** (a lost customer costs more than a wasted discount). Whether false alarms are acceptable depends on the cost of the retention action.

### What is a good F1?

The range is 0 to 1, where 1 = perfect. A real-world 1 (or 0.99) is suspicious and usually means **data leakage**.

| Strategy | F1 (churn) |
|---|---|
| everyone "stay" | 0.00 |
| random guessing | ~0.27 |
| everyone "churn" | ~0.42 |
| **Random Forest** | **0.62** (typical for this dataset: 0.60–0.65) |

Ideas to improve it: more features (`InternetService`, `PaymentMethod`, `TechSupport`, `OnlineSecurity`), gradient boosting, or tuning the decision threshold.

### Results (final run)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.742 | 0.509 | 0.799 | 0.622 |
| Decision Tree | 0.737 | 0.504 | 0.759 | 0.606 |
| KNN | 0.767 | 0.577 | 0.452 | 0.507 |

### Showing both classes in the scoreboard

By default `precision_score`, `recall_score` and `f1_score` report **class 1 only**. To add more:

```python
'F1 (churn)': f1_score(y_test, y_pred),
'F1 (stay)':  f1_score(y_test, y_pred, pos_label=0),
'F1 macro':   f1_score(y_test, y_pred, average='macro'),   # sanity check over both classes
```

Label columns clearly, e.g. "Precision (churn)".

---

## 12. Reading `classification_report`

```
              precision    recall  f1-score   support
           0       0.91      0.72      0.80      1035
           1       0.51      0.80      0.62       374
    accuracy                           0.74      1409
   macro avg       0.71      0.76      0.71      1409
weighted avg       0.80      0.74      0.76      1409
```

| Row / column | Meaning |
|---|---|
| `0` | metrics for **STAY**: precision 746/(746+75), recall 746/1035 |
| `1` | metrics for **CHURN** (the numbers in the results table) |
| `support` | how many **real** customers are in each class (a count, not a score) |
| `accuracy` | one number for all classes |
| `macro avg` | simple average, **each class counts equally** (fair for imbalance) |
| `weighted avg` | average weighted by support, so the big class counts more (can hide a weak small class). Weighted recall = accuracy |

---

## 13. Multi-class (3+ classes)

Example: 😞 disappointed / 😐 neutral / 😊 satisfied (300 customers).

|  | Said 😞 | Said 😐 | Said 😊 | support |
|---|---|---|---|---|
| **Really 😞** | **40** | 15 | 5 | 60 |
| **Really 😐** | 10 | **60** | 20 | 90 |
| **Really 😊** | 5 | 25 | **120** | 150 |

- **Accuracy** = diagonal ÷ total = 220/300 = 0.73.
- **Precision** = read down the **column** (😞 40/55 = 0.73, 😐 0.60, 😊 0.83).
- **Recall** = read across the **row** (😞 40/60 = 0.67, 😐 0.67, 😊 0.80).
- **F1** per class: 😞 0.70, 😐 0.63, 😊 0.81. `classification_report` shows **3 rows**.
- The middle class (neutral) gets confused with both neighbors. The business-critical class (😞) needs good recall.

### You must choose an average

```python
f1_score(y_test, y_pred, average='macro')      # 0.71, each class equal
f1_score(y_test, y_pred, average='weighted')   # 0.74, by class size
f1_score(y_test, y_pred, average=None)         # [0.70, 0.63, 0.81]
```

The binary scoreboard code **crashes** with 3 classes:

```
ValueError: Target is multiclass but average='binary'. Please choose another
average setting, one of [None, 'micro', 'macro', 'weighted'].
```

Fix: add `average='macro'` to precision, recall and F1. In GridSearch use `scoring='f1_macro'` instead of `'f1'` (otherwise it crashes there first).

```python
f1_each = f1_score(y_test, y_pred, average=None,
                   labels=['disappointed', 'neutral', 'satisfied'])
results.append({
    'Model': name,
    'Accuracy': accuracy_score(y_test, y_pred),
    'F1 macro': f1_score(y_test, y_pred, average='macro'),
    'F1 disappointed': f1_each[0],
    'F1 neutral': f1_each[1],
    'F1 satisfied': f1_each[2],
})
```

Other notes for multi-class:
- `stratify=y` and `StratifiedKFold` keep the ratio of all 3 classes.
- KNN, trees, forests and `class_weight='balanced'` all handle multiple classes automatically.
- Classifiers treat the classes as **unordered** categories (they don't know 😞 < 😐 < 😊).
