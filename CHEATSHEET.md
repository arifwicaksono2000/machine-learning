# Machine Learning Cheatsheet (Day 1–3 + Regression)

## Contents

**Day 1: Python & Pandas basics**
1. [Data Structures](#1-data-structures)
2. [Control Flow](#2-control-flow)
3. [Functions & Lambda](#3-functions--lambda)
4. [NumPy](#4-numpy)
5. [Pandas: Reading & Selecting Data](#5-pandas-reading--selecting-data)
6. [Pandas: Adding & Removing Columns/Rows](#6-pandas-adding--removing-columnsrows)
7. [EDA: Group By & Statistics](#7-eda-group-by--statistics)
8. [Handling Outliers (IQR)](#8-handling-outliers-iqr)

**Day 2: Visualization, Preprocessing & Clustering**

9. [Visualization](#9-visualization)
10. [Handling Missing Values](#10-handling-missing-values)
11. [Feature Engineering](#11-feature-engineering)
12. [Clustering Models](#12-clustering-models)
13. [Choosing k & Evaluating Clusters](#13-choosing-k--evaluating-clusters)
14. [Visualizing Clusters with PCA](#14-visualizing-clusters-with-pca)
15. [Full Workflow Summary](#15-full-workflow-summary)

**Day 3: Supervised Learning (Classification)**

16. [Supervised vs Unsupervised](#16-supervised-vs-unsupervised)
17. [Preparing the Data](#17-preparing-the-data)
18. [Train-Test Split & Scaling](#18-train-test-split--scaling)
19. [Classification Models](#19-classification-models)
20. [Evaluation Metrics](#20-evaluation-metrics)
21. [Cross Validation & Hyperparameter Tuning](#21-cross-validation--hyperparameter-tuning)
22. [Comparing Models & Saving the Final Model](#22-comparing-models--saving-the-final-model)
23. [Neural Network v2: One-Hot + Pipeline](#23-neural-network-v2-one-hot--pipeline)
24. [Choosing a Model: Why Trees Beat Neural Networks Here](#24-choosing-a-model-why-trees-beat-neural-networks-here)
25. [Supervised Workflow Summary](#25-supervised-workflow-summary)

**Regression**

26. [Regression Basics & Linear Regression](#26-regression-basics--linear-regression)
27. [Regression Metrics](#27-regression-metrics)
28. [Regression Models & Tuning](#28-regression-models--tuning)
29. [Plotting Actual vs Predicted](#29-plotting-actual-vs-predicted)
30. [Regression Workflow Summary](#30-regression-workflow-summary)

**Bonus**

31. [Time-Series & Market Data](#31-time-series--market-data)

---

# DAY 1

## 1. Data Structures

| Type       | Syntax | Mutable? | Accessed by |
|------------|--------|----------|-------------|
| List       | `[]`   | Yes      | index       |
| Tuple      | `()`   | No       | index       |
| Dictionary | `{}`   | Yes      | key         |
| Set        | `{}`   | Yes      | no order, no duplicates |

### List

```python
list_buah = ['mangga', 'jeruk', 'apel', 7]

list_buah[0]            # 'mangga'  -> first item
list_buah[0][0]         # 'm'       -> first letter of the first item
list_buah[-1]           # 7         -> last item
list_buah[len(list_buah) - 1]   # also the last item

list_buah[0] = 'melon'  # lists are mutable, so this works
# list_buah[0][0] = 'c' # ERROR: strings are immutable

list_buah[0:3]          # slicing: index 0, 1, 2 (the end index is NOT included)
list_buah[0:3:2]        # slicing with a step: index 0, 2

len(list_buah)          # number of items
list_buah.count('apel') # how many times 'apel' appears
list_buah.index('apel') # the index of 'apel'

[1, 2, 3] + [4]         # joining lists -> [1, 2, 3, 4]
```

### Tuple

```python
tuple_buah = ('mangga', 'jeruk', 'apel')
# tuple_buah[0] = 'x'   # ERROR: tuples are immutable

tuple_buah = list(tuple_buah)   # casting: convert to a list so it can be changed
```

### Dictionary

```python
dict_buah = {'buah': ['anggur', 'apel', 'mangga'], 1: (30000, 15000, 20000)}

dict_buah['buah']        # ['anggur', 'apel', 'mangga']
dict_buah['buah'][0]     # 'anggur'
dict_buah['buah'] = ['nanas', 'cherry', 'durian']   # replace a value
```

---

## 2. Control Flow

### if / elif / else

```python
nama = 'Bayu'

if 10 > 7:
    print(f"benar {nama.upper()}")   # f-string: puts variables inside a string
elif 4 > 6:
    print("benar 2")
else:
    print("tidak ada yang benar")
```

### Logical operators: `and`, `or`, `not`

```python
sim = True
usia = 17

if sim == True and usia >= 17:
    print("boleh mengendarai")
```

### for loop

```python
for i in range(0, 5):     # 0, 1, 2, 3, 4
    if i == 2:
        continue          # skip the rest of this iteration and go to the next one
        # break           # stop the whole loop
    print(i)              # prints 0 1 3 4

for buah in list_buah:    # loop over a list
    print(buah)

for index, buah in enumerate(list_buah):   # get the index and the value together
    print(f"index {index} -> {buah}")
```

### while loop

```python
i = 0
while i < 10:
    print(i)
    i += 1    # don't forget this, or the loop never ends
```

---

## 3. Functions & Lambda

There are 3 kinds: **built-in functions** (`print`, `len`, ...), **custom functions** (`def`), and **lambda functions**.

```python
# Custom function
def func1(a1, a2):
    return a1 + a2

func1(5, 10) + 3        # 18
```

```python
# Lambda: a short function with no name (anonymous function)
(lambda x: 5 + x)(6)    # 11
```

### One-line if expression

```python
# result_if_true if condition else result_if_false
(lambda a, b: f'{a} lebih dari {b}' if a > b else f'{a} kurang dari {b}')(11, 4)
# '11 lebih dari 4'
```

> A function without `return` gives back `None`.
> Example: `greet = print("halo")` → `type(greet)` is `NoneType`.

---

## 4. NumPy

```python
import numpy as np
```

### Basic arrays

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr1 ** 5               # the operation is applied to every item

arr2 = np.array([[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10]])          # 2D array (2 rows, 5 columns)

arr3 = np.array([[[1, 2, 3], [4, 5, 6]],
                 [[7, 8, 9], [10, 11, 12]]]) # 3D array
```

### Indexing: `arr[row, col]` (indexes start at 0)

```python
arr2[0, 4]       # 5
arr2[0, 2:4]     # [3, 4]
arr2[0:2, 1:3]   # [[2, 3], [7, 8]]
arr2[:, 1:3]     # ':' = all rows, columns 1 and 2
```

### Boolean filtering

```python
arr2 > 3                        # array of True/False
arr2[(arr2 > 3) & (arr2 < 7)]   # [4, 5, 6] -> only the values that match
```

> Use `&` (and) and `|` (or) with arrays, NOT `and`/`or`. Put each condition in `()`.

### Flatten vs ravel (N-dimensional → 1D)

```python
r = arr2.ravel()     # VIEW:  changing r ALSO changes arr2
r[0] = 100

f = arr2.flatten()   # COPY:  changing f does NOT change arr2
f[0] = 90
```

---

## 5. Pandas: Reading & Selecting Data

```python
import pandas as pd
import seaborn as sns

sns.get_dataset_names()          # list the sample datasets
df = sns.load_dataset('tips')    # load a sample dataset
df = pd.read_csv('life_insurance.csv')   # load from a CSV file
```

### Looking at the data

```python
df.head()                        # first 5 rows
df.tail()                        # last 5 rows
df.sample(10, random_state=42)   # 10 random rows (random_state = same result every time)
df.shape                         # (rows, columns)
df.columns                       # column names
```

### Selecting columns

```python
df['tip']                   # one column -> Series
df[['tip', 'total_bill']]   # several columns -> DataFrame (note the double brackets)
```

### `loc` vs `iloc`

| | `loc` | `iloc` |
|---|---|---|
| Stands for | **label** location | **integer** location |
| Slice end | **included** | **not included** |

```python
df.loc[1:4]                         # rows with labels 1 to 4 (4 is included)
df.iloc[1:3]                        # rows at positions 1 and 2
df.loc[1:3, ['total_bill', 'tip']]  # rows 1-3, only these columns
df.iloc[::2, ::2]                   # every 2nd row and every 2nd column
df.iloc[0, 0]                       # a single value
```

### Filtering rows

```python
df[df['day'] == 'Sun']                              # one condition
df[(df['day'] == 'Sun') & (df['total_bill'] > 20)]  # two conditions
len(df[(df['day'] == 'Sun') & (df['total_bill'] > 20)])   # count the rows that match

df['day'].unique()           # the distinct values in a column
df['sex'].value_counts()     # how many times each value appears
```

---

## 6. Pandas: Adding & Removing Columns/Rows

### Adding columns

```python
# 1. Same value for every row
df['week'] = 'weekend'

# 2. Using a custom function with .apply() on ONE column
def week(day):
    if (day == 'Sun') | (day == 'Sat'):
        return "weekend"
    else:
        return "weekday"

df['week'] = df['day'].apply(week)

# 3. The same thing as a one-line lambda
df['week'] = df['day'].apply(lambda x: 'weekend' if (x == 'Sun') | (x == 'Sat') else 'weekday')

# 4. Using SEVERAL columns -> df.apply(..., axis=1)  (axis=1 = go row by row)
df['visitor'] = df.apply(
    lambda x: 'High' if x['total_bill'] > 17 and x['size'] >= 3 else 'Low', axis=1)

df['tip_rate'] = df.apply(lambda x: x['tip'] / x['total_bill'] * 100, axis=1)
# simpler, and faster, way to do the same thing:
df['tip_rate'] = df['tip'] / df['total_bill'] * 100

# 5. assign() returns a NEW DataFrame (the original df is not changed)
df.assign(coba=df['tip'])
```

### Removing columns

```python
df.drop('day', axis=1)                   # axis=1 = column. Returns a copy.
df.drop(columns='day', inplace=True)     # inplace=True = change df permanently
```

> ⚠️ It is `columns=` (with an **s**). `column=` gives an error.

### Adding rows

```python
df.loc[244] = df.loc[0]    # create a new row with label 244 (a copy of row 0)
```

### Removing rows

```python
df.drop(246)                          # drop one row by label
df.drop([246, 245])                   # drop several rows
df.drop(df.loc[244:].index)           # drop every row from label 244 onward

# drop the rows that match a condition
df.drop(df[df['sex'] == 'Female'].index, inplace=True)
```

### Resetting the index

After you add or drop rows, the index can have gaps or be out of order. Reset it:

```python
df.reset_index(drop=True, inplace=True)   # drop=True = don't keep the old index as a column
```

> 💡 **`inplace` rule:** without `inplace=True`, most methods return a **new** DataFrame and the original stays the same. Either use `inplace=True`, **or** assign the result (`df = df.drop(...)`). Don't do both.

---

## 7. EDA: Group By & Statistics

EDA = **Exploratory Data Analysis**: getting to know your data before you build a model.

```python
df.groupby('sex')['total_bill'].sum()              # one column
df.groupby('sex')[['total_bill', 'tip']].sum()     # several columns

df.groupby('day')['size'].sum()      # total visitors per day
df.groupby('day')['size'].count()    # number of transactions per day

# several aggregations at once
df.groupby('week')['total_bill'].agg(['sum', 'mean'])

# named aggregations + sorting
df.groupby('day').agg(
    rerata=('total_bill', 'mean'),
    jumlah=('total_bill', 'sum')
).sort_values(by='jumlah', ascending=False)

# group by several columns
df.groupby(['class', 'embark_town'])['fare'].median()
```

### Summary statistics

```python
df.describe()                # count, mean, std, min, quartiles, max
df.corr(numeric_only=True)   # correlation between the numeric columns
```

> **Pearson correlation** measures the *linear* relationship between two variables. It ranges from **-1 to 1**. The closer it is to 1 or -1, the stronger the relationship. 0 means no linear relationship.

---

## 8. Handling Outliers (IQR)

```
IQR         = Q3 - Q1
Upper bound = Q3 + 1.5 × IQR
Lower bound = Q1 - 1.5 × IQR
Values outside these bounds are outliers.
```

```python
df['total_bill'].plot.box()          # a box plot shows outliers as dots
df['total_bill'].plot(kind='hist')   # histogram

q1 = df['total_bill'].quantile(0.25)
q3 = df['total_bill'].quantile(0.75)
iqr = q3 - q1

upper = q3 + 1.5 * iqr
lower = q1 - 1.5 * iqr

df_clean = df[(df['total_bill'] >= lower) & (df['total_bill'] <= upper)]   # remove outliers
```

---

# DAY 2

## 9. Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

### Common plots

```python
df.groupby('smoker')['size'].sum().plot.bar()                 # bar chart with pandas

sns.countplot(data=df, x='smoker', hue='sex')                 # count per category
sns.scatterplot(data=df, x='tip', y='total_bill', hue='sex')  # relationship between 2 numbers
sns.histplot(data=df, x='total_bill', kde=True)               # distribution (shows skewness)
```

`hue` = color the points or bars by a category.

### Several plots in one figure (subplots)

```python
fig, ax = plt.subplots(2, 2, figsize=(12, 8))   # 2 rows × 2 columns
ax = ax.flatten()                               # turn the 2D grid into a list: ax[0]..ax[3]

sns.scatterplot(data=df, x='tip', y='total_bill', hue='sex', ax=ax[0],
                palette=['skyblue', 'pink'])
sns.histplot(data=df, x='total_bill', kde=True, ax=ax[1])
sns.countplot(data=df, x='smoker', hue='sex', ax=ax[2])
ax[3].pie(df['sex'].value_counts(),
          labels=df['sex'].value_counts().index,
          autopct='%.2f%%')                     # show percentages

ax[0].set_title('Hubungan antara Tip dan Total Bill')
plt.savefig('chart.png')                        # save to a file
```

> Without `flatten()` you index the grid as `ax[0, 0]`, `ax[0, 1]`, and so on.

### Palettes

```python
sns.husl_palette(2)         # n evenly spaced colors
sns.dark_palette('green')   # shades of one color
```

---

## 10. Handling Missing Values

Two approaches:
1. **Delete the data**: when fixing it takes a lot of effort and there aren't many missing values.
2. **Imputation** (fill it in): with the **mean**, **median**, **mode**, or **interpolation** (only for data that follows a pattern).

```python
df = sns.load_dataset('titanic')

df.isnull().sum()                     # missing values per column
df[df['embark_town'].isnull()]        # look at the rows with missing values
pd.isnull(df['age'].values[0])        # check a single value

df.drop(columns='deck', inplace=True) # too many missing values -> drop the column
```

### Fill with a fixed value

```python
df['embark_town'] = df['embark_town'].fillna('Cherbourg')
df['embarked']    = df['embarked'].fillna('C')
```

> ⚠️ Don't write `df['col'] = df['col'].fillna(x, inplace=True)`. With `inplace=True` the method returns `None`, so the whole column becomes `None`. Use assignment **or** inplace, not both.

### Fill based on a group

```python
df.groupby('who')['age'].mean()     # man ≈ 33, woman ≈ 32, child ≈ ...

df['age'] = df.apply(
    lambda x: 33 if (x['who'] == 'man') & pd.isnull(x['age'])
    else 32 if (x['who'] == 'woman') & pd.isnull(x['age'])
    else x['age'],
    axis=1)
```

### Interpolation (for data with a pattern)

```python
data = pd.DataFrame([[1, 100], [2, 120], [3, np.nan], [4, 160]], columns=['No', 'Nilai'])
data['Nilai'] = data['Nilai'].interpolate()    # NaN -> 140 (halfway between 120 and 160)
```

---

## 11. Feature Engineering

Case study: **customer segmentation** using financial and demographic data (`life_insurance.csv`).

### 11.1 Feature Selection

| Method   | How it works |
|----------|--------------|
| Filter   | Uses descriptive statistics (for example correlation) |
| Wrapper  | Tries different combinations of features |
| Embedded | The model picks the features while it trains (L1/L2 regularization) |

```python
df = pd.read_csv('life_insurance.csv')

feature = ['age', 'gender', 'marital_status', 'number_of_dependents', 'annual_income',
           'occupation', 'health_status', 'smoking_status', 'policy_type',
           'coverage_amount', 'monthly_premium', 'policy_start_date']
X = df[feature]
```

### 11.2 Feature Transformation (fixing skewed data)

Check the skewness first:

```python
sns.histplot(data=X, x='annual_income', kde=True)
X['annual_income'].skew()
```

| Skewness | Meaning |
|----------|---------|
| < 0.5    | roughly normal |
| 0.5 – 1  | moderately skewed |
| > 1      | highly skewed |

Ways to transform:

```python
# 1. Log transformation (log1p = log(1 + x), safe when x = 0)
X['annual_income'] = np.log1p(X['annual_income'])
X['coverage_amount'] = np.log1p(X['coverage_amount'])

# 2. Square root
np.sqrt(X['annual_income'])

# 3. Binning: turn numbers into categories
bins  = [0, 100000, 150000, 200000]
label = ['rendah', 'sedang', 'tinggi']
pd.cut(X['annual_income'], bins=bins, labels=label)
```

> To undo `np.log1p`, use `np.expm1`.

### 11.3 Cleaning bad categories

```python
X['gender'].unique()                                    # look for odd values like '_RARE_'
X.drop(X[X['gender'] == '_RARE_'].index, inplace=True)
X.drop(X[X['health_status'] == '_RARE_'].index, inplace=True)
X.reset_index(drop=True, inplace=True)
```

### 11.4 Encoding (text → numbers)

| Encoding | Use it for | Example |
|----------|------------|---------|
| Label / `map` | 2 categories | Female=0, Male=1 |
| **Ordinal** | Categories that have an **order** | Fair < Good < Excellent |
| **One-hot** | Categories with **no order** (nominal) | Single / Married / Divorced |

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Label encoding with map
X['gender'] = X['gender'].map({'Female': 0, 'Male': 1})

# Ordinal encoding: you set the order yourself
oe = OrdinalEncoder(categories=[['Fair', 'Good', 'Excellent']])
X['health_status'] = oe.fit_transform(X[['health_status']])    # Fair=0, Good=1, Excellent=2

# One-hot encoding: one new 0/1 column per category
ohe = OneHotEncoder(sparse_output=False)
data_marital = pd.DataFrame(ohe.fit_transform(X[['marital_status']]),
                            columns=ohe.get_feature_names_out())
X = pd.concat([X, data_marital], axis=1)     # axis=1 = join side by side
X.drop(columns='marital_status', inplace=True)
```

> **sklearn pattern:** make an object → `.fit()` (learn) → `.transform()` / `.predict()` (apply).
> `.fit_transform()` does both steps at once.
> Note: sklearn expects a **2D** input, so use `X[['col']]` (double brackets), not `X['col']`.

### 11.5 Creating new features from dates

```python
X['tenure'] = ((pd.Timestamp.today() - pd.to_datetime(X['policy_start_date'])).dt.days / 365.25).round(1)
X.drop(columns='policy_start_date', inplace=True)    # the model can't use raw dates
```

### 11.6 Scaling

Clustering is based on **distance**, so features on a large scale (like income) would outweigh small ones (like age). Scaling puts all features on the same scale.

| Scaler | Result |
|--------|--------|
| `MinMaxScaler` | values between 0 and 1 |
| `StandardScaler` | mean = 0, std = 1 |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Saving the cleaned data

```python
X.to_csv('life_insurance_cleaned.csv', index=False)
```

---

## 12. Clustering Models

**Unsupervised learning**: the data has no labels, and the model finds groups by itself.

| Algorithm  | Based on | Main parameter(s) | Need to set the number of clusters? |
|------------|----------|-------------------|--------------------------|
| K-Means    | Centroid (center point) | `n_clusters` | Yes |
| Mean-Shift | Density | `bandwidth` | No |
| DBSCAN     | Density | `eps`, `min_samples` | No |

### K-Means

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=4, n_init=10, random_state=42)
km_cluster = km.fit_predict(X_scaled)   # train and get the labels in one step

km.inertia_           # WCSS (see section 13)
km.cluster_centers_   # the center of each cluster

# Look at what each cluster is like
X['km_cluster'] = km_cluster
X.groupby('km_cluster').mean()
```

### Mean-Shift

```python
from sklearn.cluster import MeanShift

ms = MeanShift()               # finds the bandwidth automatically if you don't give one
labels = ms.fit_predict(X_scaled)
```

### DBSCAN

- `eps`: the maximum distance for two points to count as neighbors
- `min_samples`: the minimum number of neighbors a point needs to count as part of a dense area
- Points that don't belong to any cluster get the label **`-1`** (noise)

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=2.5, min_samples=5)
db_cluster = db.fit_predict(X_scaled)

set(db_cluster)   # the cluster labels found (-1 = noise)
```

Trying several `eps` values:

```python
sh_score_db = []
for e in [0.5, 1, 1.5, 2, 2.5]:
    db = DBSCAN(eps=e, min_samples=5)   # use the loop variable here, not a fixed number
    db_cluster = db.fit_predict(X_scaled)
    sh_score_db.append(silhouette_score(X_scaled, db_cluster))
```

---

## 13. Choosing k & Evaluating Clusters

### Elbow method (inertia / WCSS)

**Inertia = WCSS** (Within-Cluster Sum of Squares): how tightly packed each cluster is. Lower is tighter, but it always goes down as k goes up. Pick the **"elbow"**, the point where the curve starts to flatten.

```python
inertia = []
for i in range(2, 20):
    km = KMeans(n_clusters=i, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(2, 20), inertia, marker='s')
```

### Evaluation metrics

| Metric | Range | Better when |
|--------|-------|-------------|
| **Silhouette Score** | -1 to 1 | closer to **1** |
| **Calinski-Harabasz** | 0 to ∞ | **larger** |

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

sh_score = []
for i in range(2, 20):
    km = KMeans(n_clusters=i, n_init=10, random_state=42)
    cluster = km.fit_predict(X_scaled)
    sh_score.append(silhouette_score(X_scaled, cluster))

plt.plot(range(2, 20), sh_score, marker='s')    # pick the k with the highest score
```

```python
km = KMeans(n_clusters=4, n_init=10, random_state=42)
km_cluster = km.fit_predict(X_scaled)

print('Silhouette Score:', silhouette_score(X_scaled, km_cluster))
print('Calinski Score:  ', calinski_harabasz_score(X_scaled, km_cluster))
```

---

## 14. Visualizing Clusters with PCA

The data has many columns, so you can't plot it directly. **PCA** (Principal Component Analysis) compresses it down to 2 dimensions while keeping as much information as it can.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_scaled_pca = pca.fit_transform(X_scaled)

pca.explained_variance_ratio_   # how much of the information each component keeps

# move the centroids into the same 2D space
centroid_pca = pca.transform(km.cluster_centers_)

plt.scatter(X_scaled_pca[:, 0], X_scaled_pca[:, 1], c=km_cluster)  # points colored by cluster
plt.scatter(centroid_pca[:, 0], centroid_pca[:, 1], c='red')       # centroids
```

You can also plot two of the original features:

```python
plt.scatter(X['annual_income'], X['coverage_amount'], c=X['km_cluster'])
```

---

## 15. Full Workflow Summary

```
1. Load data            pd.read_csv()
2. Explore (EDA)        head(), describe(), isnull().sum(), histplot, groupby
3. Clean                handle missing values, outliers, bad categories ('_RARE_')
4. Select features      pick the columns that matter for the problem
5. Transform            log1p / sqrt for skewed columns
6. Encode               map / OrdinalEncoder / OneHotEncoder
7. Scale                StandardScaler / MinMaxScaler
8. Model                KMeans / MeanShift / DBSCAN
9. Choose k & evaluate  elbow, silhouette, calinski-harabasz
10. Interpret           groupby('cluster').mean(), PCA plot
```

---

# DAY 3

Case study: predict whether a person earns **more than 50K a year** (`adult` census dataset).

## 16. Supervised vs Unsupervised

| | Unsupervised (Day 2) | Supervised (Day 3) |
|---|---|---|
| Labels / target? | No | **Yes**: the `y` column you want to predict |
| Goal | Find groups | Predict the label for new data |
| Examples | K-Means, DBSCAN | KNN, Naive Bayes, Decision Tree, Random Forest, Neural Network |

- **X** = features (the inputs)
- **y** = target (the answer)
- **Classification** = the target is a category (`<=50K` / `>50K`). **Regression** = the target is a number.

> 💡 Supervised / unsupervised describes **how you train**, not which model you use. The question is: *does training use the correct answers (`y`)?*
> - A neural network trained with `fit(X, y)` is **supervised**. It learns its own internal features, but the labels steer that learning.
> - The same kind of network can be **unsupervised** (an *autoencoder* rebuilds its input, with no labels) or **self-supervised**: the labels come from the data itself. ChatGPT and Claude learn by predicting the next word in huge amounts of text, so no human labeling is needed.

---

## 17. Preparing the Data

### Loading a CSV that has no header

```python
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/adult-all.csv',
                 header=None)                 # the file has no column-name row

df.info()                                     # data types + non-null counts
df.drop(columns=3, inplace=True)              # drop a column we don't need (by number)

df.columns = ['age', 'workclass', 'fnlwgt', 'num_edu', 'marital-status', 'occupation',
              'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
              'work-hour-week', 'country', 'income']   # give the columns names
```

### Hidden missing values

Missing values aren't always `NaN`. In this dataset they are written as `'?'`, so `isnull()` won't find them.

```python
(df == '?').sum()          # count the '?' in every column

# Option 1: replace with a new category
df['workclass']  = df['workclass'].apply(lambda x: 'unknown' if x == '?' else x)
df['occupation'] = df['occupation'].apply(lambda x: 'unknown' if x == '?' else x)

# Option 2: replace with the mode (the most common value)
df['country'].value_counts()        # 'United-States' is by far the most common
df['country'] = df['country'].apply(lambda x: 'United-States' if x == '?' else x)
```

> A shorter way to do the same thing: `df['workclass'] = df['workclass'].replace('?', 'unknown')`

### Split into X and y

```python
y = df['income']

feature = ['age', 'workclass', 'num_edu', 'occupation', 'capital-gain',
           'capital-loss', 'work-hour-week', 'country']
X = df[feature].copy()     # .copy() avoids the SettingWithCopyWarning when you change X later
```

### Encoding the target: `LabelEncoder`

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)    # '<=50K' -> 0, '>50K' -> 1
le.classes_                # array(['<=50K', '>50K']) -> index = the encoded number
```

> ⚠️ `fit_transform()` returns a **NumPy array**, so `y` no longer has `.columns` or `.name`. To save it as a CSV, rebuild it:
> `pd.DataFrame(y_test, columns=['income'])`

### Encoding several feature columns in a loop

```python
for col in ['workclass', 'occupation', 'country']:
    le = LabelEncoder()
    X['en_' + col] = le.fit_transform(X[col])    # new column, e.g. 'en_workclass'

X.drop(columns=['workclass', 'occupation', 'country'], inplace=True)
```

> ⚠️ Write `X['en_' + col]`, **not** `X['en-', col]`. The comma creates a *tuple* as the column name, `('en-', 'workclass')`, which sklearn rejects. That's why the notebook needed the `X.columns = X.columns.astype('str')` workaround.

> 💡 LabelEncoder on features gives the categories a fake order (for example Cuba=4 < USA=38). Tree models don't mind that. For **KNN**, **neural networks** and other distance- or gradient-based models, **one-hot encoding** (Day 2) is usually better. See section 23.

### Check the class balance

```python
df['income'].value_counts()
# <=50K    37155   (~76%)
# >50K     11687   (~24%)   -> imbalanced
```

When one class is much larger than the other, accuracy can look good even if the model misses most of the small class. Always check the balance.

---

## 18. Train-Test Split & Scaling

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% test, 80% train
    random_state=42,     # same split every time
    stratify=y           # keep the same 0/1 ratio in train and test
)
```

- **Train set**: the model learns from this.
- **Test set**: kept hidden until the end, to check how the model does on data it has never seen.

### Scale AFTER the split

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit (learn mean/std) ONLY on train
X_test_scaled  = scaler.transform(X_test)        # only transform test, never fit it
```

| Method | What it does |
|--------|--------------|
| `fit` | **Learns** the settings from the data (for `StandardScaler`: the mean and std of each column) |
| `transform` | **Applies** the learned settings: `(x - mean) / std` |
| `fit_transform` | Both at once. Use it **only on train** |

> **Why?** If the scaler learns from the test data, information from the test set "leaks" into training, and your score looks better than it really is. This is called **data leakage**.
>
> The test set stands in for **future data**. When the model is used for real, new rows arrive one at a time, so you can't calculate a mean from them. Test data must use the **same** mean/std the model was trained with.
>
> Example: train mean = 50, std = 10, and the test set's mean is 60.
> - ✅ `transform`: a test value of 60 → `(60-50)/10 = 1.0`, so the model sees "above average"
> - ❌ `fit_transform` on test: 60 → `(60-60)/std = 0`, so the model sees "average" and that information is lost
>
> **Rule:** fit only on train, then `transform` everything else (validation, test, production). It applies to every learned preprocessing step: scalers, encoders, imputers, PCA.

> 💡 KNN and neural networks need scaling (distance-based / gradient-based). Decision Tree and Random Forest don't need it, but it doesn't hurt them.

---

## 19. Classification Models

All sklearn models work the same way:

```python
model = SomeModel(params)
model.fit(X_train_scaled, y_train)       # learn
y_pred = model.predict(X_test_scaled)    # predict
```

| Model | How it works | Main parameters |
|-------|--------------|-----------------|
| **KNN** (K-Nearest Neighbors) | Looks at the k closest training points and takes a vote | `n_neighbors`, `weights`, `p` |
| **Gaussian Naive Bayes** | Uses probability (Bayes' theorem) | `var_smoothing` |
| **Decision Tree** | Splits the data with yes/no questions | `max_depth`, `criterion`, `min_samples_split`, `min_samples_leaf` |
| **Random Forest** | Many decision trees vote together | `n_estimators` (number of trees), plus the tree parameters |
| **Neural Network** (MLP) | Layers of neurons; each layer combines the previous one's outputs | `hidden_layer_sizes`, `alpha`, `learning_rate_init` |

### KNN

```python
from sklearn.neighbors import KNeighborsClassifier
import time

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')

start = time.time()
knn.fit(X_train_scaled, y_train)
time.time() - start          # how long training took

y_pred = knn.predict(X_test_scaled)
```

- `n_neighbors`: how many neighbors vote
- `weights='uniform'`: every neighbor's vote counts the same. `'distance'`: closer neighbors count more.
- `p=1`: Manhattan distance. `p=2`: Euclidean distance.

### Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
```

### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
```

- `max_depth`: how deep the tree can grow (`None` = no limit, which risks overfitting)
- `criterion`: `'gini'` or `'entropy'`, how the tree measures a good split
- `min_samples_split`: minimum number of samples needed to split a node
- `min_samples_leaf`: minimum number of samples allowed in a leaf

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
```

### Neural Network (MLP: Multi-Layer Perceptron)

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(max_iter=500, early_stopping=True, random_state=42)

mlp_param = {
    'hidden_layer_sizes': [(32,), (64,), (64, 32)],
    'alpha': [1e-4, 1e-3, 1e-2],
    'learning_rate_init': [1e-3, 1e-2]
}
```

- `hidden_layer_sizes`: neurons per hidden layer. `(64,)` = 1 layer of 64; `(64, 32)` = 2 layers
- `alpha`: L2 regularization. Higher = simpler model, less overfitting
- `learning_rate_init`: how big each learning step is
- `max_iter`: maximum training rounds (epochs)
- `early_stopping=True`: hold back 10% of train and stop when the score stops improving
- **Must be scaled.** Neural networks train badly on unscaled data.

---

## 20. Evaluation Metrics

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))   # precision, recall, f1 for every class
```

### Confusion matrix

```
                  Predicted 0      Predicted 1
Actual 0          TN (correct)     FP (false alarm)
Actual 1          FN (missed)      TP (correct)
```

| Metric | Formula | Question it answers |
|--------|---------|---------------------|
| **Accuracy** | (TP + TN) / all | How many predictions were right overall? |
| **Precision** | TP / (TP + FP) | When the model says "1", how often is it right? |
| **Recall** | TP / (TP + FN) | Of all the real "1"s, how many did the model find? |
| **F1-score** | 2 × P × R / (P + R) | A balance of precision and recall |

> With imbalanced data, look at **recall and F1 for the small class** (`>50K`), not only accuracy. In the notebook, KNN had 81% accuracy but only **51% recall** for `>50K`, so it missed about half of the high earners.

In `classification_report`:
- **support**: the number of real samples in each class
- **macro avg**: the plain average across classes (treats every class equally)
- **weighted avg**: the average weighted by support (the large class dominates)

---

## 21. Cross Validation & Hyperparameter Tuning

### Cross validation: `StratifiedKFold`

Instead of testing on one split, the training data is split into **k folds**. The model trains on k-1 folds and validates on the remaining one, k times. The score is the average, which is more reliable. *Stratified* = each fold keeps the same class ratio.

```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Hyperparameter tuning: `GridSearchCV`

**Hyperparameters** are the settings *you* choose before training (like `n_neighbors`). GridSearchCV tries **every combination** and keeps the best one.

```python
knn_param = {
    'n_neighbors': [5, 10, 15],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}                                    # 3 × 2 × 2 = 12 combinations × 5 folds = 60 fits

grid_knn = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=knn_param,
    scoring='accuracy',              # or 'f1', 'recall', ...
    cv=cv,
    n_jobs=-1                        # use all CPU cores
)

grid_knn.fit(X_train_scaled, y_train)

grid_knn.best_params_       # {'n_neighbors': 10, 'p': 1, 'weights': 'uniform'}
grid_knn.best_score_        # average CV score of the best combination
best_knn = grid_knn.best_estimator_   # the model with the best settings, already trained

accuracy_score(y_test, best_knn.predict(X_test_scaled))
```

Parameter grids used for the other models:

```python
gnb_param = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}

dt_param = {
    'max_depth': [5, 10, 15, 20, None],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

rfc_param = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 5]
}
```

> 💡 With imbalanced data, try `scoring='f1'` instead of `'accuracy'`, so the search picks the model that is best at finding the small class.

---

## 22. Comparing Models & Saving the Final Model

```python
models = {
    'KNN': best_knn,
    'GNB': best_gnb,
    'DT':  best_dt,
    'RFC': best_rfc,
    'MLP': best_mlp
}

for nama, model in models.items():
    y_pred = model.predict(X_test_scaled)
    print('Model:', nama)
    print(classification_report(y_test, y_pred))
```

> ⚠️ Name the dict `models` (plural). If it's also called `model`, the loop overwrites it, and running the cell a second time fails.

### Results from the notebook

| Model | Test accuracy | Recall (`>50K`) | F1 (`>50K`) |
|-------|---------------|-----------------|-------------|
| KNN   | 0.82 | 0.45 | 0.55 |
| GNB   | 0.80 | 0.32 | 0.43 |
| Decision Tree | 0.84 | 0.45 | 0.58 |
| **Random Forest** | **0.84** | **0.48** | **0.60** |
| Neural Network (MLP) | 0.83 | 0.48 | 0.57 |

Random Forest won. Best parameters: `max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100`.

### Saving and loading a model: `joblib`

```python
import joblib

final_model = best_rfc
joblib.dump(final_model, 'final_rfc.joblib')           # save to a file

loaded = joblib.load('final_rfc.joblib')               # load it later
loaded.predict(X_test_scaled)                          # ready to use, no retraining
```

> New data must go through the **same** preprocessing before you predict: same encoders, same `scaler` (`transform` only). Save the scaler with `joblib.dump` too.

### Saving the split data

```python
pd.DataFrame(X_test_scaled,  columns=X.columns).to_csv('X_test_scaled.csv',  index=False)
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('X_train_scaled.csv', index=False)
pd.DataFrame(y_test,  columns=['income']).to_csv('y_test.csv',  index=False)
pd.DataFrame(y_train, columns=['income']).to_csv('y_train.csv', index=False)
```

---

## 23. Neural Network v2: One-Hot + Pipeline

Two problems held the first MLP back:
1. **LabelEncoder** turns categories into fake ordered numbers (Cuba=4 < USA=38). Trees don't mind, but a neural network reads them as a scale. → Use **one-hot encoding**.
2. Useful columns were left out. Adding `marital-status` and `relationship` helped the most.

> `race` and `sex` were left out on purpose. Using them to predict income raises fairness concerns, and they only added about 0.3 points.

### `ColumnTransformer` + `Pipeline`

- `ColumnTransformer`: apply **different preprocessing to different columns** (scale the numbers, one-hot the categories).
- `Pipeline`: chain the preprocessing and the model into **one object** with a single `fit` / `predict`.

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_col = ['age', 'num_edu', 'capital-gain', 'capital-loss', 'work-hour-week']
cat_col = ['workclass', 'occupation', 'country', 'marital-status', 'relationship']

X2 = df[num_col + cat_col]          # raw columns, NOT pre-encoded or pre-scaled
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=.2, random_state=42, stratify=y)

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_col),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_col)   # ignore categories unseen in train
])

mlp2_pipe = Pipeline([
    ('pre', preprocess),
    ('mlp', MLPClassifier(max_iter=500, early_stopping=True, random_state=42))
])

mlp2_param = {                       # prefix = step name + double underscore
    'mlp__hidden_layer_sizes': [(32,), (64,), (64, 32)],
    'mlp__alpha': [1e-4, 1e-3, 1e-2],
    'mlp__learning_rate_init': [1e-3, 1e-2]
}

mlp2_grid = GridSearchCV(mlp2_pipe, mlp2_param, cv=cv, scoring='accuracy', n_jobs=-1)
mlp2_grid.fit(X2_train, y2_train)    # raw data in, the pipeline does the rest
best_mlp2 = mlp2_grid.best_estimator_
accuracy_score(y2_test, best_mlp2.predict(X2_test))
```

> 💡 **Why use a pipeline?** It applies the fit-on-train-only rule (section 18) for you, **inside every CV fold** too. Without one, a scaler fitted on all of `X_train` leaks a little information into each fold's validation part. It's also safer to save: `joblib.dump(pipeline)` stores the scaler, the encoder and the model together.

> 💡 Same `random_state` + `stratify` + the same rows → the same test rows as the other models, so the scores are directly comparable.

### Results (same test rows)

| Model | Accuracy | F1 (`>50K`) |
|-------|----------|-------------|
| MLP, label-encoded | 0.828 | 0.57 |
| MLP, one-hot only | 0.833 | 0.59 |
| **MLP v2** (one-hot + extra columns) | **0.857** | **0.66** |
| RFC, label-encoded | 0.843 | 0.60 |
| **RFC v2** (same changes) | **0.870** | **0.69** |

- One-hot alone added only ~0.5 points. **Better features** (the extra columns) did most of the work.
- Log-scaling `capital-gain` / `capital-loss` made no difference.
- For a fair test, give **every** model the same improvements. RFC v2 still wins.

---

## 24. Choosing a Model: Why Trees Beat Neural Networks Here

Neural networks are state of the art for **unstructured data**: images, audio, text, video. Their strength is **learning features from raw data** (pixels → edges → shapes → "cat"). That's why they power ChatGPT, image generators and speech recognition.

On **tabular data** (spreadsheets), a person has already chosen meaningful columns, so that strength has little left to do, and trees fit the data better:

| Tabular data property | Why trees handle it better |
|---|---|
| Sharp thresholds ("capital-gain > 7000 → >50K") | One split for a tree. A network has to approximate it with smooth curves |
| Columns with different meanings and scales | Trees don't care about scale. Networks need careful scaling and encoding |
| Some useless columns | Trees mostly ignore them. Networks can get distracted |
| Small data (48k rows is small) | Networks usually need much more data |

Research backs this up: *"Why do tree-based models still outperform deep learning on tabular data?"* (Grinsztajn et al., 2022). Gradient boosting (XGBoost, LightGBM, CatBoost) usually wins tabular Kaggle competitions.

### Which model, by data type

| Data | Start with |
|------|-----------|
| Images, text, audio | Neural networks |
| Tables / spreadsheets | Gradient boosting → Random Forest → neural network to compare |
| Tables with huge-cardinality IDs (user ID, product ID) at massive scale | Neural networks with embeddings (e.g. recommendation and ad-click systems) |

### What about 10 million rows?

- More rows of the **same columns** won't lift the ceiling much. Accuracy is limited by **what the columns can tell you**: two people with identical features can earn differently for reasons the data doesn't record. For Adult, the best models land around **87–88%**.
- **To really improve, add new columns, not more rows.**
- At that scale: LightGBM is likely best or tied; a proper deep network on a GPU (not `MLPClassifier`) improves most from the extra data and could tie; Random Forest gets slow and memory-hungry; KNN becomes impractical (every prediction searches 10M rows); Naive Bayes barely changes.

> **No free lunch:** no single algorithm is best for everything. Pick the tool that fits the data, and test it.

---

## 25. Supervised Workflow Summary

```
1. Load data              pd.read_csv(..., header=None) + name the columns
2. Explore                info(), value_counts(), (df == '?').sum(), class balance
3. Clean                  replace hidden missing values ('?') with 'unknown' / the mode
4. Split X and y          X = features, y = target
5. Encode                 LabelEncoder for y; one-hot for categorical features (LabelEncoder only for trees)
6. Train-test split       train_test_split(..., stratify=y)
7. Scale                  fit on TRAIN only, transform test   (or let a Pipeline do 5 + 7)
8. Train baseline         model.fit() -> predict -> classification_report
9. Tune                   GridSearchCV + StratifiedKFold
10. Compare models        accuracy + recall/F1 of the small class, same test rows for all
11. Improve               better features usually beat a fancier model
12. Save                  joblib.dump(final_model) (and the scaler), or dump the whole Pipeline
```

---

# REGRESSION

Case study: predict the **price of a house** (Boston `housing` dataset, 506 rows, 13 features). The target, column 13, is the median house value in $1000s.

## 26. Regression Basics & Linear Regression

**Regression** = supervised learning where the target is a **number** (price), not a category.

| | Classification | Regression |
|---|---|---|
| Target | Category (`<=50K` / `>50K`) | Number (24.0, 21.6, ...) |
| Models | `...Classifier` | `...Regressor`, `LinearRegression`, `SVR` |
| Metrics | accuracy, precision, recall, F1 | MAE, RMSE, R² |
| Split | `stratify=y` | no `stratify` (a continuous `y` can't be stratified) |
| CV | `StratifiedKFold` | `KFold` |

### Selecting X and y by position

```python
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/housing.csv', header=None)

X = df.iloc[:, :-1]    # every column except the last
y = df.iloc[:, -1]     # only the last column
```

### Split, scale, train

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only (section 18)
X_test_scaled  = scaler.transform(X_test)

lr = LinearRegression()          # fits a straight line (hyperplane): y = b0 + b1*x1 + b2*x2 + ...
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
```

---

## 27. Regression Metrics

```python
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

print('MAE: ',      mean_absolute_error(y_test, y_pred))
print('RMSE: ',     root_mean_squared_error(y_test, y_pred))
print('R2-Score: ', r2_score(y_test, y_pred))
```

| Metric | Meaning | Better when |
|--------|---------|-------------|
| **MAE** (Mean Absolute Error) | Average size of the error, in the target's unit. MAE 3.19 = off by ~$3,190 on average | lower |
| **RMSE** (Root Mean Squared Error) | Like MAE, but big errors are punished more | lower |
| **R²** | Share of the target's variation the model explains. 1 = perfect, 0 = no better than always guessing the mean, can go negative | closer to **1** |

> 💡 If RMSE is much larger than MAE, the model makes a few **big** mistakes.

---

## 28. Regression Models & Tuning

```python
from sklearn.model_selection import KFold, GridSearchCV

cv = KFold(n_splits=5, shuffle=True, random_state=42)   # KFold, not StratifiedKFold
```

### SVR (Support Vector Regressor)

```python
from sklearn.svm import SVR

svr_param = {
    'C': [0.1, 0.5, 1],                     # higher = fits the training data more tightly
    'epsilon': [0.1, 0.5, 1],               # errors smaller than epsilon are ignored
    'kernel': ['linear', 'rbf', 'poly']     # the shape of the function: straight line or curved
}

svr_grid = GridSearchCV(estimator=SVR(), param_grid=svr_param, cv=cv,
                        scoring='r2', n_jobs=-1)    # scoring='r2' for regression
svr_grid.fit(X_train_scaled, y_train)
best_svr = svr_grid.best_estimator_                 # {'C': 1, 'epsilon': 0.5, 'kernel': 'linear'}
```

> SVR needs scaling (like KNN).

### Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

rfr_param = {
    'n_estimators': [20, 30, 40, 50, 60],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 5]
}

rfr_grid = GridSearchCV(estimator=RandomForestRegressor(), param_grid=rfr_param,
                        cv=cv, scoring='r2', n_jobs=-1)
rfr_grid.fit(X_train_scaled, y_train)
best_rfr = rfr_grid.best_estimator_
```

> In a regressor, every tree predicts a **number** and the forest takes the **average** (a classifier takes a vote).

### Results (test set)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 3.19 | 4.93 | 0.67 |
| SVR (tuned) | 3.13 | 5.40 | 0.60 |
| **Random Forest** | **2.22** | **3.09** | **0.87** |

Random Forest won by a wide margin. Best parameters: `n_estimators=40, max_depth=None, min_samples_split=2, min_samples_leaf=1`.

> 💡 The tuned SVR picked a **linear** kernel and still scored lower than plain Linear Regression on the test set. A good CV score (0.70) doesn't guarantee a good test score, especially with only 102 test rows.

### Pipeline for regression

The same idea as section 23: the scaler goes inside, and you pass raw data.

```python
from sklearn.pipeline import Pipeline

pipeline_svr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVR())
])

pipeline_svr.fit(X_train, y_train)      # raw X_train, NOT X_train_scaled
pipeline_svr.predict(X_test)
```

---

## 29. Plotting Actual vs Predicted

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].plot(y_test.values, label='actual')        # .values -> drop the shuffled index
ax[0].plot(best_svr.predict(X_test_scaled), label='prediksi')
ax[0].legend()
ax[0].set_title('SVR')

ax[1].plot(y_test.values, label='actual')
ax[1].plot(best_rfr.predict(X_test_scaled), label='prediksi')
ax[1].legend()
ax[1].set_title('Random Forest')
```

The closer the two lines follow each other, the better the model.

> 💡 A **scatter plot** is often easier to read: `plt.scatter(y_test, y_pred)` plus the line `plt.plot([5, 50], [5, 50])`. Perfect predictions sit on that diagonal line.

---

## 30. Regression Workflow Summary

```
1. Load data             pd.read_csv(..., header=None)
2. Split X and y         X = df.iloc[:, :-1], y = df.iloc[:, -1]
3. Train-test split      train_test_split(...)   (no stratify)
4. Scale                 fit on TRAIN only, transform test   (or use a Pipeline)
5. Baseline              LinearRegression -> MAE / RMSE / R²
6. Tune                  GridSearchCV + KFold, scoring='r2'
7. Compare models        MAE / RMSE / R² on the test set
8. Visualize             actual vs predicted
```

---

# BONUS

## 31. Time-Series & Market Data

Market data (stocks, forex, crypto) is a very different game from the datasets above.

| Adult / Housing | Market data |
|---|---|
| Strong signal | Very weak signal: mostly noise |
| Patterns stay the same over time | Patterns change over time |
| Nobody works against your model | Other traders trade obvious patterns away |
| 87% accuracy is realistic | **52–55%** direction accuracy is already good, and may still lose money |

### "Priced in" (efficient market hypothesis)

If a pattern reliably predicted "price goes up tomorrow", traders would buy **today**, pushing the price up now, and the pattern disappears. Anything the market already knows or expects is **priced in**.

- **Only surprises move prices**: the gap between what was expected and what happened.
- Example: analysts expect $1.00 profit per share, the company reports $1.05, and the stock **falls** because traders quietly expected $1.10. ("Buy the rumor, sell the news.")
- To predict price moves, a model has to predict surprises, which by definition the market didn't see coming. Public data (past prices, headlines) is already largely priced in.

### What is (somewhat) predictable

| | Predictability |
|---|---|
| Price direction | Very low (a coin flip plus a tiny edge, if any) |
| **Volatility** (how much prices move) | Moderate: calm stays calm, wild stays wild (GARCH models) |
| Volume, time-of-day patterns | Moderate |
| Momentum / value effects | Real but weak, and can stop working for years |

> **Predictable ≠ profitable.** A real edge can be too small to survive spread, commission and slippage.

### The techniques above that would fool you here

1. **A shuffled `train_test_split` leaks the future.** It trains on 2024 and tests on 2023. Split **by time**: train on the past, test on what comes after. Use `TimeSeriesSplit` instead of `KFold` / `StratifiedKFold`.
2. **Look-ahead bias.** Any feature that uses future information (e.g. today's close to predict today) makes the model look like a genius. The fit-on-train rule applies to every rolling feature too.
3. **Accuracy is the wrong score.** 55% accuracy can still lose money (small wins, big losses, costs). Measure **profit after costs**, drawdown, and out-of-sample results.
4. **Testing too many ideas.** Try 200 variations and a few will look great by chance. Keep one final test period that you only use once.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)   # each fold trains on the past and validates on the period after it
# data must be sorted by time first; never shuffle
```

### Model choice

- **Gradient boosting** (LightGBM / XGBoost) on hand-made features (returns, volatility, moving-average gaps): the common practical baseline.
- **Linear / logistic regression**: surprisingly competitive, because simple models overfit noise less.
- **LSTM / Transformer** on price alone: popular in tutorials, but rarely beats simpler models. They make more sense with unusual inputs (news text, order-book data).

> The model is maybe 10% of the problem. The other 90% is honest testing: split by time, avoid leakage, include costs, and test out of sample. Most strategies that look profitable in a backtest fail with real money.
