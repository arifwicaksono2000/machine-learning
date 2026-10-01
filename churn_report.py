"""
Telco Customer Churn - experiment report.

Runs the full experiment (clean -> features -> split -> GridSearchCV for each model)
and writes a printable HTML report to reports/churn_report.html.
Open the HTML file in a browser and press Ctrl+P to print or save as PDF.

Run:  .venv\Scripts\python.exe churn_report.py
"""
import html
import os
import time
from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, 'datasets', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
REPORT_PATH = os.path.join(HERE, 'reports', 'churn_report.html')

# ---------------------------------------------------------------- setup
feature = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
           'Contract', 'MonthlyCharges', 'TotalCharges']
col_num = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
col_cat = ['gender', 'Partner', 'Dependents', 'Contract']

preprocessing = ColumnTransformer([
    ('num', StandardScaler(), col_num),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), col_cat)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'KNN': (KNeighborsClassifier(), {
        'model__n_neighbors': [5, 15, 25],
        'model__weights': ['uniform', 'distance']
    }),
    'Decision Tree': (DecisionTreeClassifier(random_state=42, class_weight='balanced'), {
        'model__max_depth': [3, 5, 10, None],
        'model__min_samples_leaf': [1, 5, 20]
    }),
    'Random Forest': (RandomForestClassifier(random_state=42, class_weight='balanced'), {
        'model__n_estimators': [100, 200],
        'model__max_depth': [5, 10, None],
        'model__min_samples_leaf': [1, 5]
    }),
}
prefix = {'KNN': 'KNN', 'Decision Tree': 'DT', 'Random Forest': 'RF'}
scoring = ['accuracy', 'precision', 'recall', 'f1']


def fmt_params(params):
    return ', '.join(f"{k.replace('model__', '')}={v}" for k, v in params.items())


# ---------------------------------------------------------------- data
df = pd.read_csv(DATA_PATH)
n_raw = len(df)
n_blank = int((df['TotalCharges'].astype(str).str.strip() == '').sum())

df = df.drop(columns='customerID')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

X = df[feature]                               # after cleaning, so X gets the numeric TotalCharges
y = df['Churn'].map({'No': 0, 'Yes': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------------- experiments
all_runs, test_rows, best_models = [], [], {}

for name, (model, params) in models.items():
    print(f'Training {name} ...')
    pipe = Pipeline([('preprocessing', preprocessing), ('model', model)])
    grid = GridSearchCV(pipe, params, cv=cv, scoring=scoring, refit='f1', n_jobs=-1)

    start = time.time()
    grid.fit(X_train, y_train)
    search_time = time.time() - start
    best_models[name] = grid.best_estimator_

    cvr = pd.DataFrame(grid.cv_results_)
    for i, row in cvr.iterrows():
        all_runs.append({
            'Run ID': f'{prefix[name]}-{i + 1:02d}',
            'Model': name,
            'Parameters': fmt_params(row['params']),
            'Accuracy': row['mean_test_accuracy'],
            'Precision': row['mean_test_precision'],
            'Recall': row['mean_test_recall'],
            'F1': row['mean_test_f1'],
            'Fit Time (s)': row['mean_fit_time'],
            'Best': '★' if i == grid.best_index_ else '',
        })

    start = time.time()
    y_pred = grid.best_estimator_.predict(X_test)
    pred_time = time.time() - start

    test_rows.append({
        'Model': name,
        'Best Run': f'{prefix[name]}-{grid.best_index_ + 1:02d}',
        'Best Parameters': fmt_params(grid.best_params_),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'Search Time (s)': search_time,
        'Predict Time (s)': pred_time,
    })

runs = pd.DataFrame(all_runs).round(3)
results = pd.DataFrame(test_rows).sort_values('F1', ascending=False).round(3)

best_name = results.iloc[0]['Model']
best_pred = best_models[best_name].predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, best_pred).ravel()
report_text = classification_report(y_test, best_pred, target_names=['Stay (0)', 'Churn (1)'])

# ---------------------------------------------------------------- console output
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 60)
print('\n=== All runs (5-fold CV on train) ===')
print(runs.to_string(index=False))
print('\n=== Final comparison (test set) ===')
print(results.to_string(index=False))
print(f'\nBest model: {best_name}\n')
print(report_text)

# ---------------------------------------------------------------- HTML report
def table(df_):
    return df_.to_html(index=False, escape=True, border=0, classes='t')


config_rows = pd.DataFrame([
    {'Model': n,
     'Important Parameters': '; '.join(f"{k.replace('model__', '')} = {v}" for k, v in p.items())
                             + ('; class_weight = balanced' if 'class_weight' in m.get_params()
                                and m.get_params()['class_weight'] == 'balanced' else '')}
    for n, (m, p) in models.items()
])

fe_rows = pd.DataFrame([
    {'No': 1, 'Operation': 'Drop column', 'Columns': 'customerID',
     'Reason': 'Unique ID per customer, no predictive value'},
    {'No': 2, 'Operation': 'Data type conversion (text to number)', 'Columns': 'TotalCharges',
     'Reason': "Stored as text; pd.to_numeric(errors='coerce')"},
    {'No': 3, 'Operation': 'Missing value imputation (fill with 0)', 'Columns': 'TotalCharges',
     'Reason': f'{n_blank} blank values, all with tenure = 0 (new customers not yet billed)'},
    {'No': 4, 'Operation': 'Feature selection', 'Columns': ', '.join(feature),
     'Reason': 'Columns relevant to churn'},
    {'No': 5, 'Operation': 'Target encoding (label mapping)', 'Columns': 'Churn',
     'Reason': 'No = 0, Yes = 1'},
    {'No': 6, 'Operation': 'One-hot encoding', 'Columns': ', '.join(col_cat),
     'Reason': "Categories have no order; handle_unknown='ignore'"},
    {'No': 7, 'Operation': 'Standard scaling', 'Columns': ', '.join(col_num),
     'Reason': 'Same scale for all numeric features (needed for KNN)'},
])

churn_rate = y.mean()
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Churn Experiment Report</title>
<style>
  @page {{ size: A4 landscape; margin: 14mm; }}
  body {{ font-family: Segoe UI, Arial, sans-serif; font-size: 11px; color: #111; background: #fff;
         max-width: 1100px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  h2 {{ font-size: 14px; margin: 22px 0 6px; border-bottom: 2px solid #333; padding-bottom: 3px; }}
  .meta {{ color: #555; margin-bottom: 10px; }}
  table.t {{ border-collapse: collapse; width: 100%; margin-bottom: 6px; }}
  table.t th, table.t td {{ border: 1px solid #999; padding: 3px 6px; text-align: left; }}
  table.t th {{ background: #e8e8e8; }}
  table.t tr:nth-child(even) td {{ background: #f7f7f7; }}
  table.t tr {{ page-break-inside: avoid; }}
  table.t thead {{ display: table-header-group; }}
  pre {{ background: #f4f4f4; border: 1px solid #ccc; padding: 8px; font-size: 11px; }}
  .cm td, .cm th {{ text-align: center !important; }}
  .note {{ color: #444; }}
  .break {{ page-break-before: always; }}
</style></head><body>

<h1>Telco Customer Churn: Experiment Report</h1>
<div class="meta">Generated {datetime.now():%Y-%m-%d %H:%M} &middot; Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv</div>

<h2>1. Dataset</h2>
<p>{n_raw:,} customers, {len(feature)} features used. Churn rate: {churn_rate:.1%}
(imbalanced). Train / test split: {len(X_train):,} / {len(X_test):,} rows (80/20, stratified, random_state=42).
Cross-validation: 5-fold StratifiedKFold. Model selection metric: F1 (churn class).</p>

<h2>2. Feature Engineering</h2>
{table(fe_rows)}

<h2>3. Model Configuration</h2>
{table(config_rows)}

<h2>4. Final Comparison (Test Set)</h2>
{table(results)}
<p class="note">Best model: <b>{html.escape(best_name)}</b>. Metrics are for the churn class (1).
Search time = full GridSearchCV time for that model.</p>

<h2>5. Best Model Detail: {html.escape(best_name)}</h2>
<table class="t cm" style="width:auto">
  <tr><th></th><th>Predicted Stay</th><th>Predicted Churn</th></tr>
  <tr><th>Actual Stay</th><td>{tn} (correct)</td><td>{fp} (false alarm)</td></tr>
  <tr><th>Actual Churn</th><td>{fn} (missed)</td><td>{tp} (caught)</td></tr>
</table>
<pre>{html.escape(report_text)}</pre>

<h2 class="break">6. Experiment Results: All Runs</h2>
<p class="note">Each row is one parameter combination (run). Metrics are 5-fold cross-validation averages
on the training set; Fit Time is the average per fold. &#9733; = best run of that model (by F1).</p>
{table(runs)}

</body></html>"""

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(page)
print(f'Report saved to {REPORT_PATH}')
