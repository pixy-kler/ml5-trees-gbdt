# ml5-trees-gbdt — Decision Trees & GBDT from Scratch 🎓

**ML Course, Module 5** · Don'tGetKicked (`IsBadBuy`) · time-based split · custom CART / ExtraTree / GBDT

## 📌 What it's about

Binary classification of "lemon" cars at auction (12.3% bad buys).
Time-based split by `PurchDate` (1/3–1/3–1/3), leak-free categorical encoding,
custom `Node` / `DecisionTreeClassifier` / `DecisionTreeRegressor` / `find_best_split` (vectorized),
interpretability (prediction path, feature importance), and comparison with sklearn / LightGBM / XGBoost / CatBoost.

**Metric:** Gini = `2 · ROC AUC − 1`.

## 🛠 Stack

Python · NumPy · Pandas · scikit-learn · LightGBM · XGBoost · CatBoost · category_encoders · Matplotlib/Seaborn · OOP · PyTest

## 📊 Results (Gini)

| Model | train | valid | test | fit, s |
|---|---|---|---|---|
| Custom CART (max_depth=7) | 0.5815 | 0.3907 | 0.3061 | 0.31 |
| Custom ExtraTree (n_thr=5) | — | 0.4401 | 0.2992 | 0.05 |
| sklearn CART (max_depth=7) | 0.5293 | 0.4011 | — | 0.13 |
| RandomForest (n=100) | 0.5875 | 0.4630 | 0.4358 | 5.87 |
| Custom GBDT (n=100) | — | 0.4654 | 0.4790 | 11.32 |
| ExtraTreesClassifier (n=100) | — | 0.4797 | 0.4072 | 7.71 |
| LightGBM | — | 0.4862 | 0.4843 | 0.38 |
| XGBoost | — | 0.4918 | 0.4951 | 2.07 |
| **CatBoost** | 0.6185 | **0.4928** | **0.4957** | 1.00 |

_— means the metric was not computed (train for ensembles/boosting, test for sklearn CART)._

**Key findings:**

- **CatBoost** is the best model on both valid (0.4928) and test (0.4957) and is stable: `valid → test = −0.0029`.
- Single trees overfit: train/valid gap up to 0.19 for Custom CART and 0.13 for sklearn CART.
- **Custom GBDT** is close to the libraries in quality (valid 0.4654, test 0.4790) but ~30× slower than LightGBM (11.3 s vs 0.38 s).
- Custom ExtraTree and Custom CART drop sharply on test (−0.14 and −0.08) — unstable.

## 🗂 Pipeline

- EDA + time-based split by `PurchDate`: train 24 084 / valid 24 084 / test 24 815.
- Leak-free preprocessing: medians/modes fit on train, `CountEncoder` for high-cardinality features, `OneHotEncoder` for low-cardinality.
- Custom `Node` / `DecisionTreeClassifier` / `DecisionTreeRegressor` / `find_best_split` (gini and mse branches, vectorized via cumsum).
- `gini_score = 2 · roc_auc_score − 1` + sanity check on synthetic data (1.0 / 0.0 / −1.0).
- Interpretability: `plot_custom_tree`, `custom_feature_importance`, `explain_prediction`.
- Final comparison table across all models (custom + sklearn + LightGBM / XGBoost / CatBoost).

## 🧠 Lessons learned

- `tree_clf` was initially overfitted as a classifier (the tree had been trained as a regressor) — fixed by retraining `DecisionTreeClassifier(max_depth=7, random_state=21)`.
- Single trees tend to overfit; ensembles and boosting give a noticeable Gini boost.
- `CountEncoder` for high-cardinality features reduces dimensionality while preserving signal (count-based features such as `Model_count` rank high in importance).

## ▶️ How to run

```bash
pip install -r requirements.txt
jupyter notebook ML5_DecisionTrees.ipynb
```

> The final comparison block is wrapped in `timeout(120)` to guard against excessively long runs.
