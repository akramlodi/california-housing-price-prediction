# This is a Machine learning project based on the california housing dataset.
# California Housing Price Prediction

- **Dataset:** California Housing dataset (median house values and features such as median income, housing median age, total rooms, total bedrooms, population, households, latitude, longitude).

- **Data exploration (EDA):**
	- Checked distributions, summary statistics and missing values.
	- Visualized feature correlations and pairwise relationships to identify important predictors and potential multicollinearity.

- **Preprocessing:**
	- Dropped target column from features and split data into train/test (80/20, random_state=42).
	- Standard scaled features using `StandardScaler` inside model pipelines.
	- No imputation required (dataset complete from sklearn loader).

- **Models trained & evaluation:**
	- Trained: Linear Regression, Ridge, SVR, RandomForest, GradientBoosting.
	- Metrics reported: cross-validated RMSE (train), RMSE, MAE, R² (test).

- **Final results (sorted by test RMSE):**

	| Model | rmse_cv | rmse_test | mae_test | r2 |
	|---|---:|---:|---:|---:|
	| RandomForest | 0.511520 | 0.505143 | 0.327425 | 0.805275 |
	| GradientBoosting | 0.534001 | 0.542217 | 0.371650 | 0.775643 |
	| SVR | 0.592471 | 0.597498 | 0.398599 | 0.727563 |
	| Ridge | 0.720528 | 0.745557 | 0.533193 | 0.575816 |
	| LinearRegression | 0.720527 | 0.745581 | 0.533200 | 0.575788 |

- **Conclusions & next steps:**
	- Tree-based ensembles (RandomForest, GradientBoosting) outperform linear models, indicating nonlinear relationships and interactions.
	- Next: hyperparameter tuning, feature engineering (spatial/interaction features), and ensembling/stacking to further reduce error.
