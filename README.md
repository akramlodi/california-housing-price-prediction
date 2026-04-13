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
	- RandomForest — rmse_cv=0.511520, rmse_test=0.505143, mae_test=0.327425, r2=0.805275
	- GradientBoosting — rmse_cv=0.534001, rmse_test=0.542217, mae_test=0.371650, r2=0.775643
	- SVR — rmse_cv=0.592471, rmse_test=0.597498, mae_test=0.398599, r2=0.727563
	- Ridge — rmse_cv=0.720528, rmse_test=0.745557, mae_test=0.533193, r2=0.575816
	- LinearRegression — rmse_cv=0.720527, rmse_test=0.745581, mae_test=0.533200, r2=0.575788

- **Conclusions & next steps:**
	- Tree-based ensembles (RandomForest, GradientBoosting) outperform linear models, indicating nonlinear relationships and interactions.
	- Next: hyperparameter tuning, feature engineering (spatial/interaction features), and ensembling/stacking to further reduce error.
