import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#read data

DATA_PATH = 'C:\\Users\\sasha\\OneDrive\\Documents\\GitHub\\ETL_hospital\\hospital data analysis.csv'

df = pd.read_csv(DATA_PATH)

# prepare features and target variables

FEATURES = ['Age', 'Gender', 'Condition', 'Procedure']
X = df[FEATURES].copy()
y_cost = df['Cost'].copy()
y_stay = df['Length_of_Stay'].copy()

# define categorical and numerical features
categorical_features = ['Gender', 'Condition']
numerical_features = ['Age']
random_state = 42

#We will use one-hot encoding for categorical features and leave numerical features as they are.

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features), 
                  ('num', 'passthrough', numerical_features)])

#train models - linear regression and random forest

def train_model():

    return {
        "Linear Regression": Pipeline([("preprocessor", preprocessor), 
                                       ("model", LinearRegression())]),

        "Random Forest": Pipeline([("preprocessor", preprocessor), 
                                   ("model", RandomForestRegressor(n_estimators=100, random_state=random_state))]),
    }

#evaluate models using MAE, MSE, RMSE, and R2 score

def evaluate_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

# target_run function to run the models on the target variable and return the results
def target_run(target_variable, model_name):
    # Split the data into training and testing sets 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, target_variable, test_size=0.2, random_state=random_state)
    # Evaluate the baseline model (mean of the target variable)
    results = {}
    baseline_predict = np.full_like(y_test, fill_value=y_train.mean(), dtype=np.float64)
    results["Mean of the baseline"] = evaluate_model(y_test, baseline_predict)

    fitted = {}

    # Train and evaluate each model
    for name, pipeline in train_model().items():
        # Fit the model on the training data
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        results[name] = evaluate_model(y_test, y_pred)
        # Store the fitted model for later use
        fitted[name] = pipeline

        # Perform cross-validation and store the results 
        cv_results = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        results[name + " CV R2"] = cv_results.mean()
        results[name + " CV R2 Std"] = cv_results.std()

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(results).T
    print(results_df.round(2).to_string())
    # Return the results DataFrame, fitted models, and the train-test split data
    return results_df, fitted, (X_train, X_test, y_train, y_test)

#________________________________________________________
# Run the target_run function for both target variables and save the results to CSV files
print("Running models for Cost prediction...")
cost_results, cost_models, cost_split = target_run(y_cost, "Cost")
print("Running models for Length of Stay prediction...")
stay_results, stay_models, stay_split = target_run(y_stay, "Length of Stay")

cost_results.to_csv('cost_results.csv', index=True)
stay_results.to_csv('stay_results.csv', index=True)

