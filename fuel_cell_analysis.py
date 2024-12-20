import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load and inspect the dataset
file_name = 'Filtered_Fuel_cell_performance_data.csv'
dataset = pd.read_csv(file_name)
print(dataset.info())
print(dataset.head())

# Set the target column and features
response_variable = 'Target4'  # Update with the actual column name
features = dataset.drop(columns=[response_variable])
target = dataset[response_variable]

# Partition the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

# Initialize models for evaluation
regressors = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Regressor": SVR()
}

# Train each model and compute performance metrics
model_evaluations = []

for model_name, model_instance in regressors.items():
    model_instance.fit(X_train, y_train)
    predictions = model_instance.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    model_evaluations.append({
        "Model": model_name,
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "R-squared (R2)": r2
    })

# Compile and display results
evaluation_results = pd.DataFrame(model_evaluations)
print(evaluation_results)

# Export results to a CSV file
output_file = 'model_evaluation_results.csv'
evaluation_results.to_csv(output_file, index=False)
print(f"Model performance metrics saved to '{output_file}'.")
