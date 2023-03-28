
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
car_data = pd.read_csv('CAR DETAILS.csv')

# Drop any missing values
car_data.dropna(inplace=True)

# Convert categorical features into numerical features
car_data = pd.get_dummies(car_data, drop_first=True)

# Split the data into train and test sets
X = car_data.drop('selling_price', axis=1)
y = car_data['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the regression models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42)
}

# Train and evaluate the models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    cv_score = np.mean(cross_val_score(model, X, y, cv=5))
    results[name] = {
        'Model': model,
        'RMSE': rmse,
        'R^2': r2,
        'CV Score': cv_score
    }

# Find the best model
best_model_name = max(results, key=lambda name: results[name]['R^2'])
best_model = results[best_model_name]['Model']

# Save the best model
joblib.dump(best_model, 'car_price_model.joblib')

# Load the model
loaded_model = joblib.load('car_price_model.joblib')

# Generate a new dataset for testing
test_data = car_data.sample(n=20, random_state=42)
X_test = test_data.drop('selling_price', axis=1)
y_test = test_data['selling_price']

# Test the model on the new dataset
y_pred = loaded_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'R^2: {r2}')
