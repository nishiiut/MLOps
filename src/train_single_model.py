import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('../data/sampregdata.csv', index_col=0)
features = ['x1', 'x2', 'x3', 'x4']
target = 'y'

best_feature = None
best_r2 = -float('inf')
best_model = None
best_predictions = None

# Try each single feature, train a linear regression, and pick the best based on R²
for feature in features:
    X = data[[feature]]
    y = data[target]
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    print(f"Feature: {feature}, R^2: {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_feature = feature
        best_model = model
        best_predictions = predictions

print(f"\nSelected best feature: {best_feature} with R^2: {best_r2:.4f}")

# Create a scatter plot comparing the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(data[best_feature], data[target], color='blue', alpha=0.5, label='Actual y')
plt.scatter(data[best_feature], best_predictions, color='red', alpha=0.5, label='Predicted y')
plt.title(f'Linear Regression (Single Feature: {best_feature})')
plt.xlabel(best_feature)
plt.ylabel(target)
plt.legend()
plt.grid(True)
# Save the plot in the models folder
plt.savefig('../models/single_feature_model_plot.png')
plt.show()

