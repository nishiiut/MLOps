import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import combinations

data = pd.read_csv('../data/sampregdata.csv', index_col=0)
features = ['x1', 'x2', 'x3', 'x4']
target = 'y'

best_pair = None
best_r2 = -float('inf')
best_model = None
best_predictions = None

# Try all combinations of two features, train a linear regression, and pick the best based on R²
for combo in combinations(features, 2):
    X = data[list(combo)]
    y = data[target]
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    print(f"Features: {combo}, R^2: {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_pair = combo
        best_model = model
        best_predictions = predictions

print(f"\nSelected best feature pair: {best_pair} with R^2: {best_r2:.4f}")

# Create a scatter plot comparing the actual vs predicted values.
plt.figure(figsize=(8, 6))
plt.scatter(data[target], best_predictions, color='green', alpha=0.5, label='Predicted vs Actual')
min_val = min(data[target].min(), best_predictions.min())
max_val = max(data[target].max(), best_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', lw=2, linestyle='--', label='Ideal Fit')
plt.title(f'Linear Regression (Two Features: {best_pair})')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.legend()
plt.grid(True)
# Save the plot in the models folder
plt.savefig('../models/two_feature_model_plot.png')
plt.show()

