
# 1D Ridge Regression from Scratch, using the analytic solution, nothing fancy
# And I am going to use the california data from sklearn
# why 1D? because the visualization can be shown nicely then

# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# import RidgeBeta function
from RidgeBeta import Ridge

# fetch data
data = fetch_california_housing()
X, y = data.data, data.target
print(type(X), type(y)) # they are given via numpy arrays! (NOT pandas dataframes!!)
print(data.target_names)
print(data.feature_names)

print(X.shape, y.shape)
y = y.reshape(-1, 1)
print(X.shape, y.shape)

coeff = Ridge(X, y, 525552)
print("Coefficients:")
print(coeff)
print("Mean Squared Loss:", np.mean((y - np.hstack([np.ones((X.shape[0], 1)), X])@coeff)**2))

# NEW: plotting using subplots!
fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns
axes = axes.flatten()  # flatten to index easily

for i in range(len(data.feature_names)):
    ax = axes[i]
    x_vals = np.linspace(X[:, i].min(), X[:, i].max(), 100)
    y_vals = coeff[0] + coeff[i+1] * x_vals  # coeff[0] is intercept

    ax.scatter(X[:, i], y, alpha=0.5)
    ax.plot(x_vals, y_vals, color='red', linewidth=1)
    ax.set_xlabel(data.feature_names[i])
    ax.set_ylabel('MedHouseVal')
    ax.set_title(f'Feature: {data.feature_names[i]}')

plt.tight_layout()
plt.show()