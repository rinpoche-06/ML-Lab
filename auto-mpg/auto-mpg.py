import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


# Load Auto MPG dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url)

# Preprocessing
df = df[['displacement', 'mpg']]
df = df.dropna()

# Input and output
X = df[['displacement']]
y = df['mpg']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# Linear Regression
# -------------------------------

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)


# -------------------------------
# Polynomial Regression
# -------------------------------

degrees = [2, 3, 4]

results = []

results.append([
    "Linear",
    1,
    linear_mse,
    linear_r2
])

polynomial_models = {}

for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    polynomial_models[degree] = model

    results.append([
        "Polynomial",
        degree,
        mse,
        r2
    ])


# -------------------------------
# Display Results
# -------------------------------

results_df = pd.DataFrame(
    results,
    columns=["Model", "Degree", "MSE", "R-Squared"]
)

print("\nModel Comparison:")
print(results_df.to_string(index=False))


# -------------------------------
# Visualization
# -------------------------------

X_plot = np.linspace(
    X['displacement'].min(),
    X['displacement'].max(),
    300
).reshape(-1, 1)

plt.figure(figsize=(10, 6))

# Actual data
plt.scatter(
    X,
    y,
    alpha=0.5,
    label="Actual Data"
)

# Linear regression
plt.plot(
    X_plot,
    linear_model.predict(X_plot),
    linewidth=2,
    label="Linear Regression"
)

# Polynomial regression
for degree, model in polynomial_models.items():

    plt.plot(
        X_plot,
        model.predict(X_plot),
        linewidth=2,
        label=f"Polynomial Degree {degree}"
    )

plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)

plt.show()