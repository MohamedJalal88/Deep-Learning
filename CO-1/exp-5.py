import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

np.random.seed(42)
X = np.sort(np.random.rand(150) * 5)[:, np.newaxis]
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

degrees = [1, 3, 9, 15]
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', alpha=0.3, label='Data')

for deg in degrees:
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.plot(X, y_pred, label=f'Degree {deg}')

plt.legend()
plt.title("Polynomial Regression Models")
plt.savefig("exp5_bias_variance.png")
plt.close()
