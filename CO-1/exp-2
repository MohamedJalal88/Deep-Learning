import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generate synthetic data (200 records, 5 blood parameters)
np.random.seed(42)
raw_data = np.random.multivariate_normal(
    mean=[14, 5, 100, 200, 25],
    cov=np.eye(5) + 0.5,
    size=200
)
df = pd.DataFrame(raw_data, columns=['Hb', 'RBC', 'Glucose', 'Cholesterol', 'BMI'])
df_scaled = (df - df.mean()) / df.std()

# 2. Covariance and Correlation matrices
cov_mat = df_scaled.cov()
corr_mat = df_scaled.corr()

# 3. Eigenvalues and Eigenvectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
idx = np.argsort(eig_vals)[::-1]
eig_vals, eig_vecs = eig_vals[idx], eig_vecs[:, idx]

# 4. 2-D plot with eigenvectors
data_2d = df_scaled.iloc[:, :2].values
cov_2d = np.cov(data_2d.T)
vals_2d, vecs_2d = np.linalg.eig(cov_2d)

plt.figure(figsize=(6, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5)
for val, vec in zip(vals_2d, vecs_2d.T):
    plt.quiver(0, 0, vec[0]*val, vec[1]*val, scale=5, color='r')
plt.title("2D Projection & Principal Axes")
plt.savefig("exp2_eigenvectors.png")
plt.close()

# 5. Variance explained
var_exp = [(i / sum(eig_vals)) * 100 for i in eig_vals]
for idx, var in enumerate(var_exp):
    print(f"PC {idx+1}: {var:.2f}% variance")
