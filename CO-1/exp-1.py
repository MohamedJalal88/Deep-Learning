import numpy as np
import matplotlib.pyplot as plt

# 1. Customer-Genre (6x5) and Genre-Book (5x4) matrices
np.random.seed(42)
R_cg = np.random.randint(1, 6, size=(6, 5))
W_gb = np.random.uniform(0.1, 1.0, size=(5, 4))

# 2. Matrix multiplication and Transpose
R_cb = np.dot(R_cg, W_gb)
R_cg_T = R_cg.T

# Inverse where valid (5x5 square matrix example)
square_mat = np.dot(R_cg.T, R_cg)
inv_mat = np.linalg.inv(square_mat)

# 3. Cosine similarity between customers
norms = np.linalg.norm(R_cg, axis=1, keepdims=True)
R_norm = R_cg / norms
cos_sim = np.dot(R_norm, R_norm.T)

# Identify two most similar customers (excluding diagonal)
np.fill_diagonal(cos_sim, -1)
i, j = np.unravel_index(np.argmax(cos_sim), cos_sim.shape)
print(f"Most similar customers: {i} and {j} with similarity {cos_sim[i, j]:.4f}")

# 4. Dense layer: y = W*x + b for customer 0
x = R_cg[0]
W = W_gb.T
b = np.ones(4)
y_manual = np.dot(W, x) + b

# 5. Visualise score matrix
plt.figure(figsize=(6, 4))
plt.imshow(R_cb, cmap="viridis")
plt.colorbar(label="Score")
plt.title("Customer-Book Scores")
plt.xlabel("Book ID")
plt.ylabel("Customer ID")
plt.savefig("exp1_score_heatmap.png")
plt.close()
