import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Observed distribution P, Model predictions Q1 and Q2
P = np.array([0.6, 0.3, 0.1])
Q1 = np.array([0.5, 0.3, 0.2])
Q2 = np.array([0.2, 0.3, 0.5])

# 1. Entropy
H_P = entropy(P, base=np.e)

# 2. Cross-entropy & KL Divergence
kl_Q1 = entropy(P, Q1)
cross_entropy_Q1 = H_P + kl_Q1

print(f"Entropy(P): {H_P:.4f}")
print(f"KL(P || Q1): {kl_Q1:.4f}")
print(f"Cross Entropy(P, Q1): {cross_entropy_Q1:.4f}")

# 3. Asymmetry of KL Divergence
kl_P_Q1 = entropy(P, Q1)
kl_Q1_P = entropy(Q1, P)
print(f"KL(P || Q1) = {kl_P_Q1:.4f} vs KL(Q1 || P) = {kl_Q1_P:.4f}")

# 4. Binary Cross-Entropy plot
p_pred = np.linspace(0.01, 0.99, 100)
bce_loss = -np.log(p_pred)  # when true y = 1
plt.plot(p_pred, bce_loss)
plt.title("BCE Loss for True Label Y=1")
plt.xlabel("Predicted Probability")
plt.ylabel("Loss")
plt.savefig("exp4_bce.png")
plt.close()
