import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 1. Distributions
n, p = 10, 0.5
binomial_data = stats.binom.rvs(n, p, size=1000)
gaussian_data = stats.norm.rvs(loc=0, scale=1, size=1000)

# 2. Bayes' Theorem: P(Disease | Positive)
# P(D) = 0.01, P(+|D) = 0.95, P(+|~D) = 0.10
p_d = 0.01
p_pos_given_d = 0.95
p_pos_given_not_d = 0.10

p_pos = (p_pos_given_d * p_d) + (p_pos_given_not_d * (1 - p_d))
p_d_given_pos = (p_pos_given_d * p_d) / p_pos
print(f"Analytical P(Disease | Positive): {p_d_given_pos:.4f}")

# 3. Monte Carlo Simulation
N = 100000
has_disease = np.random.rand(N) < p_d
test_positive = np.zeros(N, dtype=bool)
test_positive[has_disease] = np.random.rand(np.sum(has_disease)) < p_pos_given_d
test_positive[~has_disease] = np.random.rand(np.sum(~has_disease)) < p_pos_given_not_d

sim_p = np.sum(has_disease & test_positive) / np.sum(test_positive)
print(f"Simulated P(Disease | Positive): {sim_p:.4f}")

# 4. Posterior curve over prevalence variation
prevalences = np.linspace(0.001, 0.20, 100)
posteriors = [(p_pos_given_d * prev) / ((p_pos_given_d * prev) + (p_pos_given_not_d * (1 - prev))) for prev in prevalences]

plt.plot(prevalences, posteriors)
plt.title("Posterior vs Prevalence")
plt.xlabel("Prevalence")
plt.ylabel("P(Disease | Positive)")
plt.savefig("exp3_bayes.png")
plt.close()
