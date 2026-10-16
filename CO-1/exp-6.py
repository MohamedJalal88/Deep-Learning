import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Loss function: f(x, y) = x^2 + 10*y^2
def grad(x, y):
    return np.array([2*x, 20*y])

# Batch GD vs Momentum
lr = 0.08
x_gd, y_gd = 5.0, 5.0
path_gd = [(x_gd, y_gd)]

for _ in range(20):
    g = grad(x_gd, y_gd)
    x_gd -= lr * g[0]
    y_gd -= lr * g[1]
    path_gd.append((x_gd, y_gd))

# Keras training setup
X = np.random.randn(1000, 4)
y = 3*X[:, 0] - 2*X[:, 1] + 0.5*X[:, 2] + np.random.randn(1000)*0.1

def build_model(optimizer):
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(4,))])
    model.compile(optimizer=optimizer, loss='mse')
    return model

h_sgd = build_model(tf.keras.optimizers.SGD()).fit(X, y, epochs=20, verbose=0)
h_mom = build_model(tf.keras.optimizers.SGD(momentum=0.9)).fit(X, y, epochs=20, verbose=0)
h_adam = build_model(tf.keras.optimizers.Adam()).fit(X, y, epochs=20, verbose=0)

plt.plot(h_sgd.history['loss'], label='SGD')
plt.plot(h_mom.history['loss'], label='SGD+Momentum')
plt.plot(h_adam.history['loss'], label='Adam')
plt.legend()
plt.title("Optimizer Convergence")
plt.savefig("exp6_optimizers.png")
plt.close()
