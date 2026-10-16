import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense

# Time series generation
t = np.linspace(0, 100, 1000)
series = np.sin(t)

# Create windowed datasets
window_size = 24
X, y = [], []
for i in range(len(series) - window_size):
    X.append(series[i:i+window_size])
    y.append(series[i+window_size])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = tf.keras.Sequential([
    SimpleRNN(20, activation='relu', input_shape=(24, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X[:700], y[:700], epochs=10, verbose=0)
print("SimpleRNN training completed.")
