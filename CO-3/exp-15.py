import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

seq_len = 50
X = tf.random.normal((1, seq_len, 1))
cell = tf.keras.layers.SimpleRNNCell(1, activation='tanh')

with tf.GradientTape(persistent=True) as tape:
    state = tf.zeros((1, 1))
    states = []
    tape.watch(state)
    for t in range(seq_len):
        out, state = cell(X[:, t, :], [state])
        states.append(state)
    loss = states[-1]

grads = [tape.gradient(loss, s)[0].numpy()[0] for s in states]

plt.plot(grads)
plt.title("Gradient Norm wrt Hidden States over Time Steps")
plt.xlabel("Time Step")
plt.ylabel("Gradient")
plt.savefig("exp15_bptt.png")
plt.close()
