import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)
W1 = np.random.uniform(size=(2, 4))
W2 = np.random.uniform(size=(4, 1))

for _ in range(10000):
    # Forward Pass
    h_input = np.dot(X, W1)
    h_out = sigmoid(h_input)
    o_input = np.dot(h_out, W2)
    predicted_output = sigmoid(o_input)

    # Backpropagation
    error = y - predicted_output
    d_predicted_output = error * sigmoid_deriv(predicted_output)
    
    error_hidden = d_predicted_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_deriv(h_out)

    # Weights update
    W2 += h_out.T.dot(d_predicted_output) * 0.1
    W1 += X.T.dot(d_hidden) * 0.1

print("Predictions after training:\n", predicted_output)
