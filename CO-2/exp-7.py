import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.w += update * xi
                self.b += update

    def predict(self, X):
        return np.where((np.dot(X, self.w) + self.b) >= 0, 1, 0)

# XOR Dataset (Non-linear failure check)
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([0, 1, 1, 0])

p = Perceptron()
p.fit(X_xor, y_xor)
preds = [p.predict(x) for x in X_xor]
print("XOR Perceptron Predictions (Fails to match):", preds)
