import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.intercept = None
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        Xb = np.c_[np.ones( (X.shape[0], 1)), X]
        m, n = Xb.shape
        
        self.theta = np.zeros(n)

        for _ in range(self.iterations):
            predictions = self.sigmoid(Xb.dot(self.theta))
            
            errors = predictions - y

            gradient = (1/m) * Xb.T.dot(errors)
            self.theta -= self.learning_rate * gradient

        self.intercept = self.theta[0]
        self.theta = self.theta[1:]

    def predict_proba(self, X):
        Xb = np.c_[np.ones( (X.shape[0], 1)), X]
        return self.sigmoid(Xb.dot(np.r_[self.intercept, self.theta]))

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)