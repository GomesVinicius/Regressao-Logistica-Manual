import numpy as np
from LogisticReg import LogisticRegression

y = np.array( [0, 1, 1, 1, 0, 0, 0] )
X = np.array( [
                [30, 7,  1, 1.5 , 2],
                [40, 10, 0, 1.6 , 4],
                [50, 20, 1, 1.7 , 7],
                [60, 12, 0, 1.8 , 7],
                [25, 10, 1, 1.65, 8],
                [24, 12, 0, 1.9 , 9],
                [32, 22, 1, 1.62, 0],
               ] )

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_standarlized = (X-mean)/std

model = LogisticRegression()
model.fit(X_standarlized, y)

print(model.intercept, model.theta)

new_x = np.array([
        [35, 45, 0, 1.78, 4]
    ])

new_x_std = (new_x-mean)/std
print(model.predict_proba(new_x_std))
print(model.predict(new_x_std))
