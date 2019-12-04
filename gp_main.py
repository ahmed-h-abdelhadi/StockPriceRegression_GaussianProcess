import GPy
import pandas as pd
import numpy as np

# Our test grid
[Xi, Xj] = np.meshgrid(np.linspace(-np.pi, np.pi, 50), np.linspace(-np.pi, np.pi, 50))

# Number of samples [YOU CAN PLAY AROUND WITH THE NUMBER OF RANDOM SAMPLES TO SEE HOW THE FIT IS AFFECTED]
num_measurements = 50
f = lambda xi,xj: np.sin(xi) * np.sin(xj)
# Random sample locations (2-D)
X2 = np.random.uniform(-np.pi, np.pi, (num_measurements, 2))
Y2 = np.array([f(x1,x2) for (x1,x2) in zip(X2[:,0], X2[:,1])])[:,None] + 0.05 * np.random.randn(X2.shape[0], 1)
k = GPy.kern.RBF(2, active_dims=[0,1])

# Create a GP Regression model with the sample locations and observations using the RBF kernel
m = GPy.models.GPRegression(X2, Y2, k)

# Optimise the kernel parameters
m.optimize()

Xnew2 = np.vstack((Xi.ravel(), Xj.ravel())).T

mean2, Cov2 = m.predict(Xnew2)


n_dimensions = 2 * 5
n_training_points = 300
n_test_points = 300

dataset = pd.read_csv("./data/final_dataset.csv", sep='\t', encoding='utf-8')

dataset = dataset.as_matrix()
X = dataset[:n_training_points,:-1].reshape(-1, n_dimensions)
Y = dataset[:n_training_points,-1].reshape(-1, 1)
# Create a 2-D RBF kernel active over both input dimensions
k = GPy.kern.RBF(n_dimensions, active_dims=range(n_dimensions))

# Create a GP Regression model with the sample locations and observations using the RBF kernel
m = GPy.models.GPRegression(X, Y, k)

m.optimize()

X_new = dataset[n_training_points: n_training_points + n_test_points,:-1].reshape(-1, n_dimensions)
Y_new = dataset[n_training_points: n_training_points + n_test_points, -1].reshape(-1, 1)
mean, cov = m.predict(X_new)
print("DONE!")