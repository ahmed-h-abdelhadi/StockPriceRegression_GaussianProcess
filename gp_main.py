import GPy
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


n_dimensions = 1 * 5
n_training_points = 120
n_test_points = 300

dataset = pd.read_csv("./data/final_dataset.csv", sep='\t', encoding='utf-8')

dataset = dataset.as_matrix()
X = dataset[:n_training_points,:-1].reshape(-1, n_dimensions)
Y = dataset[:n_training_points,-1].reshape(-1, 1)
# Create a 2-D RBF kernel active over both input dimensions


open('./RandomSearch.txt', 'w').close()
with open("./RandomSearch.txt", "a") as file1:
    for n in range(100):
        print("Trial number: ", n)

        n_rbf_features = np.random.choice(range(3, n_dimensions+1))
        n_std_periodic_features = np.random.choice(range(4, n_dimensions+1))
        n_linear_features = np.random.choice(range(4, n_dimensions+1))
        n_exponential_features = np.random.choice(range(2, n_dimensions+1))

        rbf_features = sorted(random.sample(range(n_dimensions), n_rbf_features))
        std_periodic_features = sorted(random.sample(range(n_dimensions), n_std_periodic_features))
        linear_features = sorted(random.sample(range(n_dimensions), n_linear_features))
        exponential_features =  sorted(random.sample(range(n_dimensions), n_exponential_features))

        std_periodic_periodicity =  np.random.choice(range(19, 25))

        k = GPy.kern.Linear(4, active_dims=[0, 1, 2, 3])

        

        try:
            m = GPy.models.GPRegression(X, Y, k, normalizer = False)
            m.optimize(optimizer='lbfgs', max_iters=1000)
        except:
            print("error!")
            continue

        X_new = dataset[n_training_points: n_training_points + n_test_points,:-1].reshape(-1, n_dimensions)
        Y_new = dataset[n_training_points: n_training_points + n_test_points, -1].reshape(-1, 1)
        mean, cov = m.predict(X_new)

        error = np.sum(((mean - Y_new) ** 2)) / Y_new.shape[0]

        file1.write("RBF features:" + str(rbf_features))
        file1.write(" Linear features:" + str(linear_features))
        # file1.write(" Std features:" + str( std_periodic_features))
        # file1.write(" Std periodicity:" + str( std_periodic_periodicity))
        # file1.write(" Exp features:" + str( exponential_features))
        file1.write(" Error: "+  str(error))
        file1.write("\n")

        #m.plot(plot_density=True)
        x = range(n_test_points)
        y1 = Y_new
        y2 = mean
        plt.plot(x, y1, "-b", label="target")
        plt.plot(x, y2, "-r", label="prediction")
        plt.title("Performance of Linear Kernel Multivariate GP")
        plt.legend(loc="upper left")
        plt.show()

        naive_prediction = sum((X_new[:,0]- Y_new.squeeze()) ** 2/Y_new.shape[0])

        print("Error: ", error)

# file1.close()
print("DONE!")