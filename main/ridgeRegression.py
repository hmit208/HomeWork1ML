import numpy as np
import matplotlib.pyplot as plt

#
def ridge_regression(x_train, y_train,x_test,y_test, lam):
    X = np.array(x_train)
    ones = np.ones(len(X))
    X = np.column_stack((ones, X))
    X2 = np.array(x_test)
    one2 = np.ones(len(X2))
    X2 = np.column_stack((one2,X2))
    y = np.array(y_train)

    Xt = np.transpose(X)
    lambda_identity = lam * np.identity(len(Xt))
    #lambda_identity[0][0] = 0
    #print(lambda_identity)
    theInverse = np.linalg.inv(np.dot(Xt, X) + lambda_identity)
    w = np.dot(np.dot(theInverse, Xt), y)

    y_pred = np.dot(X2,w)
    ridge_error =  np.sum((y_pred-y_test) ** 2 ) / len(y_pred)
    #ridge_error = mean_absolute_error(y_test, y_pred)

    return w , ridge_error

with open('./data_train/prostate_training_data.csv') as csvfile:
    Xy = np.loadtxt(csvfile, delimiter=',', dtype=float,skiprows = 1)

X = Xy[:, 0:-1]
y = Xy[:, -1]


train_size = (int) (0.8 * len(y))

X_train = X[:train_size,:]
y_train = y[:train_size]

X_cross = X[train_size:,:]
y_cross = y[train_size:]



lam = []
error = []
param_grid = 0.01 * np.arange(0, 500)
min_lam = 0
min_err = 1000000
for lmda in param_grid:
    w, err = ridge_regression(X_train, y_train, X_cross, y_cross, lmda)
    #print(str(lmda)+ "  "+str(err))
    lam.append(lmda)
    error.append(err)
    if err < min_err:
        min_err = err
        min_lam = lmda

plt.plot(np.array(lam), np.array(error), 'ro')
print(min_lam)

# plt.show()
w,err = ridge_regression(X_train, y_train, X_cross, y_cross, min_lam)

with open('./data_train/20144067-test.csv') as csvfile:
    Xy = np.loadtxt(csvfile, delimiter=',', dtype=float)

X_test = Xy[:, 0:-1]
y_test = Xy[:, -1]
X = np.array(X_test)
ones = np.ones(len(X))
#
X = np.column_stack((ones, X))
y_pred = np.dot(X,w)


res = np.column_stack((X,y_pred))
res = res[:,1:]

res = np.array(res)
np.savetxt("20144067.csv", res, delimiter=',',fmt='%f')
plt.show()

