from numpy import genfromtxt
import numpy as np
my_data = genfromtxt('./data_train/prostate_training_data.csv', delimiter=',')

my_data = my_data[1:,:]
X,y = my_data[:,:-1], my_data[:,-1]

# print(X[0])
y = np.array([y]).T
# print(y)


# Them 1 vao ma tran X
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)
# print(Xbar[0])

# Tao ma tran don vi, n + 1
I = np.eye(9)

# ma tran nghic dao np.linalg.inv(a)
ma = np.dot(Xbar.T, Xbar) + np.dot(29999, I)

# w = np.linalg.inv(ma) * Xbar.T * y
a1 = np.dot(np.linalg.inv(ma), Xbar.T)
w = np.dot(a1, y)

# A = np.dot(Xbar.T, Xbar)
# b = np.dot(Xbar.T, y)
# w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]
w_6 = w[6][0]
w_7 = w[7][0]
w_8 = w[8][0]




y1 = w_0 + w_1 * 1.19662872430723 + w_2 * 3.42299611099606 + w_3 * 57 - w_4 * 1.40082787030756 + w_5 * 0 - w_6 * 0.41596301173672 + w_7 * 7 + w_8 * 5
y1 = w_0 + w_1 * 2.03568254705244 + w_2 * 3.92927543150206 + w_3 * 66 + w_4 * 2.01892993803996 + w_5 * 1 + w_6 * 2.10008762960383 + w_7 * 7 + w_8 * 60
y1 = w_0 + w_1 * 2.12747378986827 + w_2 * 4.11225383296558 + w_3 * 68 + w_4 * 1.77471958211323 + w_5 * 0 + w_6 * 1.44801588994652 + w_7 * 7 + w_8 * 40
y1 = w_0 + w_1 * 2.03596874675694 + w_2 * 3.92005392712191 + w_3 * 66 + w_4 * 1.99867071277779 + w_5 * 1 + w_6 * 2.10623365964974 + w_7 * 7 + w_8 * 60
y1 = w_0 + w_1 * 1.30304691949589 + w_2 * 4.11876070871877 + w_3 * 64 + w_4 * 2.18325553084062 + w_5 * 0 + w_6 * -1.37240665246238 + w_7 * 7 + w_8 * 5

print(y1)
