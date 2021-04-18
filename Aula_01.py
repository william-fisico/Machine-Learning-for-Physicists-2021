########################################
# William A. P. dos Santos             #
# william.wapsantos@gmail.com          #
#                                      #
# Machine Learning for Physicists 2021 #
########################################

from numpy import * # get the "numpy" library for linear algebra

N0 = 3 # input layer size
N1 = 2 # output layer size

w = random.uniform(low=-1, high=+1, size=(N1,N0)) # random weights: N1xN0
b = random.uniform(low=-1, high=+1, size=N1) # biases: N1 vector

y_in = array([0.2,0.4,-0.1]) #input values

z = dot(w,y_in) + b # result: the vector of 'z' values, length N1
y_out = 1/(1 + exp(-z)) # the sigmoid function (applied elementwise)

#Print results
print("y_in: ", y_in)
print("w: ", w)
print("b: ", b)
print("z: ", z)
print("y_out: ", y_out)