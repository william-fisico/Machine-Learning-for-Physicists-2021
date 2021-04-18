########################################
# William A. P. dos Santos             #
# william.wapsantos@gmail.com          #
#                                      #
# Machine Learning for Physicists 2021 #
########################################

from numpy import * # get the "numpy" library for linear algebra
import matplotlib.pyplot as plt

def apply_net(y_in): # a function that applies the network
    global w, b

    z = dot(w,y_in) + b # result: the vector of 'z' values, length N1
    return 1/(1 + exp(-z)) # the sigmoid function (applied elementwise)

N0 = 2 # input layer size
N1 = 1 # output layer size

w = random.uniform(low=-1, high=+1, size=(N1,N0)) # random weights: N1xN0
b = random.uniform(low=-1, high=+1, size=N1) # biases: N1 vector

# Testing
y_in = array([0.8,0.3]) #input values
print(apply_net(y_in))

### Note: this is NOT the most efficient way to do this! (but simple) ###
M = 50 # will create picture of size MxM
y_out = zeros([M,M]) # array MxM, to hold the result

for j1 in range(M):
    for j2 in range(M):
        value0 = (float(j1)/M) - 0.5
        value1 = (float(j2)/M) - 0.5
        y_out[j1,j2] = apply_net([value0,value1])[0]

plt.imshow(y_out, origin='lower', extent=(-0.5,0.5,-0.5,0.5))
plt.colorbar()
plt.show()