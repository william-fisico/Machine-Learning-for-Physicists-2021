########################################
# William A. P. dos Santos             #
# william.wapsantos@gmail.com          #
#                                      #
# Machine Learning for Physicists 2021 #
########################################

#(1)* Implement a network that computes XOR (arbitrary number of hidden layers); 
#meaning: the output should be +1 for y1 y2<0 and 0 otherwise!

import numpy as np # get the "numpy" library for linear algebra
import matplotlib.pyplot as plt


def apply_layer(y_in, w, b, activation): # a function that applies a layer
    z = np.dot(w,y_in) + b # result: the vector of 'z' values, length N1
    if activation=='sigmoid':
        return(1/(1+np.exp(-z)))
    elif activation=='jump':
        return(np.array(z>0,dtype='float'))
    elif activation=='linear':
        return(z)
    elif activation=='reLU':
        return((z>0)*z)

def apply_net(y_in):
    global w1, b1, w2, b2, w3, b3

    print(y_in)
    y1 = apply_layer(y_in, w1, b1, 'reLU')
    print(y1)
    y2 = apply_layer(y1, w2, b2, 'jump')
    print(y2)
    return(y2)



# From input layer to hidden layer:
w1 = np.array([[1,0],[0,-1]]) # random weights: N1xN0
b1 = np.array([0,0]) # biases: N1 vector


# From hidden layer to output layer:
w2 = np.array([[1,-1]]) 
b2 = np.array([0]) 

### Note: this is NOT the most efficient way to do this! (but simple) ###
M = 4 # will create picture of size MxM
y_out = np.zeros([M,M]) # array MxM, to hold the result

for j1 in range(M):
    for j2 in range(M):
        value0 = (float(j1)/M) - 0.5
        value1 = (float(j2)/M) - 0.5
        y_out[j1,j2] = apply_net([value0,value1])[0]

plt.imshow(y_out, origin='lower', extent=(-0.5,0.5,-0.5,0.5))
plt.colorbar()
plt.show()
