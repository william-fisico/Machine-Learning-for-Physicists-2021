########################################
# William A. P. dos Santos             #
# william.wapsantos@gmail.com          #
#                                      #
# Machine Learning for Physicists 2021 #
########################################

from numpy import array, zeros, exp, random, dot, shape, reshape, meshgrid, linspace, transpose
import matplotlib.pyplot as plt # for plotting
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display
# import functions for updating display 
# (simple animation)
from IPython.display import clear_output
from time import sleep

'''
def net_f_df(z): # calculate f(z) and f'(z) ==> using sigmoid
    val = 1 / (1+exp(-z))
    return (val, exp(-z)*(val**2)) # return both f and f'
'''

# For a change: Set up rectified linear units (relu) 
# instead of sigmoid
def net_f_df(z): # calculate f(z) and f'(z)
    val=z*(z>0)
    return(val,z>0) # return both f and f'


def forward_step(y,w,b): # calculate values in next layer
    z = dot(y,w) + b # w=weights, b=bias vector for next layer
    return (net_f_df(z)) # apply nonlinearity


def apply_net(y_in): # one forward pass through the network
    global Weights, Biases, NumLayers
    global y_layer, df_layer # store y-values and df/dz

    y = y_in # start with input values
    y_layer[0] = y
    for j in range(NumLayers): # loop trough all layers
        # j=0 corresponds to the first layer above input
        y,df = forward_step(y, Weights[j], Biases[j])
        df_layer[j] = df # store f'(z)
        y_layer[j+1] = y # store f(z)
    return(y)


def backward_step(delta, w, df):
    # delta at layer N, of batchsize x layersize(N)
    # w[layersize(N-1) x layersiza(N) matrix]
    # df = df/dz at layer N-1, of batchsize x layersize(N-1)
    return (dot(delta,transpose(w))*df)


def backprop(y_target): # one backward pass
    # the result will be the 'dw_layer' matrices with
    # the derivatives of the cost function with respect to
    # the corresponding weight (similar for biases)
    global y_layer, df_layer, Weights, Biases, NumLayers
    global dw_layer, db_layer # dCost/dw and dCost/db => w,b=weights,biases
    global batchsize

    delta = (y_layer[-1] - y_target)*df_layer[-1]
    dw_layer[-1] = dot(transpose(y_layer[-2]),delta)/batchsize
    db_layer[-1] = delta.sum(0)/batchsize
    for j in range(NumLayers-1):
        delta = backward_step(delta, Weights[-1-j], df_layer[-2-j])
        dw_layer[-2-j] = dot(transpose(y_layer[-3-j]),delta)/batchsize
        db_layer[-2-j] = delta.sum(0)/batchsize

def apply_net_simple(y_in): # one forward pass through the network
    # no storage for backprop (this is used for simple tests)
    y=y_in # start with input values
    y_layer[0]=y
    for j in range(NumLayers): # loop through all layers
        # j=0 corresponds to the first layer above the input
        y,df=forward_step(y,Weights[j],Biases[j]) # one step
    return(y)


def gradient_step(eta): # update weights & biases (after backprop!)
    global dw_layer, db_layer, Weights, Biases
    
    for j in range(NumLayers):
        Weights[j]-=eta*dw_layer[j]
        Biases[j]-=eta*db_layer[j]


def train_net(y_in,y_target,eta): # one full training batch
    # y_in is an array of size batchsize x (input-layer-size)
    # y_target is an array of size batchsize x (output-layer-size)
    # eta is the stepsize for the gradient descent
    global y_out_result
    
    y_out_result=apply_net(y_in)
    backprop(y_target)
    gradient_step(eta)
    cost=((y_target-y_out_result)**2).sum()/batchsize
    return(cost)


#### Setup for a particular set of layer sizes

# set up all the weights and biases

NumLayers=4 # does not count input-layer (but does count output)
LayerSizes=[2,20,30,10,1] # input-layer,hidden-1,hidden-2,...,output-layer

# initialize random weights and biases for all layers (except input of course)
Weights=[random.uniform(low=-1,high=+1,size=[ LayerSizes[j],LayerSizes[j+1] ]) for j in range(NumLayers)]
Biases=[random.uniform(low=-1,high=+1,size=LayerSizes[j+1]) for j in range(NumLayers)]


# set up all the helper variables

y_layer=[zeros(LayerSizes[j]) for j in range(NumLayers+1)]
df_layer=[zeros(LayerSizes[j+1]) for j in range(NumLayers)]
dw_layer=[zeros([LayerSizes[j],LayerSizes[j+1]]) for j in range(NumLayers)]
db_layer=[zeros(LayerSizes[j+1]) for j in range(NumLayers)]

# define the batchsize
batchsize=1000


def myFunc(x0,x1):
    r2=x0**2+x1**2
    return(exp(-5*r2)*abs(x1+x0))

xrange=linspace(-0.5,0.5,40)
X0,X1=meshgrid(xrange,xrange)
plt.imshow(myFunc(X0,X1),interpolation='nearest',origin='lower')
plt.show()


def make_batch():
    global batchsize

    inputs=random.uniform(low=-0.5,high=+0.5,size=[batchsize,2])
    targets=zeros([batchsize,1]) # must have right dimensions
    targets[:,0]=myFunc(inputs[:,0],inputs[:,1])
    return(inputs,targets)

# try to evaluate the (randomly initialized) network
# on some area in the 2D plane
test_batchsize=shape(X0)[0]*shape(X0)[1]
testsample=zeros([test_batchsize,2])
testsample[:,0]=X0.flatten()
testsample[:,1]=X1.flatten()


eta=0.001 # learning rate
nsteps=1000

costs=zeros(nsteps)
for j in range(nsteps):
     # the crucial lines:
    y_in,y_target=make_batch() # random samples (points in 2D)
    costs[j]=train_net(y_in,y_target,eta) # train network (one step, on this batch)
    testoutput=apply_net_simple(testsample) # check the new network output in the plane

    if j >= (nsteps-1):
        clear_output(wait=True)
        fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(8,4)) # prepare figure
        ax[1].axis('off') # no axes
        
       
        
        img=ax[1].imshow(reshape(testoutput,shape(X0)),interpolation='nearest',origin='lower') # plot image
        ax[0].plot(costs)
        
        ax[0].set_title("Cost during training")
        ax[0].set_xlabel("number of batches")
        ax[1].set_title("Current network prediction")
        plt.show()
        sleep(0.1)
