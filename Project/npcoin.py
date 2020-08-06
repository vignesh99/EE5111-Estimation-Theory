#Import libraries
from pylab import *
from random import *

                                #Input data/observations
N = 1000
X = np.random.randint(0,2,N)    #Actual data
err = np.random.randint(0,4,N)  #Error indices
inderr = np.where(err == 0)[0]  
Y = np.copy(X)
Y[inderr] = ~Y[inderr]+2          #Observations

Py = array([len(np.where(Y==0)[0])/len(Y), len(np.where(Y==1)[0])/len(Y)])
Pyx = array([[3/4,1/4],[1/4,3/4]])
Xdiag = array([[0,0],[0,1]])
Pyxinv = 2*array([[3/4,-1/4],[-1/4,3/4]])
Ly = Pyx @ Xdiag @ Pyxinv @ Py
Ly = Ly/Py
print(Ly)

