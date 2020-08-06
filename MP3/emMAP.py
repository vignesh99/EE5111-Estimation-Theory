#Import libraries
from pylab import *
from scipy.stats import beta
import scipy.optimize

                        #Obtain observations with given parameters
def getXZ(pi, p, q, m=10, n=1000):
    # A=0, B=1; H=0, T=1
    pq = array([p, q])
    z = (rand(n)>pi).astype('int')
    x = rand(n, m)
    x = (x<pq[z].reshape(-1, 1)).astype('int')
    return x, z
    
                        #EM with prior on pi
def update_em3(x,theta,n,m,prior=[1,1]) :
    pi, p, q = theta
    a,b = prior
    heads = count_nonzero(x, axis=-1)
    A = (p**heads)*(1-p)**(m-heads)
    B = (q**heads)*(1-q)**(m-heads)
    Ez = (1-pi)*B/(pi*A + (1-pi)*B)
    prob = heads/m
    new_pi = (a-1+(1-Ez).sum())/(n+a+b-2)
    new_q = (prob*Ez).sum()/Ez.sum()
    new_p = (prob*(1-Ez)).sum()/(1-Ez).sum()
    theta_new = [new_pi, new_p, new_q]
    return array(theta_new)
    
                        #Use update and do iterative run    
def EM(x, theta_0, update_fn, maxiter=3000, **kwargs):
    n, m = x.shape
    theta_all = [theta_0]
    theta_all.append(update_fn(x, theta_0, n, m, **kwargs))
    err = 1
    i=0
    while err>1e-6 and i<maxiter:
        theta_all.append(update_fn(x, theta_all[-1], n, m, **kwargs))
        err = abs(theta_all[-1] - theta_all[-2]).max()
#         print(err)
        i+=1
    return array(theta_all)

pipq2 = 0.25, 0.35, 0.6    
theta0 = array([0.5, 0.5, 0.45])
x,z = getXZ(*pipq2, m=10, n=10000)
print(EM(x, theta0, update_em3))
