from pylab import *
from scipy.optimize import minimize
from scipy.stats import cauchy

#Get Cauchy likelihood(-ve)
def Cauchyll(A,gamma,x) :
    n = len(x)
    l = -sum(np.log(gamma**2 + (x-A)*(x-A)))
    l = l + n*np.log(gamma/pi)
    #l = np.log(cauchy.pdf(x,loc = A,scale = gamma))
    return -l
   
#Get Cauchy MLE 
def CauchyMLE(gamma,x) :
    Aopt = minimize(Cauchyll,5,args = (gamma,x))
    
    return Aopt
    
#Generate Cauchy R.Vec
gamma = 1.89
A = 10
x = cauchy.rvs(loc = A, scale = gamma, size = 10000)
print(x)
xopt = CauchyMLE(gamma,x)
print(xopt)
