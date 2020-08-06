#Import libraries
from pylab import *
from scipy.stats import beta,binom

rcParams['figure.figsize'] = 12,9
rcParams['axes.grid'] = True
rcParams['font.size'] = 18
rcParams['figure.facecolor'] = 'w'

def likelihood(Nx=20) :
    x = np.arange(0,1,1/Nx)         #X sample space
    y = np.arange(0,Nx)
    Pyx = np.zeros((Nx,Nx))         #Binomial Likelihood
    for i in range(0,Nx) :
        Pyx[:,i] = binom.pmf(y,Nx-1,p = x[i])
    
    return Pyx
                                    #Obtain MSE w/o prior using Bayes
def pfreeBayes(N,Pyx,a=3,b=3,Nx=20) :  
                                    #Observations/samples
    lenPy = 0
    iters = 0
    while(lenPy != Nx) :            #Loop till Py is well defined
        X = beta.rvs(a,b,size = N)
        Y = np.random.binomial(Nx-1,X)
        Py = (np.unique(np.sort(Y),return_counts = True)[1])/N
        lenPy = len(Py)
        iters = iters+1
    #print(iters)
                                    #Learn prior
    x = np.arange(0,1,1/Nx)
    Pyxinv = np.linalg.solve(Pyx,Py)
    L = Pyx @ np.diag(x) @ Pyxinv
    Ly = L/Py
    X = beta.rvs(a,b,size = N)
    Y = np.random.binomial(Nx-1,X)
    Ly = Ly[Y]
    mseX = ((Ly-X)**2).mean()       #MSE wrt X
    #print(mseX)
    
    return mseX
                                    #Obtain MSE of BLS with prior
def optimalBLS(N,a=3,b=3,Nx=20) :
                                    #Data & obsrevations
    X = beta.rvs(a,b,size = N)
    Y = np.random.binomial(Nx-1,X)
    Ex = (a+Y)/(a+b+Nx-1)           #Estimated X
    mseX = ((Ex-X)**2).mean()       #MSE wrt X
    
    return mseX
    
                                    #Obtain MSE of MLE
def MLE(N,Pyx,a=3,b=3,Nx=20) :
                                    #Data & obsrevations
    x = np.arange(0,1,1/Nx)
    X = beta.rvs(a,b,size = N)
    Y = np.random.binomial(Nx-1,X)
    Ex = x[argmax(Pyx,axis=1)]      #Estimated X
    Ex = Ex[Y]
    mseX = ((Ex-X)**2).mean()       #MSE wrt X
    
    return mseX
    
                                    #Obtain MSE of supervised BLS
def supBLS(N,a=3,b=3,Nx=20) :
    lenPy = 0
    while(lenPy != Nx) :
                                        #Data & obsrevations
        X = beta.rvs(a,b,size = N)
        Y = np.random.binomial(Nx-1,X)
        Py = (np.unique(np.sort(Y),return_counts = True)[1])/N
        lenPy = len(Py)
        
    Ex = np.zeros(Nx)    
    for i in range(0,Nx):            #Estimated X
        Ex[i] = X[np.where(Y == i)[0]].mean()
    Ex = Ex[Y]
    mseX = ((Ex-X)**2).mean()       #MSE wrt X
    
    return mseX
    
def msevar(reps = 100,Ntot = range(2,6)) :
    
    means_pf = []
    var_pf = []
    means_bls = []
    var_bls = []
    means_mle = []
    var_mle = []
    means_sup = []
    var_sup = []
    for n in Ntot :
        N = 10**n
        print(N)
        Pyx = likelihood()
        mse_pf = []
        mse_bls = []
        mse_mle = []
        mse_sup = []
        for i in range(0,reps) :
              mseX = pfreeBayes(N,Pyx)
              mse_pf.append(mseX)
              
              mseX = optimalBLS(N)
              mse_bls.append(mseX)
              
              mseX = MLE(N,Pyx)
              mse_mle.append(mseX)
              
              mseX = supBLS(N)
              mse_sup.append(mseX)
              
        mse_pf = array(mse_pf)      #Compute MSE mean and var for prior free
        means_pf.append(mse_pf.mean())
        var_pf.append(mse_pf.std())
        mse_bls = array(mse_bls)      #Compute MSE mean and var for optimal BLS
        means_bls.append(mse_bls.mean())
        var_bls.append(mse_bls.std())
        mse_mle = array(mse_mle)      #Compute MSE mean and var for MLE
        means_mle.append(mse_mle.mean())
        var_mle.append(mse_mle.std())
        mse_sup = array(mse_sup)      #Compute MSE mean and var for supervised BLS
        means_sup.append(mse_sup.mean())
        var_sup.append(mse_sup.std())
    semilogx()
    errorbar(10**array(Ntot), means_pf, var_pf, capsize=6, marker='o', label='prior-free')
    errorbar(10**array(Ntot), means_bls, var_bls, capsize=6, marker='o', label='optimal BLS(with prior)')
    errorbar(10**array(Ntot), means_mle, var_mle, capsize=6, marker='o', label='MLE')
    errorbar(10**array(Ntot), means_sup, var_sup, capsize=6, marker='o', label='supervised BLS(with X and Y)')
    plt.legend()
    plt.show()
    
    return None
#Pyx = likelihood()
#mseX = pfreeBayes(100,Pyx)
msevar()

