from pylab import *

#Varaibles
n = 512
N = 1000
L = 32
gb = 180
l = 0.2
sig = 0.01

#Functions 
def getF(n=512, L=32):
    i = arange(n)
    j = arange(L)    
    F = exp(1j*2*pi*outer(i, j)/n)

    return F
    
def getXF(n=512, L=32, N=12):
    F = getF(n=n, L=L)
    X  = ((rand(N, n)>0.5).astype('int')*2-1) + ((rand(N, n)>0.5).astype('int')*2-1)*1j
    XF = einsum('ij,jk->ijk', X, F)
    
    return XF
    
def get_h(L=32, l=0.2):
    k = arange(L)
    p = exp(-l*k)
    ab = randn(L)*sqrt(0.5) + randn(L)*1j*sqrt(0.5)
    h = ab*p/norm(p)
    
    return h

def get_y(XF, h, N = 12 ,n=512,sig = 0.1) :
     w = sqrt(sig/2)*randn(N,n) + sqrt(sig/2)*randn(N,n)*1j   
     y = XF @ h + w   
     return y  
     
def get_gbh(XF, h, N = 12, L = 32) :     
     #Gaurd band constraint
     XF[:,:gb] = 0
     XF[:,-gb:] = 0
     y = get_y(XF, h, N=N)
     h_hat = np.zeros((N,L))
     for itr in range(0,N) :
          h_hat[itr] = lstsq(XF[itr],y[itr])[0]
             
     return h_hat
     
def get_rh(XF, h, alp = [0.001], N = 12, L = 32) :
     #Gaurd band constraint
     XF[:,:gb] = 0
     XF[:,-gb:] = 0
     
     y = get_y(XF, h, N=N)
     h_hat = np.zeros((N,len(alp),L))
     
     for i in range(0,len(alp)) :
          for itr in range(0,N) :
               h_hat[itr][i] = solve(((XF[itr].T @ XF[itr]) + alp[i]*np.identity(shape(XF[itr])[1])), XF[itr].T @ y[itr])
                  
     return h_hat

def get_spgbh(XF, h, N=12, L=32) :
     #Sparsity
     k0 = 6
     inds = permutation(arange(L))[:k0]
     
     #Gaurd band constraint
     XF[:,:gb] = 0
     XF[:,-gb:] = 0
     hsp = np.zeros(shape(h))
     hsp[inds] = h[inds]
     XFsp = XF[:,:,inds]
     y = get_y(XF, hsp, N=N)
     h_hat = np.zeros((N,len(inds)))
     for itr in range(0,N) :
          h_hat[itr] = lstsq(XFsp[itr],y[itr])[0]
     #Reconstruct h
     h_full = np.zeros((N,L))
     h_full[:,inds] = h_hat
     
     return h_full,hsp
     

#Call functions
h = get_h()
XF = getXF(N=N) 
#print(f"condition number = {cond(XF[0]):.6e}")
h_hat = get_gbh(XF, h,N=N)
Eh_hat = h_hat.mean(axis = 0)

alp = logspace(4,-1,6)
rh_hat = get_rh(XF, h,alp = alp,N=N)
#print(shape(rh_hat))
Erh_hat = rh_hat.mean(axis = 0)

mse = norm(h-rh_hat,2,axis = -1)
mse = mse**2
mse = mse.mean(axis = 0)
#print(shape(mse))

h_spgb,hsp = get_spgbh(XF, h, N=N)
Eh_spgb = h_spgb.mean(axis = 0)

#Plots 
'''
for i in range(0,len(alp)) :
     plt.plot(Erh_hat[i],'--',label = f"Estimate with $\\alpha$ = {alp[i]}")
     plt.plot(h, label = "True")
     plt.title("Channel estimate for different regularizations")
     plt.xlabel("Taps")
     plt.ylabel("h")
     plt.legend()
     plt.show()
'''
'''
Sigma = svd((XF[0].T @ XF[0]) + 0.001*np.identity(shape(XF[0])[1]), compute_uv = False)
plt.semilogy(Sigma)
plt.show()
'''
'''
min_mse = min(mse)
plt.semilogx(alp, mse)
plt.plot(alp,min_mse*np.ones((len(alp),1)))  
plt.title("MSE as a function of $\\alpha$")
plt.xlabel("$\\alpha")
pltt.ylabel("MSE")
plt.show()
'''
plt.plot(Eh_spgb, '--', label = "Estimate")
plt.plot(hsp,label = "True")
plt.title("Sparse channel estimate with gaurd band")
plt.xlabel("Taps")
plt.ylabel("h")
plt.legend()
plt.show()

plt.plot(h_spgb[0], '--', label = "Estimate")
plt.plot(hsp,label = "True")
plt.title("Sparse channel estimate with gaurd band")
plt.xlabel("Taps")
plt.ylabel("h")
plt.legend()
plt.show()

