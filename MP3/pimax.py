#Import libraries
from pylab import *
from sympy import *

                                    #Define the symbols
p,q,pi = symbols("p q pi")          #Symbols of theta
n,m = symbols("n m")                #Symbols of counts
n1,n2,n3 = symbols("n1 n2 n3")      #Symbols of no of heads in each trial
a,b = symbols("a b")                #Symbols related to prior

                                    #Expressions
psum = ((pi*(p**n1)*((1-p)**(m-n1)))/(((pi*(p**n1)*((1-p)**(m-n1))))+ ((1-pi)*(q**n1)*((1-q)**(m-n1))))) 
psum = psum + ((pi*(p**n2)*((1-p)**(m-n2)))/(((pi*(p**n2)*((1-p)**(m-n2))))+ ((1-pi)*(q**n2)*((1-q)**(m-n2)))))
psum = psum + ((pi*(p**n3)*((1-p)**(m-n3)))/(((pi*(p**n3)*((1-p)**(m-n3))))+ ((1-pi)*(q**n3)*((1-q)**(m-n3)))))

qsum = (((1-pi)*(q**n1)*((1-q)**(m-n1)))/(((pi*(p**n1)*((1-p)**(m-n1))))+ ((1-pi)*(q**n1)*((1-q)**(m-n1))))) 
qsum = qsum + (((1-pi)*(q**n2)*((1-q)**(m-n2)))/(((pi*(p**n2)*((1-p)**(m-n2))))+ ((1-pi)*(q**n2)*((1-q)**(m-n2)))))
qsum = qsum + (((1-pi)*(q**n3)*((1-q)**(m-n3)))/(((pi*(p**n3)*((1-p)**(m-n3))))+ ((1-pi)*(q**n3)*((1-q)**(m-n3)))))

#psum = 1
#qsum = 1

piExpr = log(pi)*(psum) + log(1-pi)*(qsum) + (a-1)*log(pi) + (b-1)*log(1-pi)

pdpi = diff(piExpr,pi)
pimax = solveset(Eq(pdpi,0),pi)
#print(pimax)
print(latex(pimax))
