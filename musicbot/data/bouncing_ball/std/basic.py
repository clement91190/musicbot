from pylab import *
from loader import gload, load, save, gsave, fload
gray()
def show_mat(x):
    hold(False)
    imshow(x,interpolation='nearest')
def show(V):
    if V.ndim==1:
        res = int(sqrt(shape(V)[0]))
        if abs(res**2 - shape(V)[0])>0.00001: #i.e., if its not an integer
            print "check the dimensionality of the vector, not a square"
        show_mat(V.reshape(res, res))
    if V.ndim==2:
        show_mat(V)
    if V.ndim>2:
        print "don't know how to print such a vector!"

def show_seq(V):
    T   = len(V)
    res = int(sqrt(shape(V)[1]))
    for t in range(T):
        print t
        show(V[t])
def unsigmoid(x): return log (x) - log (1-x)


def print_aligned(w):
    n1 = int(ceil(sqrt(shape(w)[1])))
    n2 = n1
    r1 = int(sqrt(shape(w)[0]))
    r2 = r1
    Z = zeros(((r1+1)*n1, (r1+1)*n2), 'd')
    i1, i2 = 0, 0
    for i1 in range(n1):
        for i2 in range(n2):
            i = i1*n2+i2
            if i>=shape(w)[1]: break
            Z[(r1+1)*i1:(r1+1)*(i1+1)-1, (r2+1)*i2:(r2+1)*(i2+1)-1] = w[:,i].reshape(r1,r2)
    return Z


def aux(a):
    (s1, s2)=shape(a)
    mu=.5*ones((s1,1),'d')
    return concatenate((a,mu),1)
def aux2(a):
    (s1, s2)=shape(a)
    mu=.5*ones((1,s2),'d')
    return concatenate((a,mu),0)


def print_seq(x): # x is assumed to be a sequence.

    (T, s0)=shape(x)
    s1 = int(sqrt(s0))
    assert(s1**2 == s0)
    
    Z=aux2(concatenate([aux(f) for f in x.reshape(T,s1,s1)],1))
    return Z

def stochastic(x): return array(rand(*shape(x))<x,'d')
def Rsigmoid(x): return stochastic(sigmoid(x))
def sigmoid(x):  return 1./(1. + exp(-x))
def Dsigmoid(x):
    s = sigmoid(x)
    return s*(1-s)
def monomial(n, shape=(1,)): return array(floor(rand(*shape)*n),'i')
def id(x): return x

