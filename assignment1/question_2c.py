import numpy as np

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # restore
    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print(ix, grad[ix])
    it.iternext() # step to next dimension
  return grad

np.random.seed(1)
X = np.random.randn(1,5)
W1 = np.random.randn(5,3)
B1 = np.random.randn(1,3)
W2 = np.random.randn(3,5)
b2 = np.random.randn(1,5)
y = np.array([[0,0,1,0,0]])

def forward(x,w1,b1):
    z1 = x.dot(w1)+b1 # 1x5 dot 5x3 = 1x3
    h = 1./(1+np.exp(-z1))
    z2 = h.dot(W2)+b2 # 1x3 dot 3x5 = 1x5
    p = np.exp(z2 - np.max(z2,1,keepdims=True))
    yhat = p/np.sum(p,1,keepdims=True)
    loss = -np.sum(np.log(yhat[y==1]))
    return loss,yhat,h

def backward(yhat,h):
    d1 = yhat - y
    print("d1({}) yhat({}) y({})".format(d1.shape,yhat.shape,y.shape))
    d2 = d1.dot(W2.T) # 1x5 dot 5x3 = 1x3
    d3 = d2*h*(1-h) # 1x3
    
    dx = d3.dot(W1.T) # 1x3 dot 3x5 = 1x5
    dW1 = X.T.dot(d3) # 5x1 dot 1x3 = 5x3
    db1 = np.sum(d3,0,keepdims=True)
    print("dW1({}\n{})".format(dW1.shape, dW1))
    return dx,dW1,db1

_,yh,h = forward(X,W1,B1)
dx,dW1,dB1 = backward(yh,h)

xnumgrad = eval_numerical_gradient(lambda g: forward(g,W1,B1)[0], X)
wnumgrad = eval_numerical_gradient(lambda g: forward(X,g,B1)[0], W1)
bnumgrad = eval_numerical_gradient(lambda g: forward(X,W1,g)[0], B1)
print("dx({})\nxng({})\ndiff({})".format(dx,xnumgrad,dx-xnumgrad))
print("dW({})\nWng({})\ndiff({})".format(dW1,wnumgrad,dW1-wnumgrad))
print("dB({})\nBng({})\ndiff({})".format(dB1,bnumgrad,dB1-bnumgrad))
