import numpy as np

def eval_numerical_gradient_array(f, x, verbose=True, h=0.00001):
  grad = np.zeros((f(x).shape[1],x.shape[1]))  
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
    grad[:,ix[1]] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print(ix, grad[ix], fxph)
    it.iternext() # step to next dimension
  return grad

#np.random.seed(1)
X = np.random.randn(1,5)
Z = np.random.randn(5,23)
W = np.random.randn(23,6)

def forward(x,w):
    return x.dot(Z).dot(w) # 1x5 dot 5x23 dot 23x6 = 1x6

def backward(x):
#    return W.T.dot(Z.T)
    return Z.dot(W).T

x2 = forward(X,W)
dx = backward(X)

xnumgrad = eval_numerical_gradient_array(lambda g: forward(g,W), X)
print("x({})\ndx({})\nxng({})\ndiff({})".format(X,dx,xnumgrad,dx-xnumgrad))
