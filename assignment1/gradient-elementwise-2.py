import numpy as np

def eval_numerical_gradient_array(f, x, verbose=True, h=0.00001):
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
    grad[ix] = np.sum(fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print(ix, grad[ix])
    it.iternext() # step to next dimension
  return grad

np.random.seed(1)
X = np.random.randn(1,5)
W = np.random.randn(5,3)
T = np.random.randn(3,6)

def forward(x,w,t):
    z1 = x.dot(w) # 1x5 dot 5x3 = 1x3
    h = 1./(1+np.exp(-z1))
    z2 = h.dot(t) # 1x3 dot 3x6 = 1x6
    g = 1./(1+np.exp(-z2))
    return g,h,z1,z2

def backward(g,h,z1,z2):
    dz2 = g*(1-g)
    dh = dz2.dot(T.T) # 1x6 dot 6x3 = 1x3
#    dz1 = dh * (np.exp(-z1)) / np.power((1+np.exp(-z1)),2)
    dz1 = dh * h * (1-h)
    dx = dz1.dot(W.T) # 1x3 dot 3x5 = 1x5
    dW = X.T.dot(dz1) # 5x1 dot 1x3 = 5x3
    return dx,dW

g,h,z1,z2 = forward(X,W,T)
dx,dW = backward(g,h,z1,z2)

xnumgrad = eval_numerical_gradient_array(lambda g: forward(g,W,T)[0], X)
wnumgrad = eval_numerical_gradient_array(lambda g: forward(X,g,T)[0], W)
print("dx({})\nxng({})\ndiff({})".format(dx,xnumgrad,dx-xnumgrad))
print("dW({})\nWng({})\ndiff({})".format(dW,wnumgrad,dW-wnumgrad))
