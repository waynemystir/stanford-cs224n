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
y = np.array([[0,0,1]])

def forward(x,w):
    z = x.dot(w) # 1x5 dot 5x3 = 1x3
    h = 1./(1+np.exp(-z))
    return h,z

def backward(h,z):
    dz = h*(1-h)
    dx = dz.dot(W.T) # 1x3 dot 3x5 = 1x5
    dW = X.T.dot(dz) # 5x1 dot 1x3 = 5x3
    return dx,dW

h,z = forward(X,W)
dx,dW = backward(h,z)

xnumgrad = eval_numerical_gradient_array(lambda g: forward(g,W)[0], X)
wnumgrad = eval_numerical_gradient_array(lambda g: forward(X,g)[0], W)
print("dx({})\nxng({})\ndiff({})".format(dx,xnumgrad,dx-xnumgrad))
print("dW({})\nWng({})\ndiff({})".format(dW,wnumgrad,dW-wnumgrad))
