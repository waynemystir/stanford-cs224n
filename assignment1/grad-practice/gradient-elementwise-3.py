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
      print(ix, grad[ix], fxph)
    it.iternext() # step to next dimension
  return grad

np.random.seed(1)
X = np.random.randn(1,5)

def forward(x):
    x2 = x**2
    return x2

def backward(x):
    return 2*x

x2 = forward(X)
dx = backward(X)

xnumgrad = eval_numerical_gradient_array(lambda g: forward(g), X)
print("x({})\ndx({})\nxng({})\ndiff({})".format(X,dx,xnumgrad,dx-xnumgrad))
