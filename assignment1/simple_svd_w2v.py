import numpy as np
la = np.linalg
import matplotlib.pyplot as plt

E=np.array([[0,2,1,0,0,0,0,0],
     [2,0,0,1,0,1,0,0],
     [1,0,0,0,0,0,1,0],
     [0,1,0,0,1,0,0,0],
     [0,0,0,1,0,0,0,1],
     [0,1,0,0,0,0,0,1],
     [0,0,1,0,0,0,0,1],
     [0,0,0,0,1,1,1,0]])

words = ["I","like","enjoy","deep","learning","NLP","flying","."]

U,s,Vh = la.svd(E,full_matrices=False)
for i in range(len(words)):
    print("i (%d) word (%s) U[i,0] (%s) U[i,1] (%s)" % (i, words[i], U[i,0], U[i,1]))
    plt.text(U[i,0], U[i,1], words[i])

#xmin = np.min(U[:,0])
#xmax = np.max(U[:,0])
xmin = -0.7
xmax = -0.1
print("xmin(%f) xmax(%f)" % (xmin, xmax))
#axplt = np.linspace(-1,1,7)
axplt = np.linspace(xmin,xmax,6)
#plt.xticks((-1,1))
plt.xticks(axplt)
plt.yticks((-1,1))
plt.show()
