import numpy as np
import random
import time
from utils.treebank import StanfordSentiment

def gradcheck_naive(f,x):
    rs = random.getstate()
    random.setstate(rs)
    _,grad=f(x)
    h=1e-4
    it = np.nditer(x,flags=['multi_index'])
    while not it.finished:
        ix = it.multi_index
        x[ix] -= h
        random.setstate(rs)
        fxb = f(x)[0]
        x[ix] += 2*h
        random.setstate(rs)
        fxa = f(x)[0]
        x[ix] -= h
        numgrad = (fxa-fxb)/(2*h)
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(fxa), abs(fxb))
        if reldiff > 1e-5:
            print("Gradient check FAILED at %s. Analytic grad (%f) and numerical grad (%f)" % (ix, grad[ix], numgrad))
            return
        it.iternext()
    print("Gradient check is GOOD!!!")

def softmax(x):
    p = np.exp(x - np.max(x, axis=x.ndim-1, keepdims=True))
    return p / np.sum(p, axis=p.xdim-1, keepdims=True)

def softmaxCAG(pred, target, outputVectors, dataset, K=10, verbose=False):
    yy = softmax(outputVectors.dot(pred)) # ŷ
    cost = -np.log(yy[target])
    yy[target] -= 1 # ŷ - y
    gradPred = outputVectors.T.dot(yy)
    grad = yy.reshape(outputVectors.shape[0],1).dot(pred.reshape(outputVectors.shape[1]))  # = np.outer(yy,pred), which is slower for smaller matrices like these
    return cost, gradPred, grad

def sigmoid(z): return 1./(1+np.exp(-z))
def sigmoid_grad(f): return f * (1-f)

def getSampleTokenIndices(target,dataset,K):
    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices

def negSamplingCAG(pred, target, outputVectors, dataset, K=10, verbose=False):
    ids = [target] + [i for i in getSampleTokenIndices(target,dataset,K)]
    grad,outputWords,dirs=np.zeros_like(outputVectors),outputVectors[ids,:],np.array([[1] + [-1 for _ in range(K)]])
    δ1 = sigmoid(outputWords.dot(pred) * dirs)
    cost,δ2 = -np.sum(np.log(δ1)),(δ1-1)*dirs
    gradPred = δ2.reshape(1,K+1).dot(outputWords).flatten()
    gTmp = δ2.reshape(K+1,1).dot(pred.reshape(1,outputVectors.shape[1]))
    for k in range(K+1):
        grad[ids[k]] += gTmp[k,:]
    return cost,gradPred,grad

def skipgram(currentWord, contextWords, tokens, inputVectors, outputVectors, dataset, w2vCAG=negSamplingCAG, verbose=False):
    vhat,cost,gradIn,gradOut = inputVectors[tokens[currentWord]],0.,np.zeros_like(inputVectors),np.zeros_like(outputVectors)
    for word in contextWords:
        cst,gi,go = w2vCAG(vhat,tokens[word],outputVectors,dataset)
        cost += cst
        gradIn[tokens[currentWord]] += gi
        gradOut += go
    return cost,gradIn,gradOut

def cbow(currentWord, contextWords, tokens, inputVectors, outputVectors, dataset, w2vCAG=negSamplingCAG, verbose=False):
    vhat_indices,gradIn = [tokens[cw] for cw in contextWords],np.zeros_like(inputVectors)
    vhat_vectors = inputVectors[vhat_indices]
    vhat = np.sum(vhat_vectors,axis=0)
    cost,gi,gradOut = w2vCAG(vhat,tokens[currentWord],outputVectors,dataset)
    for i in vhat_indices:
        gradIn[i] += gi
    return cost,gradIn,gradOut

def sgd_wrapper(tokens, vectors, contextSize, dataset, w2vModel=skipgram, w2vCAG=negSamplingCAG, verbose=False):
    batchsize,N = 50,vectors.shape[0]//2
    cost,grad = 0.,np.zeros_like(vectors)
    inputVectors,outputVectors = vectors[:N,:],vectors[N:,:]
    for _ in range(batchsize):
        C1 = random.randint(1,contextSize)
        currentWord,contextWords = dataset.getRandomContext(C1)
        cst,gi,go = w2vModel(currentWord,contextWords,tokens,inputVectors,outputVectors,dataset,w2vCAG=w2vCAG)
        cost += cst/batchsize
        grad[:N] += gi/batchsize
        grad[N:] += go/batchsize
    return cost,grad

def sgd(f,x0,itrs,learning_rate,print_every=1000):
    x = x0
    expcost=None
    for i in range(itrs):
        cost,grad=f(x)
        x -= learning_rate*grad
        if expcost is None: expcost = cost
        else: expcost = 0.9*expcost + 0.1*cost
        if i % print_every == 0:
            print("SGD iter(%d) expcost(%f) cost(%f)" % (i,expcost,cost))
    return x

def run():
    random.seed(319)
    dataset = StanfordSentiment()
    tokens_encoded = dataset.tokens()
    for k,v in tokens_encoded.items():
        if type(k) == str:
            tokens_encoded.pop(k)
            tokens_encoded[k.encode('latin1')] = v
    tokens = dict((k.decode('latin1'),v) for k,v in tokens_encoded.items())
    V,D = len(tokens),10
    random.seed(31919)
    np.random.seed(419)
    vectors = np.concatenate((np.random.randn(V,D),np.zeros((V,D))),axis=0)
    st = time.time()
    vectors = sgd(lambda vecs: sgd_wrapper(tokens_encoded,vectors,5,dataset,w2vModel=skipgram,w2vCAG=negSamplingCAG), vectors, 5001, 3e-1)
    print("run-sgd finished in (%f) seconds" % (time.time()-st))

def testit():
    dataset = type('dummy',(),{})()
    def dummySampleTokenIdx():
        return random.randint(0,4)
    def dummyRandomContext(C):
        tokens=['a','b','c','d','e']
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for _ in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = dummyRandomContext
    tokens = {'a':0,'b':1,'c':2,'d':3,'e':4,}
    random.seed(319)
    np.random.seed(419)
    vectors = np.random.randn(10,3)
    st,contextSize = time.time(),17
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens, vecs, contextSize, dataset, w2vModel=skipgram), vectors)
    print("##########################  Gradient Check skipgram negSamplingCAG contextSize=%d in (%f) seconds" % (contextSize, time.time()-st))
    st,contextSize = time.time(),5
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens, vecs, contextSize, dataset, w2vModel=skipgram), vectors)
    print("##########################  Gradient Check skipgram negSamplingCAG contextSize=%d in (%f) seconds" % (contextSize, time.time()-st))
    st,contextSize = time.time(),17
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens, vecs, contextSize, dataset, w2vModel=cbow), vectors)
    print("##########################  Gradient Check cbow negSamplingCAG contextSize=%d in (%f) seconds" % (contextSize, time.time()-st))
    st,contextSize = time.time(),5
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens, vecs, contextSize, dataset, w2vModel=cbow), vectors)
    print("##########################  Gradient Check cbow negSamplingCAG contextSize=%d in (%f) seconds" % (contextSize, time.time()-st))
    st,contextSize = time.time(),17
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens, vecs, contextSize, dataset, w2vModel=skipgram), vectors)
    print("##########################  Gradient Check skipgram negSamplingCAG contextSize=%d in (%f) seconds" % (contextSize, time.time()-st))

if __name__=='__main__':
#    testit()
    run()
