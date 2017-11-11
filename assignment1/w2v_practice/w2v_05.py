import numpy as np
import random
import time
from utils.treebank import StanfordSentiment

def gradcheck_naive(f,x):
    rs = random.getstate()
    random.setstate(rs)
    _,grad = f(x)
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
        numgrad = (fxa - fxb) / (2*h)
        relerror = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if relerror > 1e-5:
            print("Gradient check failed at %s. Analytic gradient (%f) and numerical gradient (%f)" % (ix,grad[ix],numgrad))
            return
        it.iternext()
    print("Gradient check PASSED!!!")

def softmax(x):
    p = np.exp(x - np.max(x, axis=x.ndim-1, keepdims=True))
    return p / np.sum(p, axis=p.ndim-1, keepdims=True)

def softmaxCAG(pred, target, outputVectors, dataset, K=10, verbose=False):
    yy = softmax(outputVectors.dot(pred)) # ŷ
    cost = -np.log(yy[target])
    yy[target] -= 1
    gradPred = outputVectors.T.dot(yy)
    grad = yy.reshape(outputVectors.shape[0],1).dot(pred.reshape(1,outputVectors.shape[1]))
    return cost,gradPred,grad

def sigmoid(z): return 1./(1+np.exp(-z))
def sigmoid_grad(f): f*(1-f)

def getSampleTokens(target,dataset,K):
    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices

def negSamplingCAG(pred, target, outputVectors, dataset, K=10, verbose=False):
    ids = [target] + [i for i in getSampleTokens(target,dataset,K)]
    ψ,α,grad=outputVectors[ids,:],np.array([[1] + [-1 for _ in range(K)]]),np.zeros_like(outputVectors)
    δ = sigmoid(ψ.dot(pred) * α)
    cost,φ=-np.sum(np.log(δ)),(δ-1)*α
    gradPred = ψ.T.dot(φ.reshape(K+1,1)).flatten() # φ.reshape(1,K+1).dot(outputWords).flatten()
    go = φ.reshape(K+1,1).dot(pred.reshape(1,ψ.shape[1]))
    for k in range(K+1): grad[ids[k]] += go[k,:]
    return cost,gradPred,grad

def negSamplingCAG_Origin(pred, target, outputVectors, dataset, K=10, verbose=False):
    ids = [target] + [i for i in getSampleTokens(target,dataset,K)]
    ψ,α,grad=outputVectors[ids,:],np.array([[1] + [-1 for _ in range(K)]]),np.zeros_like(outputVectors)
    δ = sigmoid(ψ.dot(pred) * α)
    cost,φ=-np.sum(np.log(δ)),(δ-1)*α
    gradPred = φ.reshape(1,K+1).dot(ψ).flatten()
    go = φ.reshape(K+1,1).dot(pred.reshape(1,ψ.shape[1]))
    for k in range(K+1): grad[ids[k]] += go[k,:]
    return cost,gradPred,grad

def skipgram(currentWord, contextWords, tokens, inputVectors, outputVectors, dataset, w2vCAG=negSamplingCAG, verbose=False):
    vhat,cost,gradIn,gradOut=inputVectors[tokens[currentWord]],0,np.zeros_like(inputVectors),np.zeros_like(outputVectors)
    for word in contextWords:
        cst,gi,go = w2vCAG(vhat,tokens[word],outputVectors,dataset)
        cost+=cst
        gradIn[tokens[currentWord]] += gi
        gradOut += go
    return cost,gradIn,gradOut

def cbow(currentWord, contextWords, tokens, inputVectors, outputVectors, dataset, w2vCAG=negSamplingCAG, verbose=False):
    vhat_indices,gradIn = [tokens[cw] for cw in contextWords],np.zeros_like(inputVectors)
    vhat_vectors = inputVectors[vhat_indices]
    vhat = np.sum(vhat_vectors,axis=0,keepdims=False)
    cost,gi,gradOut = w2vCAG(vhat,tokens[currentWord],outputVectors,dataset)
    for idx in vhat_indices: gradIn[idx] += gi
    return cost,gradIn,gradOut

def sgd_wrapper(tokens, vectors, dataset, contextSize, w2vModel=skipgram, w2vCAG=negSamplingCAG):
    batchsize,N=50,vectors.shape[0]//2
    cost,grad=0,np.zeros_like(vectors)
    inputVectors,outputVectors=vectors[:N,:],vectors[N:,:]
    for _ in range(batchsize):
#        C1 = random.randint(1,contextSize)
        C1 = 7
        currentWord,contextWords = dataset.getRandomContext(C1)
        cst,gi,go=w2vModel(currentWord,contextWords,tokens,inputVectors,outputVectors,dataset)
        cost+=cst/batchsize
        grad[:N]+=gi/batchsize
        grad[N:]+=go/batchsize
    return cost,grad

def sgd(f,x0,itrs,lr,print_every=1000):
    x=x0.copy()
    expcost,st=None,time.time()
    for i in range(itrs):
        cost,grad=f(x)
        x -= lr*grad
        expcost = cost if expcost is None else 0.95*expcost + 0.5*cost
        if i % print_every == 0:
            print("SGD i(%d) exp(%f) cost(%f) in (%d) seconds" % (i,expcost,cost,time.time()-st))
            st=time.time()
    return x

def run():
    dataset = StanfordSentiment()
    tokens_encoded = dataset.tokens()
    for k,v in tokens_encoded.items():
        if type(k) == str:
            tokens_encoded.pop(k)
            tokens_encoded[k.encode('latin1')] = v
    tokens = dict((k.decode('latin1'),v) for k,v in tokens_encoded.items())
    V,D = len(tokens),10
    random.seed(319)
    np.random.seed(419)
    vectors = np.concatenate((np.random.randn(V,D), np.zeros((V,D))), axis=0)
    vectors = sgd(lambda vecs: sgd_wrapper(tokens_encoded,vecs,dataset,7,w2vModel=skipgram), vectors, 5001, 3e-1)

def test_w2v():
    dataset = type('justadummy',(),{})()
    def dummySampleTokenIdx(): return random.randint(0,4)
    def dummyRandomContext(C):
        tokens = ['a','b','c','d','e']
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for _ in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = dummyRandomContext
    tokens = {'a':0,'b':1,'c':2,'d':3,'e':4}
    np.random.seed(419)
    vectors = np.random.randn(10,3)
    st = time.time()
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens,vecs,dataset,7,w2vModel=skipgram,w2vCAG=negSamplingCAG), vectors)
    print("SKIP NS (%f)" % (time.time()-st))
    st=time.time()
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens,vecs,dataset,7,w2vModel=skipgram,w2vCAG=negSamplingCAG_Origin), vectors)
    print("SKIP OR (%f)" % (time.time()-st))
    st=time.time()
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens,vecs,dataset,7,w2vModel=cbow,w2vCAG=negSamplingCAG), vectors)
    print("CBOW NS (%f)" % (time.time()-st))
    st=time.time()
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens,vecs,dataset,7,w2vModel=cbow,w2vCAG=negSamplingCAG_Origin), vectors)
    print("CBOW OR (%f)" % (time.time()-st))
    st=time.time()
#    gradcheck_naive(lambda vecs: sgd_wrapper(tokens,vecs,dataset,7,w2vModel=skipgram,w2vCAG=softmaxCAG), vectors)
#    gradcheck_naive(lambda vecs: sgd_wrapper(tokens,vecs,dataset,7,w2vModel=cbow,w2vCAG=softmaxCAG), vectors)
#    print()

if __name__=='__main__':
    test_w2v()
#    run()


