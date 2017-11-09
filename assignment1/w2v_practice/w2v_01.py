import numpy as np
import random
import time
from utils.treebank import StanfordSentiment

def gradcheck_naive(f,x):
    rndst = random.getstate()
    random.setstate(rndst)
    _,grad = f(x)
    h = 1e-4
    it = np.nditer(x,flags=['multi_index'])
    while not it.finished:
        ix = it.multi_index
        x[ix] -= h
        random.setstate(rndst)
        fxb = f(x)[0]
        x[ix] += 2*h
        random.setstate(rndst)
        fxa = f(x)[0]
        x[ix] -= h
        numgrad = (fxa-fxb)/(2*h)
        relerror = abs(numgrad - grad[ix]) / max(1, abs(fxa), abs(fxb))
        if relerror > 1e-5:
            print("Gradient Check Found Error at %s. Analytic grad (%f) and numerical grad (%f)" % (ix, grad[ix], numgrad))
            return
        it.iternext()
    print("Gradient check is good!!!!!")

def softmax(x):
    p = np.exp(x - np.max(x,axis=x.ndim-1,keepdims=True))
    return p / np.sum(p, axis=p.ndim-1,keepdims=True)

def softmaxCAG(pred, target, outputVectors, dataset, K=10, verbose=False):
    yy = softmax(outputVectors.dot(pred)) # ŷ
    cost = -np.log(yy[target])
    yy[target] -= 1 # ŷ - y
    gradPred = outputVectors.T.dot(yy)
    grad = np.outer(yy,pred)
    return cost, gradPred, grad

def sigmoid(z):
    return 1./(1+np.exp(-z))

def sigmoid_grad(f):
    return f * (1 - f)

def getSampleIdxs(target,dataset,K):
    indices = [None] * K
    for k in range(K):
        newidx = dataset.getSampleIdx()
        while newidx == target:
            newidx = dataset.getSampleIdx()
        indices[k] = newidx
    return indices

def negSamplingCAG(pred, target, outputVectors, dataset, K=10, verbose=False):
    ids = [target]
    ids.extend(getSampleIdxs(target,dataset,K))
    D,grad,outputWords = outputVectors.shape[1],np.zeros_like(outputVectors),outputVectors[ids,:]
    directions = np.array([[1] + [-1 for _ in range(K)]])
    if verbose: print("negSamplingCAG opW {} pred {} dirs {}".format(outputWords.shape,pred.shape,directions.shape))
    δ1 = sigmoid(outputWords.dot(pred) * directions)
    δ2 = (δ1-1) * directions
    cost = -np.sum(np.log(δ1))
    gradPred = δ2.reshape(1,K+1).dot(outputWords).flatten()
    gTmp = δ2.reshape(K+1,1).dot(pred.reshape(1,D))
    for k in range(K+1):
        grad[ids[k]] += gTmp[k,:]
    return cost, gradPred, grad

def skipgram(currentWord, contextWords, tokens, inputVectors, outputVectors, dataset, w2vCAG=negSamplingCAG, verbose=False):
    vhat,cost,gradIn,gradOut = inputVectors[tokens[currentWord]],0,np.zeros_like(inputVectors),np.zeros_like(outputVectors)
    for word in contextWords:
        cst,gin,got = w2vCAG(vhat, tokens[word], outputVectors, dataset, verbose=verbose)
        cost += cst
        gradIn[tokens[currentWord]] += gin
        gradOut += got
    return cost, gradIn, gradOut

def cbow(currentWord, contextWords, tokens, inputVectors, outputVectors, dataset, w2vCAG=negSamplingCAG, verbose=False):
    cost,gradIn = 0,np.zeros_like(inputVectors)
    vhat_indices = [tokens[cw] for cw in contextWords]
    vhat_vectors = inputVectors[vhat_indices,:]
    vhat = np.sum(vhat_vectors,axis=0)
    cost,gin,gradOut = w2vCAG(vhat, tokens[currentWord], outputVectors, dataset, verbose=verbose)
    for vi in vhat_indices:
        gradIn[vi] += gin
    return cost, gradIn, gradOut

def sgd_wrapper(tokens, vectors, dataset, contextSize, w2vmodel=skipgram, w2vCAG=negSamplingCAG, verbose=False):
    batchsize,N=50,vectors.shape[0]//2
    inputVectors,outputVectors = vectors[:N,:],vectors[N:,:]
    cost,grad = 0,np.zeros_like(vectors)
    for _ in range(batchsize):
        C1 = random.randint(1,contextSize)
        currentWord,contextWords = dataset.getRandomContext(C1)
        cst,gin,got = w2vmodel(currentWord,contextWords,tokens,inputVectors,outputVectors,dataset,w2vCAG=w2vCAG,verbose=verbose)
        cost += cst/batchsize
        grad[:N,:] += gin/batchsize
        grad[N:,:] += got/batchsize
    return cost,grad

def test_w2v():
    random.seed(319)
    np.random.seed(419)
    dataset = type('dummy',(),{})()
    def dummySampleIdx():
        return random.randint(0,2)
    def dummyRandomContext(C):
        tokens = ['a','b','c']
        return tokens[random.randint(0,2)], [tokens[random.randint(0,2)] for _ in range(2*C)]
    dataset.getSampleIdx = dummySampleIdx
    dataset.getRandomContext = dummyRandomContext
    tokens = {'a':0,'b':1,'c':2}
    vectors = np.random.randn(6,2)
    print("================================ Gradient check skipgram negSampling ================================")
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens, vecs, dataset, 17, w2vmodel=skipgram, w2vCAG=negSamplingCAG, verbose=False), vectors)
    print("================================ Gradient check cbow negSampling ================================")
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens, vecs, dataset, 17, w2vmodel=cbow, w2vCAG=negSamplingCAG, verbose=False), vectors)
    print("================================ Gradient check skipgram softmax ================================")
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens, vecs, dataset, 17, w2vmodel=skipgram, w2vCAG=softmaxCAG, verbose=False), vectors)

if __name__=="__main__":
    test_w2v()






