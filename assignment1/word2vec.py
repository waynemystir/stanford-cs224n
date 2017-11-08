import numpy as np
import random
from utils.treebank import StanfordSentiment
import time

def gradcheck_naive(f,x):
    rndst = random.getstate()
    random.setstate(rndst)
    fx,grad = f(x)
    itr = np.nditer(x, flags=['multi_index'])
    h = 1e-4
    while not itr.finished:
        ix = itr.multi_index
        x[ix] -= h
        random.setstate(rndst)
        fxb = f(x)[0]
        x[ix] += 2*h
        random.setstate(rndst)
        fxa = f(x)[0]
        x[ix] -= h
        numgrad = (fxa-fxb)/(2*h)
        reldiff = abs(numgrad-grad[ix]) / max(1,abs(fxa),abs(fxb))
        if reldiff>1e-5:
            print("Gradient check failed at {}".format(ix))
            return
        itr.iternext()
    print("Gradient check passed!!!")

def softmax(x):
    p = np.exp(x - np.max(x,axis=x.ndim-1,keepdims=True))
    return p/np.sum(p,axis=x.ndim-1,keepdims=True)

def softmaxCAG(pred, target, outputVectors, dataset, K=10,verbose=False):
    yy = softmax(outputVectors.dot(pred)) # ŷ
    cost = -np.log(yy[target])
    yy[target] -= 1 # ŷ - y
    gradPred = outputVectors.T.dot(yy)
    # np.outer is slow for smaller vectors like that
#    grad = np.outer(yy,pred)
    grad = yy.reshape(outputVectors.shape[0],1).dot(pred.reshape(1,outputVectors.shape[1]))
    return cost, gradPred, grad

def getSampleIndices(target,dataset,K):
    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices

def sigmoid(z):
    return 1./(1+np.exp(-z))

def sigmoid_grad(f):
    return f * (1-f)

def negSamplingCAG(pred, target, outputVectors, dataset, K=10,verbose=False):
    ids = [target]
    ids.extend(getSampleIndices(target,dataset,K))
    D,directions,grad,outputWords = outputVectors.shape[1],np.array([[1] + [-1 for _ in range(K)]]),np.zeros_like(outputVectors),outputVectors[ids,:]
    if verbose: print("negSampling oW({}) pred({}) dir({}) oW.pred({})".format(outputWords.shape,pred.shape,directions.shape,outputWords.dot(pred).shape))
    δ1 = sigmoid(outputWords.dot(pred) * directions)
    δ2 = (δ1-1) * directions
    cost = -np.sum(np.log(δ1))
    gradPred = δ2.reshape(1,K+1).dot(outputWords).flatten()
    gTmp = δ2.reshape(K+1,1).dot(pred.reshape(1,D))
    # np.outer is faster for very large vectors, but much slower for small vectors like these
#    gTmp = np.outer(δ2,pred)
    for k in range(K+1):
        grad[ids[k]] += gTmp[k,:]
    return cost,gradPred,grad

def skipgram(currentWord, contextWords, tokens, inputVectors, outputVectors, dataset, w2vfun=negSamplingCAG,verbose=False):
    cost,gradIn,gradOut = 0,np.zeros_like(inputVectors),np.zeros_like(outputVectors)
    for word in contextWords:
        if verbose: print("skipgram-skipgram word({}) tok({})".format(word,tokens[word]))
        cst,gi,go = w2vfun(inputVectors[tokens[currentWord]], tokens[word], outputVectors, dataset, verbose=verbose)
        cost += cst
        gradIn[tokens[currentWord]] += gi
        gradOut += go
    return cost,gradIn,gradOut

def cbow(currentWord, contextWords, tokens, inputVectors, outputVectors, dataset, w2vfun=negSamplingCAG,verbose=False):
    cost,gradIn = 0,np.zeros_like(inputVectors)
    pred_ids = [tokens[cw] for cw in contextWords]
    pred_vecs = inputVectors[pred_ids]
    vhat = np.sum(pred_vecs,axis=0)
    if verbose: print("CBOW-CBOW inputVecs\n{}\npred_ids({})\npred_vecs\n{}\nvhat\n{}".format(inputVectors,pred_ids,pred_vecs,vhat))
    cost,gin,gradOut = w2vfun(vhat, tokens[currentWord], outputVectors, dataset, verbose=verbose)
    for idx in pred_ids:
        gradIn[idx] += gin
    return cost,gradIn,gradOut

def sgd_wrapper(tokens, vectors, C, dataset,soc=skipgram,w2vfun=negSamplingCAG,verbose=False):
    batchsize,cost,grad,N,C1 = 50,0.,np.zeros_like(vectors),vectors.shape[0] // 2,random.randint(1,C)
    if verbose: print("sgd_wrapper C({}) C1({})".format(C,C1))
    inVecs,outVecs = vectors[:N,:],vectors[N:,:]
    for _ in range(batchsize):
        currentWord,contextWords = dataset.getRandomContext(C1)
        cst,grad_in,grad_out = soc(currentWord,contextWords,tokens,inVecs,outVecs,dataset,w2vfun=w2vfun,verbose=verbose)
        cost += cst/batchsize
        grad[:N,:] += grad_in/batchsize
        grad[N:,:] += grad_out/batchsize
    return cost,grad

def sgd(f,x0,iters,learning_rate,postprocess=None,print_every=1000):
    x = x0.copy()
    expcost,cost = None,0
    for i in range(iters):
        cost,grad = f(x)
        x -= learning_rate * grad
        if expcost is None: expcost = cost
        else: expcost = 0.95 * expcost + 0.05 * cost
        if i % print_every == 0: print("sgd({}) expcost({}) cost({})".format(i,expcost,cost))
    return x, cost

def run():
    random.seed(319)
    dataset = StanfordSentiment()
    tokens_encoded = dataset.tokens()
    for k,v in tokens_encoded.items():
        if type(k) == str:
            tokens_encoded.pop(k)
            tokens_encoded[k.encode('latin1')] = v
    tokens = dict((k.decode('latin1'),v) for k,v in tokens_encoded.items())
    nWords = len(tokens)
    dimVectors = 10
    C = 5
    random.seed(31919)
    np.random.seed(41717)
    start_time = time.time()
    vectors = np.concatenate((np.random.randn(nWords,dimVectors), np.zeros((nWords,dimVectors))), axis=0)
    vectors,cost = sgd(lambda vecs: sgd_wrapper(tokens_encoded,vecs,C,dataset,soc=skipgram), vectors, 40000, 3e-1)
    print("SGD finished in ({}) seconds with cost ({})".format(time.time()-start_time, cost))

def test_w2v():
#    random.seed(123)
    np.random.seed(456)
    tokens = {'a':0,'b':1,'c':2}
    vectors = np.random.randn(6,2)
    def dummySampleTokenIdx():
        return random.randint(0,2)
    def dummyRandomContext(C):
        tokens=['a','b','c']
        return tokens[random.randint(0,2)],[tokens[random.randint(0,2)] for _ in range(2*C)]
    dataset = type('dummy',(),{})()
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = dummyRandomContext
    print("============================ Gradient check skipgram negSamplingCAG ============================")
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens, vecs, 9, dataset, soc=skipgram, w2vfun=negSamplingCAG, verbose=False), vectors)
    print("============================ Gradient check skipgram softmaxCAG ============================")
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens, vecs, 9, dataset, soc=skipgram, w2vfun=softmaxCAG, verbose=False), vectors)
    print("============================ Gradient check CBOW negSamplingCAG ============================")
    gradcheck_naive(lambda vecs: sgd_wrapper(tokens, vecs, 7, dataset, soc=cbow, w2vfun=negSamplingCAG, verbose=False), vectors)

if __name__=="__main__":
    test_w2v()
#    run()
