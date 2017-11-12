import numpy as np
import random
import time
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices

def negSamplingCAG(pred, target, outputVectors, dataset, K=10, verbose=False):
    ids = [target] + [i for i in getSampleIdxs(target,dataset,K)]
    grad,outputWords,dirs = np.zeros_like(outputVectors),outputVectors[ids,:],np.array([[1] + [-1 for _ in range(K)]])
    δ1 = sigmoid(outputWords.dot(pred) * dirs)
    cost,δ2 = -np.sum(np.log(δ1)),(δ1-1) * dirs
    gradPred = δ2.reshape(1,K+1).dot(outputWords).flatten()
#    gradPred = outputWords.T.dot(δ2.reshape(K+1,1)).flatten()
    gTmp = δ2.reshape(K+1,1).dot(pred.reshape(1,outputVectors.shape[1]))
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
    vhat_indices,gradIn = [tokens[cw] for cw in contextWords],np.zeros_like(inputVectors)
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

def sgd(f,x0,itrs,learning_rate,print_every=1000):
    x = x0.copy()
    expcost,cost = None,0.
    st = time.time()
    for i in range(itrs):
        cost,grad  = f(x)
        x -= learning_rate * grad
        expcost = cost if expcost is None else 0.9 * expcost + 0.1 * cost
        if i % print_every == 0:
            print("SGD (%d) expcost (%f) cost (%f) in (%f) seconds" % (i,expcost,cost,time.time()-st))
            st = time.time()
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
    np.random.seed(41717)
    vectors = np.concatenate((np.random.randn(V,D),np.zeros((V,D))),axis=0)
    start_time = time.time()
    vectors = sgd(lambda vecs: sgd_wrapper(tokens_encoded,vecs,dataset,5,w2vmodel=skipgram), vectors, 24001, 3e-1)
    print("w2v run in (%f) seconds" % (time.time()-start_time))
    visualize_words=['smart','dumb','tall','short','good','bad','king','queen','man','woman']
    visualize_indices = [tokens[w] for w in visualize_words]
    visualize_vecs = vectors[visualize_indices, :]
    temp = (visualize_vecs - np.mean(visualize_vecs, axis=0))
    covariance = 1.0 / len(visualize_indices) * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2])
    for i in range(len(visualize_words)):
        plt.text(coord[i,0], coord[i,1], visualize_words[i],
            bbox=dict(facecolor='green', alpha=0.1))
    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))
    plt.savefig('q3_word_vectors.png')

def test_w2v():
    random.seed(319)
    np.random.seed(419)
    dataset = type('dummy',(),{})()
    def dummySampleIdx():
        return random.randint(0,2)
    def dummyRandomContext(C):
        tokens = ['a','b','c']
        return tokens[random.randint(0,2)], [tokens[random.randint(0,2)] for _ in range(2*C)]
    dataset.sampleTokenIdx = dummySampleIdx
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
#    test_w2v()
    run()
