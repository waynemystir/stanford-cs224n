#!/usr/bin/env python

import numpy as np
import random
import time

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    x /= np.sqrt(np.sum(x**2,1,keepdims=True))
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print(x)
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    yy = softmax(outputVectors.dot(predicted)) # ŷ
    cost = -np.log(yy[target])
    yy[target] -= 1.0 # ŷ - y
    gradPred = outputVectors.T.dot(yy)
    grad = np.outer(yy, predicted)
    # np.outer handles the reshapes and .T from below for us
#    grad = predicted.reshape(predicted.shape[0],1).dot((yhat - yonehot).reshape(yhat.shape[0],1).T)
    # Also notice that we switched the arguments in np.outer from 
    # the answer to (3b) where we got vc.dot((ŷ - y).T)
    # This is because outputVectors is |V|xD rather than Dx|V|
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient01(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
#    indices = getNegativeSamples(target, dataset, K)

    ### YOUR CODE HERE
#    ns = np.random.choice(outputVectors.shape[0], K, replace=False)
#    NS = outputVectors[ns,:]
    D,NS,uo,grad = outputVectors.shape[1], outputVectors[indices[1:],:], outputVectors[target], np.zeros_like(outputVectors)
    uop,nsp = sigmoid(uo.dot(predicted)),sigmoid(NS.dot(predicted))
    cost = -np.log(uop)-np.sum(np.log(1.-nsp))
    gradPred = (uop-1.) * uo + nsp.dot(NS)
    gTmp = np.insert(nsp,0,uop-1.).reshape(K+1,1).dot(predicted.reshape(1,D)) # K+1xD
    for k in range(K+1):
        grad[indices[k]] += gTmp[k,:]
    ### END YOUR CODE

    return cost, gradPred, grad


def negSamplingCostAndGradient02(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
#    indices = getNegativeSamples(target, dataset, K)

    ### YOUR CODE HERE
#    ns = np.random.choice(outputVectors.shape[0], K, replace=False)
#    NS = outputVectors[ns,:]
    D,NS,uo,grad = outputVectors.shape[1], outputVectors[indices[1:],:], outputVectors[target], np.zeros_like(outputVectors)
    uop,nsp = sigmoid(uo.dot(predicted)),sigmoid(NS.dot(predicted))
    cost = -np.log(uop)-np.sum(np.log(1.-nsp))
    gradPred = (uop-1.) * uo + nsp.dot(NS)
    gTmp = np.insert(nsp,0,uop-1.) * np.tile(predicted, (K+1,1)).T # DxK+1
    for k in range(K+1):
        grad[indices[k]] += gTmp[:,k]
    ### END YOUR CODE

    return cost, gradPred, grad


def negSamplingCostAndGradient03(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
#    indices = getNegativeSamples(target, dataset, K)

    ### YOUR CODE HERE
#    ns = np.random.choice(outputVectors.shape[0], K, replace=False)
#    NS = outputVectors[ns,:]
    D,NS,uo,grad = outputVectors.shape[1], outputVectors[indices[1:],:], outputVectors[target], np.zeros_like(outputVectors)
    uop,nsp = sigmoid(uo.dot(predicted)),sigmoid(NS.dot(predicted))
    cost = -np.log(uop)-np.sum(np.log(1.-nsp))
    gradPred = np.insert(nsp,0,uop-1.).reshape(1,K+1).dot(np.concatenate((uo.reshape(1,D),NS))).flatten()
    gTmp = np.insert(nsp,0,uop-1.).reshape(K+1,1).dot(predicted.reshape(1,D)) # K+1xD
    for k in range(K+1):
        grad[indices[k]] += gTmp[k,:]
    ### END YOUR CODE

    return cost, gradPred, grad


def negSamplingCostAndGradient04(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
#    indices = getNegativeSamples(target, dataset, K)

    ### YOUR CODE HERE
#    ns = np.random.choice(outputVectors.shape[0], K, replace=False)
#    NS = outputVectors[ns,:]
    D,NS,uo,grad = outputVectors.shape[1], outputVectors[indices[1:],:], outputVectors[target], np.zeros_like(outputVectors)
    uop,nsp = sigmoid(uo.dot(predicted)),sigmoid(NS.dot(predicted))
    cost = -np.sum(np.log(np.insert(1.-nsp,0,uop)))
    gradPred = np.insert(nsp,0,uop-1.).reshape(1,K+1).dot(np.concatenate((uo.reshape(1,D),NS))).flatten()
    gTmp = np.insert(nsp,0,uop-1.).reshape(K+1,1).dot(predicted.reshape(1,D)) # K+1xD
    for k in range(K+1):
        grad[indices[k]] += gTmp[k,:]
    ### END YOUR CODE

    return cost, gradPred, grad


def negSamplingCostAndGradient05(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
#    indices = getNegativeSamples(target, dataset, K)

    ### YOUR CODE HERE
#    ns = np.random.choice(outputVectors.shape[0], K, replace=False)
#    NS = outputVectors[ns,:]
    D,outWords,grad = outputVectors.shape[1], outputVectors[indices,:], np.zeros_like(outputVectors)
    uop,nsp = sigmoid(outWords[0].dot(predicted)),sigmoid(outWords[1:,:].dot(predicted))
    cost = -np.log(uop)-np.sum(np.log(1.-nsp))
    gradPred = np.insert(nsp,0,uop-1.).reshape(1,K+1).dot(outWords).flatten()
    gTmp = np.insert(nsp,0,uop-1.).reshape(K+1,1).dot(predicted.reshape(1,D)) # K+1xD
    for k in range(K+1):
        grad[indices[k]] += gTmp[k,:]
    ### END YOUR CODE

    return cost, gradPred, grad


def negSamplingCostAndGradient06(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
#    indices = getNegativeSamples(target, dataset, K)

    ### YOUR CODE HERE
#    ns = np.random.choice(outputVectors.shape[0], K, replace=False)
#    NS = outputVectors[ns,:]
    D,outWords,grad = outputVectors.shape[1], outputVectors[indices,:], np.zeros_like(outputVectors)
    uop,nsp = sigmoid(outWords[0].dot(predicted)),sigmoid(outWords[1:,:].dot(predicted))
    cost = -np.log(uop)-np.sum(np.log(1.-nsp))
    tw = np.insert(nsp,0,uop-1.)
    gradPred = tw.reshape(1,K+1).dot(outWords).flatten()
    gTmp = tw.reshape(K+1,1).dot(predicted.reshape(1,D)) # K+1xD
    for k in range(K+1):
        grad[indices[k]] += gTmp[k,:]
    ### END YOUR CODE

    return cost, gradPred, grad


def negSamplingCostAndGradient07(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
#    indices = getNegativeSamples(target, dataset, K)

    ### YOUR CODE HERE
#    ns = np.random.choice(outputVectors.shape[0], K, replace=False)
#    NS = outputVectors[ns,:]
    D,outWords,grad = outputVectors.shape[1], outputVectors[indices,:], np.zeros_like(outputVectors)
    wp = sigmoid(outWords.dot(predicted))
    cost = -np.log(wp[0])-np.sum(np.log(1.-wp[1:]))
    wp[0] -= 1
    gradPred = wp.reshape(1,K+1).dot(outWords).flatten()
    gTmp = wp.reshape(K+1,1).dot(predicted.reshape(1,D)) # K+1xD
    for k in range(K+1):
        grad[indices[k]] += gTmp[k,:]
    ### END YOUR CODE

    return cost, gradPred, grad


def negSamplingCostAndGradient08(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    directions,grad,D = np.array([1] + [-1 for k in range(K)]), np.zeros_like(outputVectors), outputVectors.shape[1]
    outputWords = outputVectors[indices,:]
    delta = sigmoid(np.dot(outputWords,predicted) * directions)
    deltaMinus = (delta - 1) * directions;
    cost = -np.sum(np.log(delta));
    gradPred = np.dot(deltaMinus.reshape(1,K+1),outputWords).flatten()
    gradMin = np.dot(deltaMinus.reshape(K+1,1),predicted.reshape(1,D))

    for k in range(K+1):
        grad[indices[k]] += gradMin[k,:]
    ### END YOUR CODE

    return cost, gradPred, grad


def negSamplingCostAndGradient09(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    directions,grad,D = np.array([1] + [-1 for k in range(K)]), np.zeros_like(outputVectors), outputVectors.shape[1]
    outputWords = outputVectors[indices,:]
    δ1 = sigmoid(np.dot(outputWords,predicted) * directions)
    δ2 = (δ1 - 1) * directions;
    cost = -np.sum(np.log(δ1));
    gradPred = np.dot(δ2.reshape(1,K+1),outputWords).flatten()
    gradMin = np.outer(δ2,predicted)

    for k in range(K+1):
        grad[indices[k]] += gradMin[k,:]
    ### END YOUR CODE

    return cost, gradPred, grad


def negSamplingCostAndGradientNonVectorized(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models
    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.
    Note: See test_word2vec below for dataset's initialization.
    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    cost = 0
    z = sigmoid(np.dot(outputVectors[target], predicted))

    cost -= np.log(z)
    grad[target] += predicted * (z - 1.0)
    gradPred += outputVectors[target] * (z - 1.0)

    for k in range(K):
        samp = indices[k + 1]
        z = sigmoid(np.dot(outputVectors[samp], predicted))
        cost -= np.log(1.0 - z)
        grad[samp] += predicted * z
        gradPred += outputVectors[samp] * z
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    cword_idx = tokens[currentWord]
    vhat = inputVectors[cword_idx]

    for j in contextWords:
        u_idx = tokens[j]
        c_cost,c_grad_in,c_grad_out = word2vecCostAndGradient(vhat, u_idx, outputVectors, dataset)
        cost += c_cost
        gradIn[cword_idx] += c_grad_in
        gradOut += c_grad_out
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    predicted_indicies = [tokens[cw] for cw in contextWords]
    predicted_vectors = inputVectors[predicted_indicies]
    predicted = np.sum(predicted_vectors, axis=0)
    target = tokens[currentWord]
    cost, gradIn_predicted, gradOut = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
    for i in predicted_indicies:
        gradIn[i] += gradIn_predicted
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0] // 2
    inputVectors = wordVectors[:N,:]
    outputVectors = wordVectors[N:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N, :] += gin / batchsize / denom
        grad[N:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    startTime = time.time()
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient01),
        dummy_vectors)
    print("nSCAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG1111111111111111 took %f seconds" % (time.time() - startTime))
    startTime = time.time()
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient02),
        dummy_vectors)
    print("nSCAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG2222222222222222 took %f seconds" % (time.time() - startTime))
    startTime = time.time()
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient03),
        dummy_vectors)
    print("nSCAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG3333333333333333 took %f seconds" % (time.time() - startTime))
    startTime = time.time()
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient04),
        dummy_vectors)
    print("nSCAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG4444444444444444 took %f seconds" % (time.time() - startTime))
    startTime = time.time()
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient05),
        dummy_vectors)
    print("nSCAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG5555555555555555 took %f seconds" % (time.time() - startTime))
    startTime = time.time()
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient06),
        dummy_vectors)
    print("nSCAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG6666666666666666 took %f seconds" % (time.time() - startTime))
    startTime = time.time()
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient07),
        dummy_vectors)
    print("nSCAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG7777777777777777 took %f seconds" % (time.time() - startTime))
    startTime = time.time()
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient08),
        dummy_vectors)
    print("nSCAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG8888888888888888 took %f seconds" % (time.time() - startTime))
    startTime = time.time()
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient09),
        dummy_vectors)
    print("nSCAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG9999999999999999 took %f seconds" % (time.time() - startTime))
    startTime = time.time()
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradientNonVectorized),
        dummy_vectors)
    print("nSCAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGNVNVNVNVNVNVNVNV took %f seconds" % (time.time() - startTime))
    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient01),
        dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient01))
    print(cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient01))


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
