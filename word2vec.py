import numpy as np
import random
from utils.utils import softmax, sigmoid, normalizeRows
from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp

def naiveSoftmaxLossAndGradient(
    centerWordVec, outsideWordIdx,
    outsideVectors, dataset
):
    v_c = centerWordVec #(D, )
    o = outsideWordIdx 
    U = outsideVectors #(V, D)
    y_hat = softmax(np.dot(U, v_c)) #(V, )
    loss = -np.log(y_hat[o])

    # dv_c = u.(y_hat-y)
    delta = y_hat #(V, )
    delta[o] -= 1 
    gradCenterVec = np.dot(U.T, delta) #(D, ) 
    gradOutsideVecs = np.dot(delta[:, np.newaxis], v_c[np.newaxis, :])#(V, D)

    return loss, gradCenterVec, gradOutsideVecs

def getNegativeSamples(outsideWordIdx, dataset, K):
    negsampleinds = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negsampleinds[k] = newidx

    return negsampleinds

def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    v_c = centerWordVec #(W,)
    o = outsideWordIdx 
    U = outsideVectors #VxW
    u_o = U[o]
    u_k = U[negSampleWordIndices]
    loss = 0.0
    gradCenterVec = np.zeros(v_c.shape)
    gradOutsideVecs = np.zeros(U.shape)
    z = sigmoid(np.dot(u_o, v_c))
    z_k = sigmoid(-np.dot(u_k, v_c))
    loss += -np.log(z)
    loss += -np.sum(np.log(z_k))
    
    gradCenterVec += (z-1) * u_o
    gradCenterVec -= np.dot((z_k-1), u_k)

    gradOutsideVecs[o] += (z-1) * v_c
    
    for i, ind in enumerate(negSampleWordIndices):
        gradOutsideVecs[ind] += (1 - z_k[i]) * v_c 

    return loss, gradCenterVec, gradOutsideVecs

def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    cur_center_word_idx = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[cur_center_word_idx]

    for o in outsideWords:
        outsideWordIdx = word2Ind[o]
        l, gradCenter, gradOutside = \
        word2vecLossAndGradient(centerWordVec, outsideWordIdx, \
                                outsideVectors, dataset)
        loss += l
        gradCenterVecs[cur_center_word_idx] += gradCenter
        gradOutsideVectors += gradOutside
    
    return loss, gradCenterVecs, gradOutsideVectors

def cbow(currentCenterWord, windowSize, outsideWords, word2Ind,
        centerWordVectors, outsideVectors, dataset,
        word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    
    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)
    
    out_idxs = [word2Ind[word] for word in outsideWords]
    outVecs = outsideVectors[out_idxs]
    cbow_centerVec = np.sum(outVecs, axis=0)     
    cbow_outVecIdx = word2Ind[currentCenterWord]
    loss, gradCenter, gradOutsideVectors = word2vecLossAndGradient(
                                        cbow_centerVec, cbow_outVecIdx, 
                                        outsideVectors, dataset)

    for i in out_idxs:
        gradCenterVecs[i] += gradCenter

    return loss, gradCenterVecs, gradOutsideVectors
