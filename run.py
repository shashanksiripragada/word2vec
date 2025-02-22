import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import argparse
from sgd import *
from word2vec import *

random.seed(314)
dataset = StanfordSentiment()
word2Ind = dataset.tokens()
nWords = len(word2Ind)

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, 
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient)

        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


def train(dimVectors=10, C=5, lr=0.3):
    random.seed(31415)
    np.random.seed(9265)
    startTime=time.time()

    #init word vectors of size 2VxD #center, outside vecs
    wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors)-0.5)/dimVectors,  
                                   np.zeros((nWords, dimVectors))),
                                   axis=0)
    #sgd(f, x0, step, iterations, postprocessing=None,useSaved=False, PRINT_EVERY=10):
    wordVectors = sgd(
                    lambda vec: word2vec_sgd_wrapper(skipgram, word2Ind, vec, dataset, 
                                        C, negSamplingLossAndGradient),
                    wordVectors, 
                    lr=0.3, 
                    iterations=40000, 
                    postprocessing=None, 
                    useSaved=True, 
                    PRINT_EVERY=10
                    )

    print("sanity check: cost at convergence should be around or below 10")
    print("training took %d seconds" % (time.time() - startTime))

def eval():
    _, wordVectors, _ = load_saved_params()
    visualizeWords = [
    "the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
    "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
    "annoying"]
    visualizeIdx = [word2Ind[word] for word in visualizeWords]
    visualizeVecs = wordVectors[visualizeIdx, :]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", dest="train", action="store_true",
                       help="train on sst dataset")
    group.add_argument("--eval", dest="eval", action="store_true",
                       help="use trained vectors for evaluation")
    args = parser.parse_args()
    if args.train:
        train()
    if args.eval:
        eval()