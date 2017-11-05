#!/usr/bin/env python

import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from q3_word2vec import *
from q3_sgd import *
from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens_encoded = dataset.tokens()
strcnt,bytcnt,othcnt,unks = 0,0,0,0
for k,v in tokens_encoded.items():
    if type(k) == str:
        strcnt+=1
        print("the string is (%s)" % (k))
        tokens_encoded.pop(k)
        tokens_encoded[k.encode('latin1')] = v
    elif type(k) == bytes:
        bytcnt+=1
        if k == b'unk': print("UNKUNKUNKUNKUNKUNKUNKUNKUNKUNKUNK")
    else: othcnt+=1
print("str(%d)byt(%d)oth(%d)" % (strcnt,bytcnt,othcnt))
tokens = dict((k.decode('latin1'),v) for (k,v) in tokens_encoded.items())
nWords = len(tokens)

print("SCREEN ({})".format(tokens['screenwriter']))
print("SCRBBB ({})".format(tokens[b'screenwriter']))
print("nWords ({})".format(nWords))
fws = take(20, tokens.items())
print("first words {}".format(fws))
print("the ({})".format(tokens['the']))
