import sys, os, re
from cPickle import load, dump
from termcolor import colored
import numpy as np
from lda import *
import math

def info(s):
  print colored(s, 'yellow')

def readDocs(directory):
  docs = []
  pattern = re.compile('[\W_]+')
  for root, dirs, files in os.walk(directory):
    for filename in files:
      # print colored(filename, 'red')
      with open(root + '/' + filename) as f:
        header = True
        content = []
        for line in f:
          if not header:
            words = [pattern.sub('', w.lower()) for w in line.split()]
            content.extend(words)
          elif line.startswith('Lines: '):
            header = False
        
        docs.append(content)
  return docs

def preprocess(directory):
  info('Reading corpus')
  docs = readDocs(directory)
  stopwords = load(open('stopwords.pickle'))

  info('Building vocab')
  vocab = set()
  for doc in docs:
    for w in doc:
      if len(w) > 1 and w not in stopwords:
        vocab.add(w)
  
  vocab       = list(vocab)
  lookupvocab = dict([(v, k) for (k, v) in enumerate(vocab)])

  info('Building BOW representation')
  m = np.zeros((len(docs), len(vocab)))
  for d, doc in enumerate(docs):
    for w in doc:
      if len(w) > 1 and w not in stopwords:
        m[d, lookupvocab[w]] += 1
  return m, vocab


def discoverTopics(n = 20):
  matrix, vocab = preprocess('../data/20_newsgroup')
  # matrix, vocab = preprocess('../data/toy2')
  sampler = LdaSampler(n)

  info('Starting!')
  for it, phi in enumerate(sampler.run(matrix, 100)):
      print colored("Iteration %s" % it, 'yellow')
      print "Likelihood", sampler.loglikelihood()
      
      
      for topicNum in xrange(n):
        s = colored(topicNum, 'green')
        words = [(proba, w) for (w, proba) in enumerate(phi[topicNum, :]) if proba > 0]
        words = sorted(words, reverse = True)
        for i in range(10):
          proba, w = words[i]
          s += ' ' + vocab[w]
        print s

discoverTopics()