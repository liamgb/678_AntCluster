import nltk
import nltk.data
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.parse.stanford import StanfordDependencyParser

from pprint import pprint
from pprint import pformat

import random
import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
import operator

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

import ast
from sklearn.cluster import MeanShift, estimate_bandwidth

#text file of story
file = "LRRH.txt"

#smaller chunking = more clusters/events
chunkingSize = 65
#65 = 50 clusters/events in Falling.txt
#255 = 14

#show the plot of entities and clusters? (True/False)
showPlot = True

#trim elements that appear only once
trimUnique = True


def run(text):
    #Get list of entities
    neList = createNElist(text)

    #Find the clusters of entities that might define events
    clusters, entities = eventClusters(text, neList)

    #Isolate the sections of the story where the clusters are
    tokenClusters = clusterTokens(text, clusters)

    #EXTRACT EVENTS!!!
    events = list()
    d = TreebankWordDetokenizer()

    for c in range(len(tokenClusters)):
        sent = d.detokenize(tokenClusters[c])

        print(str(c) + " \"" + sent + "\"\n")

        nlp = spacy.load('en_core_web_sm')
        doc = nlp(sent)
        '''
        for token in doc:
            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                  token.shape_, token.is_alpha, token.is_stop)
        '''
        subj = list()
        obj = list()
        verb = list()
        for token in doc:
            if token.dep_ == "nsubj" or token.dep_ == "csubj" or token.dep_ == "nsubjpass" or token.dep_ == "csubjpass" or token.dep_ == "pobj":
                if token.lemma_ == "-PRON-":
                    subj.append(token.text.lower())
                else:
                    subj.append(token.lemma_)
            if token.dep_ == "obj" or token.dep_ == "dobj":
                if token.lemma_ == "-PRON-":
                    obj.append(token.text.lower())
                else:
                    obj.append(token.lemma_)
            if token.dep_ == "ROOT":
                if token.lemma_ == "-PRON-":
                    verb.append(token.text.lower())
                else:
                    verb.append(token.lemma_)

        #print("Subjects" + str(subj))
        #print("Objects" + str(obj))
        #print("Actions" + str(verb))

        for e in verb:
            if e == "be":
                verb.remove("be")

        #print(most_common(subj))
        #print(most_common(verb))
        #print(most_common(obj))
        events.append([most_common(subj),most_common(verb),most_common(obj), "LOCATION", c])

    pprint(events)

def clusterTokens(text, clusters):
    tokens = nltk.word_tokenize(text)

    tokenClusters = list()
    for c in range(len(clusters)):
        stop = max(clusters[c])
        start = min(clusters[c])
        while True:
            if start == 0:
                break
            if tokens[start] == '.':
                start = start + 1
                break
            start = start - 1
        while True:
            if stop == len(tokens):
                break
            if tokens[stop] == '.':
                break
            stop = stop + 1
        tokenClusters.append(tokens[start:stop+1])

    return tokenClusters

def eventClusters(text, neList):
    tokens = nltk.word_tokenize(text)

    order = list()
    for t in tokens:
        if t.lower() in neList.keys() and t.lower() not in order:
            order.append(t.lower())

    #print(order)

    plt.axis([0,len(tokens),-1,len(order)])
    plt.ylabel('Entities')
    plt.xlabel('Location')
    plt.title('Clustering of Entities in Story')

    elementLocs = list()
    for i in range(len(order)):
        for j in range(len(tokens)):
            if order[i] == tokens[j].lower():
                plt.plot([j],[i],'ko')
                elementLocs.append(j)

    #for i in range(len(order)):
    #    plt.plot([0, len(tokens)], [i,i], 'k')

    textstr = ""
    for x in range(len(order)):
        textstr = textstr + " " + str(x) + ": " + order[x] + " " + str(neList[order[x]]) + "\n"
    plt.text(0.02, 0.01, textstr, fontsize=6.7, transform=plt.gcf().transFigure)

    #plt.show()

    X = np.array(list(zip(elementLocs,np.zeros(len(elementLocs)))))

    '''CLUSTER CALL HERE!!!'''
    #bandwidth = estimate_bandwidth(X, quantile=0.1)
    ms = MeanShift(bandwidth=chunkingSize, bin_seeding=True)
    #ms = MeanShift(bandwidth=chunkingSize, bin_seeding=True)
    #print("BANDWIDTH: " + str(bandwidth))

    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    '''CLUSTER CALL END!!!'''

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    clusters = list()
    starts = list()
    for k in range(n_clusters_):
        my_members = labels == k
        #print("Cluster {0}: {1}".format(k, X[my_members, 0]))
        temp = ast.literal_eval("{1}".format(k, X[my_members, 0]).replace('.',','))
        clusters.append(temp)
        plt.plot([min(temp), min(temp)], [0,len(order)], 'r')
        starts.append(min(temp))

    #print(len(clusters))

    clusters.sort(key = getmin)
    if showPlot:
        plt.show()

    return clusters, order

def getmin(element):
    return min(element)

def createNElist(text):
    foci = dict()

    sent = nltk.pos_tag(nltk.word_tokenize(text))

    for x in sent:
        if x[0].lower() == 'i' or x[0].lower() == 'he' or x[0].lower() == 'she':
            if x[0].lower() in foci.keys():
                foci[x[0].lower()] = foci[x[0].lower()] + 1
            else:
                foci[x[0].lower()] = 1

    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)
    iob_tagged = tree2conlltags(cs)
    parsed = pformat(iob_tagged)
    parsed = ast.literal_eval(parsed)

    tempString = ""
    for x in parsed:
        if x[2] == 'B-NP' or x[2] == 'I-NP':
            tempString = tempString + x[0].lower() + " "
        if x[2] == 'O' and len(tempString) > 0:
            tempString = tempString.replace('a ', '').replace('an ', '').replace('no ', '').replace('this ', '').replace('the ', '')
            if tempString.rstrip() in foci.keys():
                foci[tempString.rstrip()] = foci[tempString.rstrip()] + 1
            else:
                foci[tempString.rstrip()] = 1
            tempString = ""

    doc = nlp(text)
    parsed = pformat([(X.text, X.label_) for X in doc.ents])
    parsed = ast.literal_eval(parsed)
    for x in parsed:
        if x[1] == 'PERSON' or x[1] == 'ORG' or x[1] == 'PRODUCT' or x[1] == 'LOC' or x[1] == 'FAC':
            tempString = x[0].lower().replace('a ', '').replace('an ', '').replace('no ', '').replace('this ', '').replace('the ', '')
            if tempString in foci.keys():
                foci[tempString] = foci[tempString] + 1
            else:
                foci[tempString] = 1

    #if trimUnique:
    #    foci = {k: v for k, v in foci.items() if v != 1}

    foci["we"] = 0

    for k in foci.keys():
        foci[k] = 0
    for t in nltk.word_tokenize(text):
        if t.lower() in foci.keys():
            foci[t.lower()] = foci[t.lower()] + 1

    #pprint(foci.keys())

    return foci

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

def plotClusterComparison14(max,annotations,color):
    for x in annotations:
        plt.plot([x, x], [0,max], color)

def main():
    text = open(file, "r").read()
    run(text)

if __name__ == '__main__':
    main()
