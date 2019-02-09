import nltk
import nltk.data
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.parse.stanford import StanfordDependencyParser
import spacy
from spacy import displacy
from collections import Counter
from pprint import pprint
from pprint import pformat
import ast
import en_core_web_sm
nlp = en_core_web_sm.load()
import re
import matplotlib.pyplot as plt
import numpy as np

file = "Falling.txt"

def find_elements(text, full=False, trim=True, low_trim_limit = 2, high_trim_limit = 2000):
    sent = nltk.pos_tag(nltk.word_tokenize(text))
    elements = dict()
    if full: #do all nouns
        for x in sent:
            if x[1] == "NN" or x[1] == "NNS" or x[1] == "NNP" or x[1] == "NNPS" or x[1] == "PRP":
                elements[x[0].lower()] = 0
    else: #do only NE + extra
        for x in sent:
            if x[1] == "PRP":
                elements[x[0].lower()] = 0

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
                tempString = tempString.rstrip()
                tempString = re.sub("^a ", "", re.sub("^an ", "", re.sub("^no ", "", re.sub("^this ", "", re.sub("^the ", "", tempString)))))
                elements[tempString.rstrip().lower()] = 0
                tempString = ""

        doc = nlp(text)
        parsed = pformat([(X.text, X.label_) for X in doc.ents])
        parsed = ast.literal_eval(parsed)
        for x in parsed:
            if x[1] == 'PERSON' or x[1] == 'ORG' or x[1] == 'PRODUCT' or x[1] == 'LOC' or x[1] == 'FAC':
                tempString = x[0].lower().replace('a ', '').replace('an ', '').replace('no ', '').replace('this ', '').replace('the ', '').replace('\n', '')
                elements[tempString] = 0

    if trim:
        text = text.lower()
        for x in elements.keys():
            elements[x] = my_count(text, x)
        elements = {k: v for k, v in elements.items() if v > low_trim_limit}
        elements = {k: v for k, v in elements.items() if v <= high_trim_limit}

    pprint(elements)
    return elements

def my_count(string, substring):
    substring = substring
    string_size = len(string)
    substring_size = len(substring)
    count = 0
    for i in range(0,string_size-substring_size+1):
        if string[i:i+substring_size] == substring:
            count+=1
    return count

def create_feature_array(text, elements):
    tokens = nltk.word_tokenize(text.lower())
    order = list()
    for t in tokens:
        if t in elements.keys() and t.lower() not in order:
            order.append(t.lower())

    plt.axis([0,len(tokens),-1,len(order)])
    plt.ylabel('Entities')
    plt.xlabel('Location')
    plt.title('Clustering of Entities in Story')

    elementLocs = list()
    elementIDs = list()
    for i in range(len(order)):
        for j in range(len(tokens)):
            if order[i] == tokens[j].lower():
                plt.plot([j],[i],'ko')
                elementLocs.append(j)
                elementIDs.append(i)

    textstr = ""
    for x in range(len(order)):
        textstr = textstr + " " + str(x) + ": " + order[x] + " " + str(elements[order[x]]) + "\n"
    print(textstr)
    plt.text(0.02, 0.01, textstr, fontsize=6.7, transform=plt.gcf().transFigure)

    plt.show()

    return elementLocs, elementIDs

def raw_features_to_np_array(element_locations, element_order):
    array = np.zeros([len(element_locations),2])
    for x in range(len(element_locations)):
        array[x,0] = element_locations[x]
        array[x,1] = element_order[x]
    np.set_printoptions(threshold=np.nan, suppress=True)
    pprint(array)
    return array


def preprocess(file):
    text = open(file, "r").read()
    elements = find_elements(text)
    element_locations, element_order = create_feature_array(text, elements)
    raw_features_to_np_array(element_locations, element_order)

if __name__ == '__main__':
    preprocess(file)
