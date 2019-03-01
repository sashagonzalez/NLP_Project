#!/usr/bin/env python

import re, sys, nltk, operator
from nltk.stem.wordnet import WordNetLemmatizer

from qa_engine.base import QABase
    
def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root':
            return node
    return None
    
def find_node(word, graph):
    for node in graph.nodes.values():
        if node["word"] == word:
            return node
    return None
    
def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph)
        
    return results


def find_answer(qgraph, sgraph):
    qmain = find_main(qgraph)
    qword = qmain["word"]
    print(qword)
    snode = find_node(qword, sgraph)

    for node in sgraph.nodes.values():
        if node.get('head', None) == snode["address"]:
            if node['rel'] == "nmod":
                deps = get_dependents(node, sgraph)
                deps = sorted(deps+[node], key=operator.itemgetter("address"))
                return " ".join(dep["word"] for dep in deps)
    return None


if __name__ == '__main__':
    driver = QABase()

    # Get the first question and its story
    q = driver.get_question("fables-01-1")
    print("question:", q["text"])
    """
    ==> Where was the crow sitting?
    """

    story = driver.get_story(q["sid"])
    sents = nltk.sent_tokenize(story["sch"])
    print("story second sentence: ", sents[1])
    """
    ==> The crow was sitting on a branch of a tree.
    """

    # get the dependency graph of the first question
    qgraph = q["dep"]
    #print("qgraph:", qgraph)

    # The answer is in the second sentence
    # You would have to figure this out like in the chunking demo
    sgraph = story["sch_dep"][1]
    nodes = list(sgraph.nodes.values())
    """
    {'address': 2, 'word': 'crow', 'lemma': 'crow', 'ctag': 'NN', 
    'tag': 'NN', 'feats': '', 'head': 4, 'rel': 'nsubj',
    'deps': defaultdict(<class 'list'>, {'det': [1]})}
    
    """

    print(nodes[3])

    for j, node in enumerate(nodes):
        print("\nNode:", j)
        for k,v in node.items():
            print("{}: {}".format(k, v))

    """ 
    ==>
    {'address': 4, 'word': 'sitting', 'lemma': 'sitting', 'ctag': 'VBG', 
    'tag': 'VBG', 'feats': '', 'head': 0, 'rel': 'root',
    'deps': defaultdict(<class 'list'>, {'nsubj': [2], 'aux': [3], 'nmod': [7]})}
    {
        'address': 4,
        'word': 'sitting',
        'lemma': 'sitting',
        'ctag': 'VBG',
        'tag': 'VBG',
        'feats': '',
        'head': 0,
        'rel': 'root',
        'deps': defaultdict(<class 'list'>,
            {'nsubj': [2],
             'aux': [3],
             'nmod': [7]
             })
    }

    """




    lmtzr = WordNetLemmatizer()
    for node in sgraph.nodes.values():
        tag = node["tag"]
        word = node["word"]
        if word is not None:
            if tag.startswith("V"):
                print(lmtzr.lemmatize(word, 'v'))
            else:
                print(lmtzr.lemmatize(word, 'n'))
    print()
    """
    ==> 
        The
        crow
        sit
        be
        on
        branch
        a
        of
        tree
        a

    """


    answer = find_answer(qgraph, sgraph)
    print("answer:", answer)
    """
    ==>  on a branch of a tree
    """
