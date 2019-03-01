import re
import nltk, operator
from nltk import WordNetLemmatizer

#comment

from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers

from nltk.corpus import wordnet as wn
from word2vec_extractor import Word2vecExtractor


STOPWORDS = set(nltk.corpus.stopwords.words("english"))
glove_w2v_file = "data/glove-w2v.txt"
W2vecextractor = Word2vecExtractor(glove_w2v_file)

GRAMMAR = """
                N: {<PRP>|<NN.*>}
                V: {<V.*>}
                ADJ: {<JJ.*>}
                NP: {<DT>? <ADJ>* <N>}
                PP: {<IN> <NP>}
                VP: {<TO>? <V> (<NP>|<PP>)*}
            """


# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):

    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences


def get_bow(tagged_tokens, stopwords):

    return set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])


# qtokens: is a list of pos tagged question tokens with SW removed
# sentences: is a list of pos tagged story sentences
# stopwords is a set of stopwords
def baseline_best(qbow, sentences, stopwords):
    # Collect all the candidate answers
    answers = []
    for sent in sentences:
        # A list of all the word tokens in the sentence
        sbow = get_bow(sent[0], stopwords)

        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        intersect = qbow & sbow
        overlap = len(intersect)

        answers.append((overlap, sent))

    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer
    best_sentence = answers[0][1][0]
    best_tree = answers[0][1][1]

    return best_sentence, best_tree


def baseline_answer(qbow, subtrees, best_tree, stopwords):
    # Collect all the candidate answers
    answers = []
    for tree in subtrees:
        # A list of all the word tokens in the sentence
        tbow = get_bow(" ".join(best_tree.leaves()), stopwords)

        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        intersect = qbow & tbow
        overlap = len(intersect)

        answers.append((overlap, tree))

    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the answer
    answer = " ".join(" ".join(tup[1].leaves())for tup in answers)

    # print(answer)

    return answer


# Given a list of sentences, returns the sentence that appears the most if there is one
def find_best_sentence(stext, qtype, qtext, story):

    if "|" in qtype:
        qtype = "sch"

    story_sents = get_sentences(stext)

    # print(str(qtype.lower()) + "_par")

    story_trees = []
    for tree in story[qtype + "_par"]:
        story_trees.append(tree)

    question_sents = get_sentences(qtext)[0]

    qbow = get_bow(question_sents, STOPWORDS)
    sentences = list(zip(story_sents, story_trees))
    best_sentence, best_tree = baseline_best(qbow, sentences, STOPWORDS)

    return best_sentence, best_tree


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):

    if is_adjective(tag):
        return wn.ADJ

    elif is_noun(tag):
        return wn.NOUN

    elif is_adverb(tag):
        return wn.ADV

    elif is_verb(tag):
        return wn.VERB

    return 'n'


def find_pattern(qtext, stext):

    qtype = qtext.split(" ")[0].lower()

    # print(qtext)
    # print("\n\n")
    # print(stext)
    # print("\n\n")

    if "where" in qtype:
        return "(PP)"

    elif "when" in qtype:
        return "(PP)"

    elif "what" in qtype:
        return "(PP)"

    elif "who" in qtype:
        return "(NP)"

    elif "why" in qtype:
        return "(SBAR)"

    else:
        return "(NP)"


# See if our pattern matches the current root of the tree
def matches(pattern, root):
    # Base cases to exit our recursion
    # If both nodes are null we've matched everything so far
    if root is None and pattern is None:
        return root

    # We've matched everything in the pattern we're supposed to (we can ignore the extra
    # nodes in the main tree for now)
    elif pattern is None:
        return root

    # We still have something in our pattern, but there's nothing to match in the tree
    elif root is None:
        return None

    # A node in a tree can either be a string (if it is a leaf) or node
    plabel = pattern if isinstance(pattern, str) else pattern.label()
    rlabel = root if isinstance(root, str) else root.label()

    # If our pattern label is the * then match no matter what
    if plabel == "*":
        return root
    # Otherwise they labels need to match
    elif plabel == rlabel:
        # If there is a match we need to check that all the children match
        # Minor bug (what happens if the pattern has more children than the tree)
        for pchild, rchild in zip(pattern, root):
            match = matches(pchild, rchild)
            if match is None:
                return None
        return root

    return None


def pattern_matcher(pattern, tree):
    for subtree in tree.subtrees():
        node = matches(pattern, subtree)
        if node is not None:
            return node
    return None


def find_phrase(qbow, sent):
    tokens = nltk.word_tokenize(sent)
    # Travel from the end to begin.
    for i in range(len(tokens) - 1, 0, -1):
        word = tokens[i]
        # If find a word that match the question,
        # return the phrase that behind that word.
        # For example, "lion" occur in the question,
        # So we will return "want to eat the bull" which originally might look like this "... the lion want to eat the bull"
        if word in qbow:
            return " ".join(tokens[i+1:])


def get_story_text(qtype, story):

    if qtype == "Sch":
        stext = story["sch"]

    else:
        stext = story["text"]

    return stext


def get_question_text(question):

    qtext = question["text"]

    return qtext


def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: str
    question is a dictionary with keys:
        dep -- A list of dependency graphs for the question sentence.
        par -- A list of constituency parses for the question sentence.
        text -- The raw text of story.
        sid --  The story id.
        difficulty -- easy, medium, or hard
        type -- whether you need to use the 'sch' or 'story' versions
                of the .
        qid  --  The id of the question.
    story is a dictionary with keys:
        story_dep -- list of dependency graphs for each sentence of
                    the story version.
        sch_dep -- list of dependency graphs for each sentence of
                    the sch version.
        sch_par -- list of constituency parses for each sentence of
                    the sch version.
        story_par -- list of constituency parses for each sentence of
                    the story version.
        sch --  the raw text for the sch version.
        text -- the raw text for the story version.
        sid --  the story id
    """
    ###     Your Code Goes Here         ###
    answer = None
    qtype = question["type"].lower()
    qtext = get_question_text(question)
    stext = get_story_text(qtype, story)

    best_sentence, best_tree = find_best_sentence(stext, qtype, qtext, story)

    pattern = nltk.ParentedTree.fromstring(find_pattern(qtext, stext))

    subtrees = pattern_matcher(pattern, best_tree)

    if subtrees is not None:
        qbow = get_bow(get_sentences(qtext)[0], STOPWORDS)
        answer = baseline_answer(qbow, subtrees, best_tree, STOPWORDS)

    answer_sentence = " ".join(t[0] for t in best_sentence)

    # print(answer)
    # print("\n\n")

    # if answer is not None:
    #     return answer
    # else:
    #     return answer_sentence

    return answer_sentence
    # return answer
    # return answer_sentence


#############################################################
###     Dont change the code in this section
#############################################################
class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa(evaluate=False):
    QA = QAEngine(evaluate=evaluate)
    QA.run()
    QA.save_answers()

#############################################################


def main():
    # set evaluate to True/False depending on whether or
    # not you want to run your system on the evaluation
    # data. Evaluation data predictions will be saved
    # to hw6-eval-responses.tsv in the working directory.
    run_qa(evaluate=False)
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()


if __name__ == "__main__":
    main()