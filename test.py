#
#   Question Answering System
#   Nick Warren - ncwarren
#   Andrew McCree - amccree
#   Sri Boinapalli - sboinapa
#   CMPS 143 - Spring 17
#
import csv
import sys, nltk, operator, re
from collections import OrderedDict
from nltk.parse import DependencyGraph
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tree import Tree

SRC = "data/" ## !!!Be sure to change to empty string before turn-in!!!!

##################################################################################################
## Utility Functions ##(copied from read_write_stub.py)###########################################
##################################################################################################

# Read the file from disk
# filename can be fables-01.story, fables-01.sch
def read_file(filename):
	fh = open(SRC+filename, 'r')
	text = fh.read()
	fh.close()
	return text

# Read file from disk by line, better for .par
def read_file_lines(parfile):
	fh = open(SRC+parfile, 'r')
	lines = fh.readlines()
	fh.close()
	return lines

def load_wordnet_ids(filename):
	file = open(filename, 'r')
	if "noun" in filename: type = "noun"
	else: type = "verb"
	csvreader = csv.DictReader(file, delimiter=",", quotechar='"')

	word_ids = nltk.defaultdict()

	#word_ids = defaultdict()

	for line in csvreader:
		word_ids[line['synset_id']] = {'synset_offset': line['synset_offset'], 'story_'+type: line['story_'+type], 'stories': line['stories']}
	return word_ids

#reads in file name and creates dictionary containing data in all file types of fname
def get_data_dict(fname):
	data_dict = {}
	data_types = ["story", "sch", "questions"]
	parser_types = ["par", "dep"]
	for dt in data_types:
		data_dict[dt] = read_file(fname + "." + dt)
		for tp in parser_types:
			data_dict['{}.{}'.format(dt, tp)] = read_file_lines(fname + "." + dt + "." + tp)
	return data_dict

# returns a dictionary where the question numbers are the key
# and its items are another dict of difficulty, question, type, and answer
# e.g. story_dict = {'fables-01-1': {'Difficulty': x, 'Question': y, 'Type':}, 'fables-01-2': {...}, ...}
def getQA(qa_data):
	question_dict = {}
	for m in re.finditer(r"QuestionID:\s*(?P<id>.*)\nQuestion:\s*(?P<ques>.*)\n(Answer:\s*(?P<answ>.*)\n){0,1}Difficulty:\s*(?P<diff>.*)\nType:\s*(?P<type>.*)\n*", qa_data):
		qid = m.group("id")
		question_dict[qid] = {}
		question_dict[qid]['Question'] = m.group("ques")
		question_dict[qid]['Answer'] = m.group("answ")
		question_dict[qid]['Difficulty'] = m.group("diff")
		question_dict[qid]['Type'] = m.group("type")
	return question_dict

# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):
	sentences = nltk.sent_tokenize(text)
	sentences = [nltk.word_tokenize(sent) for sent in sentences]
	sentences = [nltk.pos_tag(sent) for sent in sentences]
	return sentences

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
	nodes = []
	for subtree in tree.subtrees():
		node = matches(pattern, subtree)
		if node is not None:
			nodes.append(node)
	return nodes

#takes a pos_tagger() tag and converts it to use with wordnet lemmatizer
def convert_tag(tag):
	if tag.startswith('J'):
		return wordnet.ADJ
	elif tag.startswith('V'):
		return wordnet.VERB
	elif tag.startswith('N'):
		return wordnet.NOUN
	elif tag.startswith('R'):
		return wordnet.ADV
	else:
		return ''

def get_lemma(tagged_tokens):
	#porter = nltk.PorterStemmer()
	lmtzr = WordNetLemmatizer()
	words = []
	for token, tag in tagged_tokens:
		if(token=="standing" and tag=="NN"): tag = "VBD"
		if(token=="felt"): words.append("feel")
		if(convert_tag(tag) is not ''):
			words.append(lmtzr.lemmatize(token, convert_tag(tag)))
		else:
			words.append(lmtzr.lemmatize(token))
	return set(words)
#return set([lmtzr.lemmatize(token, convert_tag(tag)) for token, tag in tagged_tokens])

def get_bow(tagged_tokens):
	stopwords = set(nltk.corpus.stopwords.words("english"))
	return set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])

def find_pattern(qsentence): #needs work!
	#print(qtree[0])
	#print()

	if(qsentence[0][0] == "Where"):
		return "(PP)"
	elif(qsentence[0][0] == "What"):
		if("happened" in qsentence): return "(VP)"
		return "(NP)"
	elif(qsentence[0][0] == "Who"):
		if("about" in qsentence): return "(NN)"
		return "(NP)"
	elif(qsentence[0][0] == "When"):
		return "(PP)"
	elif(qsentence[0][0] == "Why"):
		return "(NP (*) (PP))"
	elif(qsentence[0][0] == "How"):
		return "(ADVP)"
	else:
		return "(NP)"

# qsentence: the questions string
# source: data dict of all story files (.sch, .par)
def find_best_sentence(qsentence, qtype, source):

	## read parse files and story sentences
	if ("|" in qtype): #need both types
		sentences = get_sentences(source["sch"] + source["story"])
		story_trees = [Tree.fromstring(line) for line in (source["sch.par"] + source["story.par"])]
	else:
		sentences = get_sentences(source[qtype.lower()])
		story_trees = [Tree.fromstring(line) for line in source[qtype.lower()+".par"]]

	#print(story_trees[0])

	sents = list(zip(sentences, story_trees))
	qbow = get_bow(qsentence)
	qlem = get_lemma(qsentence)

	# Collect all the candidate answers
	answers = []
	for sent in sents:
		# A list of all the word tokens in the sentence
		sbow = get_bow(sent[0])
		slem = get_lemma(sent[0])

		## PRINTING SENTENCES AND CORRESPONDING PARSE TREE
		#print(sent)
		#print(story_trees[i])

		# Count the # of overlapping words between the Q and the A
		# & is the set intersection operator
		overlap_count = len(qlem & slem) + len(qbow & sbow)
		#print("bow " + str(qbow & sbow))

		answers.append((overlap_count, sent))

	# Sort the results by the first element of the tuple (i.e., the count)
	# Sort answers from smallest to largest by default, so reverse it
	answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

	# we know this sentence and corresponding tree contains the answer
	best_sentence = (answers[0])[1][0]
	best_tree = answers[0][1][1]
	return best_sentence, best_tree

# def get_wordnet(qsentence):
# 	#TODO: use synsets from noun/verb to find words in story
# 	# use wordnet/wordnet_demo.py
# 	new_sentence = []
# 	word_ids = None
# 	word_synsets = None
# 	word_hypo = None
# 	stopwords = set(nltk.corpus.stopwords.words("english"))
# 	noun_ids = load_wordnet_ids("stub_code/wordnet/Wordnet_nouns.csv")
# 	verb_ids = load_wordnet_ids("stub_code/wordnet/Wordnet_verbs.csv")
#
# 	for i, (word, tag) in enumerate(qsentence):
#
# 		if tag.startswith("N") or tag.startswith("V") and word not in stopwords:
#
# 			if tag.startswith("N"):
# 				word_ids = noun_ids
# 			elif tag.startswith("V"):
# 				word_ids = verb_ids
#
# 			for synset_id, items in word_ids.items():
# 				stories = items['stories']
# 			 	#print(stories)
#
# 			word_synsets = wordnet.synsets(word)
#
# 			for synset in word_synsets:
#
# 				# checking if synonym is in wordnet
# 				if synset.name() in word_ids:
#
# 					qsentence[i] = (synset.name()[0:synset.name().index(".")], tag)
#
#                 # checking for hyponyms
#                 word_hypo = word_synsets.hyponyms()
#
#                 elif synset.name() in word_hypo:
#                 	qsentence[i] = synset.name()[0:synset.name().index(".")], tag)
#
#                 # checking for hypernyms
#                 word_hyper = word_synsets.hypernyms()
#                 else synset.name in word_hyper:
#                 	qsentence[i] = synset.name()[0.synset.name().index(".")], tag)
#
# 	return qsentence
"""
    for synset_id, items in noun_ids.items():
        noun = items['story_noun']
        stories = items['stories']

    for synset_id, items in verb_ids.items():
        verb = items['story_verb']
        stories = items['stories']

    #hyponyms
	word_synsets = wn.synsets(word)
    print("word synsets: %s" % word_synsets)

    print(word+ " hyponyms")
    for word_synset in word_synsets:
        word_hypo = word_synset.hyponyms()
        print("%s: %s" % (word_synset, word, hypo))

        for hypo in word_hypo:
            print(hypo.name()[0:hypo.name().index(".")])

    #hypernyms
	hyp_synsets = wn.synsets(word)

	for hyp_synset in hyp_synsets:
		word_hypernym = hyp_synset.hypernyms()
		print ("%s: %s" % (hyp_synset, word_hypernym))

"""



# specifically for answering "why" questions
def answer_why(sentence):
	#TODO: take in a sentence and return everything after "because", "to", or "for" and don't include punctuation
	#print(sentence)
	a = ""
	for word, tag in sentence:
		if word == "because":
			#print(sentence[sentence.index((word, tag)):])
			return sentence[sentence.index((word, tag)):len(sentence)-1]
	
	for word, tag in sentence:
		if word == "to":
			#print(sentence[sentence.index((word, tag)):])
			return sentence[sentence.index((word, tag)):len(sentence)-1]
	
	return sentence

def get_answer(question, source):
	qsentence = get_sentences(question["Question"])[0]
	qtype = question["Type"]

	#check question difficulty (right now it does baseline no matter what)
	# question["Difficulty"] is "Easy":

	# if question["Difficulty"] == "Hard":
	# 	#print("old q" + str(qsentence))
	# 	qsentence = get_wordnet(qsentence)
	# 	#print("new q" + str(qsentence))

	#return the best sentence and corresponding tree for the best sentence
	best_sentence, best_tree = find_best_sentence(qsentence, qtype, source)

	if qsentence[0][0] == "Why":
		answer = answer_why(best_sentence)
		a = " ".join(t[0] for t in answer)
		#print(best_sentence)
		
	else:
		#find pattern based on question word and run through matcher to get a best subtree
		pattern = nltk.ParentedTree.fromstring(find_pattern(qsentence))
		subtrees = pattern_matcher(pattern, best_tree)

		#if a mathcing subtree was found, use that as best tree
		if(subtrees):
			answers = []
			for t in subtrees:
				tbow = get_bow(" ".join(best_tree.leaves()))
				qbow = get_bow(qsentence)
				overlap_count = len(qbow & tbow)
				answers.append((overlap_count, t))

			answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
			best_tree = answers[0][1]

		a = " ".join(best_tree.leaves())

	bs = " ".join(t[0] for t in best_sentence)
	return bs, a # we only continue to return the best sesntence for debug

# Gets the answer for each question and writes the answer-file
def write_answers(f, questions, source):

	for qid in questions:
		question = questions[qid]

		sent, answer = get_answer(question, source) #HERE call function that passes question and returns an answer string
		#print(answer)
		#a = " ".join(t[0] for t in answer) #construct answer sentece string
		#tr = " ".join(tree.leaves()) #construct answer tree string

		f.write("QuestionID: "+ qid +"\n")
		#f.write("Question: " + question["Question"] + "\n") #**MUST be removed in final!!
		f.write("Answer: "+ answer + "\n\n")
		#f.write("Best Sentence: " + sent + "\n\n") #**MUST be removed in final!!

if __name__ == '__main__':

	#read in filenames from argument
	data_file = "hw7-stories.tsv"#sys.argv[1]
	data = read_file(data_file).split("\n")


	output = open("train_my_answers.txt", 'w')
	for d in data:
		if d is not "":
			print("Processing..." + d)

			data_dict = get_data_dict(d) #dictionary holding all fable files/data
			#print(data_dict["story.par"].split("("))
			question_dict = getQA(data_dict["questions"]) #buld dictionay of all question key-values


			write_answers(output, question_dict, data_dict)

	output.close()

#lmtzr = WordNetLemmatizer()
# porter = nltk.PorterStemmer()
#lancaster = nltk.LancasterStemmer()

# lmtzr = WordNetLemmatizer()
# print(lmtzr.lemmatize("feel", "v"))
# print(lmtzr.lemmatize("felt", "v"))

