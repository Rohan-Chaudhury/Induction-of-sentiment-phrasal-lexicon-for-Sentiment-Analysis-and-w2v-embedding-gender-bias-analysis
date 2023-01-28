import sys
import getopt
import os
import math
import collections
import operator
import numpy as np
import re
class NaiveBayes:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.BOOLEAN_NB = False
    # self.stopList = set(self.readFile('../data/english.stop'))
    self.numFolds = 10
    self.phrases_excellent={}
    self.phrases_poor={}
    self.distance=10
    self.pos_pattern=["^(JJ)__((NN)|(NNS))__(.)+",
                 "^((RB)|(RBR)|(RBS))__(JJ)__(?!(NN)|(NNS))",
                 "^(JJ)__(JJ)__(?!(NN)|(NNS))",
                 "^((NN)|(NNS))__(JJ)__(?!(NN)|(NNS))",
                 "^((RB)|(RBR)|(RBS))__((VB)|(VBD)|(VBN)|(VBG))__(.)+"]
    self.excellent_count=0.01
    self.poor_count=0.01

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts.
  #
  #
  # If any one of the FILTER_STOP_WORDS and BOOLEAN_NB flags is on, the
  # other one is meant to be off.

    self.total_pos_doc=0
    self.total_neg_doc=0
    self.all_words=[]
    # self.len_unique_words=0
    self.total_docs=0
    self.total_words=0
    self.word_dictionary={}
    self.total_word_count={'pos':0,'neg':0}

  def classify(self, words):
    # """ TODO
    #   'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    # """
    polarity_score=0
    words_split=[]
    for i in range(len(words)):
        words_split.append(words[i].split('_'))
    for i in range(len(words_split)-2):
        pos_tag = words_split[i][1]+ "__"+words_split[i+1][1] +"__"+ words_split[i+2][1] 
        for pattern in self.pos_pattern:
            if re.match(pattern,pos_tag):
                phrase=words_split[i][0]+"__"+words_split[i+1][0]
                polarity_score+=self.calculate_semantic_orientation(phrase)
                break
    # print(polarity_score)
    if polarity_score>0:
      return 'pos'
    else:
      return 'neg'

  def calculate_semantic_orientation(self, phrase):
    so=0
    if phrase in self.phrases_excellent:
      so+=np.log2((self.phrases_excellent[phrase]*self.poor_count))
      so-=np.log2((self.phrases_poor[phrase]*self.excellent_count))
    return so

  def NEAR_operator(self, index, key, words_split, pos_words, neg_words):
    for j in range(index-self.distance, index+self.distance+1):
      for pos in pos_words:
        if j >= 0 and j < len(words_split) and words_split[j][0].lower() == pos:
            self.phrases_excellent[key]+=1
      for neg in neg_words:
        if j >= 0 and j < len(words_split) and words_split[j][0].lower() == neg:
            self.phrases_poor[key]+=1


  def find_semantic_phrases(self,words,pos_words,neg_words):

    words_split=[]
    for i in range(len(words)):
        word_split=words[i].split('_')
        words_split.append(word_split)
        for pos in pos_words:
            if word_split[0].lower()==pos.lower():
              self.excellent_count+=1
        for neg in neg_words:
            if word_split[0].lower()==neg.lower():
              self.poor_count+=1

    
    for i in range(len(words_split)-2):
        pos_tag = words_split[i][1]+ "__"+words_split[i+1][1] +"__"+ words_split[i+2][1] 
        for pattern in self.pos_pattern:
            if re.match(pattern,pos_tag):
                key=words_split[i][0]+"__"+words_split[i+1][0]
                # print (key.replace("__", " ")," of type ", pos_tag.replace("__", " "))
                if key not in self.phrases_excellent:
                  self.phrases_excellent[key]=0.01
                  self.phrases_poor[key]=0.01
                self.NEAR_operator(i, key, words_split, pos_words, neg_words)
                break



  def addExample(self, klass, words):

    pos_words=["excellent","best","positive"]
    neg_words=["poor","bad","negative"]
    self.find_semantic_phrases(words, pos_words, neg_words)

    
  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)


  def crossValidationSplits(self, trainDir):
    """Returns a list of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB):
  nb = NaiveBayes()
  splits = nb.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    accuracy = 0.0
    for example in split.train:
      words = example.words
      classifier.addExample(example.klass, words)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print('[INFO]\tAccuracy: %f' % avgAccuracy)
    
    
def classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB, trainDir, testDir):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  classifier.BOOLEAN_NB = BOOLEAN_NB
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit)
  testSplit = classifier.trainSplit(testDir)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print('[INFO]\tAccuracy: %f' % accuracy)


def main():
  FILTER_STOP_WORDS = False
  BOOLEAN_NB = False
  (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
  if ('-f','') in options:
    FILTER_STOP_WORDS = True
  elif ('-b','') in options:
    BOOLEAN_NB = True
  
  if len(args) == 2:
    classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB,  args[0], args[1])
  elif len(args) == 1:
    test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB)

if __name__ == "__main__":
    main()
