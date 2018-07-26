# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 15:51:41 2018

@author: Karthick
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import time
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors


print("The version of the Tensorflow being used")
print('TensorFlow Version: {}'.format(tf.__version__))


####################  Filtering the Data  #####################################

reviews = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\ML\\Text Summarization\\Reviews.csv")
print("The dimension of the dataset...",reviews.shape)
print("Few example data... {}",reviews.head())  

#The number of null values 

reviews.isnull().sum()

# Remove null values and unneeded features
reviews = reviews.dropna()
reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator',
                        'Score','Time'], 1)
reviews = reviews.reset_index(drop=True)

for i in range(5):
    print(reviews.Summary[i],"\n")
    print(reviews.Text[i],"\n")
    
################# Preparing the data ##########################################

contractions = {
                "ain't": "am not",
                "aren't": "are not",
                "can't": "cannot",
                "can't've": "cannot have",
                "'cause": "because",
                "could've": "could have",
                "couldn't": "could not",
                "couldn't've": "could not have",
                "didn't": "did not",
                "doesn't": "does not",
                "don't": "do not",
                "hadn't": "had not",
                "hadn't've": "had not have",
                "hasn't": "has not",
                "haven't": "have not",
                "he'd": "he would",
                "he'd've": "he would have",
                "he'll": "he will",
                "he's": "he is",
                "how'd": "how did",
                "how'll": "how will",
                "how's": "how is",
                "i'd": "i would",
                "i'll": "i will",
                "i'm": "i am",
                "i've": "i have",
                "isn't": "is not",
                "it'd": "it would",
                "it'll": "it will",
                "it's": "it is",
                "let's": "let us",
                "ma'am": "madam",
                "mayn't": "may not",
                "might've": "might have",
                "mightn't": "might not",
                "must've": "must have",
                "mustn't": "must not",
                "needn't": "need not",
                "oughtn't": "ought not",
                "shan't": "shall not",
                "sha'n't": "shall not",
                "she'd": "she would",
                "she'll": "she will",
                "she's": "she is",
                "should've": "should have",
                "shouldn't": "should not",
                "that'd": "that would",
                "that's": "that is",
                "there'd": "there had",
                "there's": "there is",
                "they'd": "they would",
                "they'll": "they will",
                "they're": "they are",
                "they've": "they have",
                "wasn't": "was not",
                "we'd": "we would",
                "we'll": "we will",
                "we're": "we are",
                "we've": "we have",
                "weren't": "were not",
                "what'll": "what will",
                "what're": "what are",
                "what's": "what is",
                "what've": "what have",
                "where'd": "where did",
                "where's": "where is",
                "who'll": "who will",
                "who's": "who is",
                "won't": "will not",
                "wouldn't": "would not",
                "you'd": "you would",
                "you'll": "you will",
                "you're": "you are"
        }

# function to remove 1.Remove contractions with longer forms
#                    2.Finding Regex patterns and replacing them
#                    3.Removing Stop words
def cleanText(text,removeStopwords = True):
    text = text.lower() # All lower charecters
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
        
    # Format words and remove unwanted characters using Re
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    #Optionally, remove stop words
    if removeStopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
    
    return text

#we use a small part of the data for training 35173 rows
# Clean the summaries and texts
cleanSummaries = []
for summary in reviews.Summary:
    cleanSummaries.append(cleanText(summary, removeStopwords=False))
print("Summaries are complete.")

cleanTexts = []
for text in reviews.Text:
    cleanTexts.append(cleanText(text))
print("Texts are complete.")

def countWords(countDict,text):
    #counting the number of occurances or each word in the text
    for sentences in text:
        for word in sentences.split():
            if word not in countDict:
                countDict[word] = 1
            else:
                countDict[word] += 1

#################### Building the vocabulary ##################################
                
#building the vocabulary
wordCounts ={}
countWords(wordCounts, cleanSummaries)
countWords(wordCounts, cleanTexts)
print("Size of the Vocabulary: ", len(wordCounts))

# Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe
#https://github.com/commonsense/conceptnet-numberbatch

embeddingIndex = {}
with open('C:\\Users\\Lenovo\\Desktop\\ML\\Text Summarization\\numberbatch-en-17.02.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddingIndex[word] = embedding

print('Word embeddings:', len(embeddingIndex))

#Number of words that are not on the Conceptnet Numberbatch and are used more than a certain threshold
missingWords = 0
threshold = 20 #through trail and error ., reducing the threshold will make the model train more words

for word, count in wordCounts.items():
    if count > threshold:
        if word not in embeddingIndex:
            missingWords += 1
    
missingRatio = round(missingWords/len(wordCounts),4)*100

print("Number of words missing from CN:", missingWords)
print("Percent of words that are missing from vocabulary: {}%".format(missingRatio))

########################## Vocabulary to Intergers ############################

#every word will have a unique number
#One hot encoding the words
vocabToInt = {}

value = 0

for word, count in wordCounts.items():
    if count>= threshold or word in embeddingIndex:
        vocabToInt[word] = value
        value += 1
    
#Special tokens to be used for padding the sentences
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]
#UNK for unknown words
#PAD for spaces to make it of equal length
#EOS for End of Sentence
#GO for start of the sentence

#Add codes to Vocab
for code in codes:
    vocabToInt[code] = len(vocabToInt)
    
vocabToInt['<UNK>']
#Building a reverse dictionary for conversion of integers to words

intToVocab = {}

for word , count in vocabToInt.items():
    intToVocab[count] = word
    
usageRatio = round(len(vocabToInt) / len(wordCounts),4)*100

print("Total number of unique words:", len(wordCounts))
print("Number of words we will use:", len(vocabToInt))
print("Percent of words we will use: {}%".format(usageRatio))


################# Embedding Matrix #############################################

embeddingDim = 300 # to match the CN embedding vectors
nbWords = len(vocabToInt) #Total number of words for training

#creating matrix with default values of Zero

wordEmbeddingMatrix = np.zeros ((nbWords,embeddingDim),dtype = np.float32)
for word,i in vocabToInt.items():
    if word in embeddingIndex:
        wordEmbeddingMatrix[i] = embeddingIndex[word]
    else:
        #if word not in CN, then create a random wordEmbedding for it
        newEmbedding = np.array(np.random.uniform(-1.0,1.0,embeddingDim))
        wordEmbeddingMatrix[i] = newEmbedding

print(len(wordEmbeddingMatrix))

#Converting words in the text to an integer
#if the word is not in the vocab to int , use UNK's integer
#total number of words and UNK's
# Add EOS taken to the end of the texts

def convertToInt(text,wordCount,unkCount,eos = False):
    ints = []
    for sentence in text:
        sentenceInts = []
        for word in sentence.split():
            wordCount += 1
            if word in vocabToInt:
                sentenceInts.append(vocabToInt[word])
            else:
                sentenceInts.append(vocabToInt["<UNK>"])
                unkCount += 1
        if eos:
            sentenceInts.append(vocabToInt["<EOS>"])
        ints.append(sentenceInts)
    return ints, wordCount, unkCount

#Applying convertToInts to cleanSummaries and cleanTexts

wordCount = 0
unkCount = 0

intSummaries ,wordCount, unkCount = convertToInt(cleanSummaries,wordCount, unkCount)
intTexts, wordCount, unkCount = convertToInt(cleanTexts,wordCount,unkCount, eos= True)

unkPercent = round(unkCount/wordCount,4)*100

print("Total number of words in headlines:", wordCount)
print("Total number of UNKs in headlines:", unkCount)
print("Percent of words that are UNK: {}%".format(unkPercent))

################# Setting up lengths ##########################################

def createLengths(text):
    lengths =[]
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])

lengthSummaries = createLengths(intSummaries)
lengthTexts = createLengths(intTexts)
    
def unkCounter(sentence):
    '''Counts the number of time UNK appears in a sentence.'''
    unkCount = 0
    for word in sentence:
        if word == vocabToInt["<UNK>"]:
            unkCount += 1
    return unkCount

# sorting the summaries and texts up the lengths of the texts, shortest to Longest
#Limiting the lenth of summaries and text based on the min and max ranges
#remove the reviews that include too many UNK values

sortedSummaries = []
sortedTexts = []
maxTextLengths = 85 #based on the describe option
maxSummaryLengths = 15
minLength = 3
unkTextLimit = 100 # use any threshold ., more dataset use less UNK text limit
unkSummaryLimit = 100 # use any threshold ., more dataset use less UNK text limit

for length in range(min(lengthTexts.counts), maxTextLengths): 
    for count, words in enumerate(intSummaries):
        if (len(intSummaries[count]) >= minLength and
            len(intSummaries[count]) <= maxSummaryLengths and
            len(intTexts[count]) >= minLength and
            unkCounter(intSummaries[count]) <= unkSummaryLimit and
            unkCounter(intTexts[count]) <= unkTextLimit and
            length == len(intTexts[count])
           ):
            sortedSummaries.append(intSummaries[count])
            sortedTexts.append(intTexts[count])
        
# Compare lengths to ensure they match
print(len(sortedSummaries))
print(len(sortedTexts))




        