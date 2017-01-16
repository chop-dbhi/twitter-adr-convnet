# Copyright 2016 The Children's Hospital of Philadelphia. All Rights Reserved.
# Created by Dr. Aaron J. Masino
# May 6, 2016

import re
import itertools
import numpy as np
from word2vec.word2vecReader import Word2Vec

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    modified to account for repeated punctuation
    """
    string = re.sub(r"\n"," ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",+", " , ", string)
    string = re.sub(r"!+", " ! ", string)
    string = re.sub(r"\(+", " \( ", string)
    string = re.sub(r"\)+", " \) ", string)
    string = re.sub(r"\?+", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_raw_tweet_text(path):
    tweets = list(open(path, 'r').readlines())
    tweets = [tweet.strip().replace("\n","").replace("\"","\"\"") for tweet in tweets]
    return tweets

def clean_strip_split(tweets):
    tweets = [tweet.strip() for tweet in tweets]
    tweets = [clean_str(tweet) for tweet in tweets]
    #split into word lists
    return [tweet.split(" ") for tweet in tweets]

def load_tweets(path):
    '''
    :param path: file containing one tweet per line
    :return: tweets as a list of word lists
    '''
    tweets = list(open(path, 'r').readlines())
    return clean_strip_split(tweets)

def pad_tweets(tweets, pad_word = '<PAD/>'):
    '''
    :param tweets: list of word lists
    :param pad_word: string to use as padding token
    :return: equal length word lists - length is max of tweet lengths
    '''
    max_length = max(len(tweet) for tweet in tweets)
    padded_tweets = []
    for tweet in tweets:
        pad_count = max_length - len(tweet)
        padded_tweet = tweet + [pad_word] * pad_count
        padded_tweets.append(padded_tweet)
    return padded_tweets, max_length

def build_vocab(tweets):
    ''' use tweets to construct a comprehensive vocabularly
    :param tweets: list of word lists
    :return: vocabulary - a mapping from word to index
             unique_tokens - mapping from index to word
    '''
    # mapping from index to word
    unique_tokens = list(set(itertools.chain(*tweets)))
    # mapping from word to index
    vocabulary = {w: i for i, w in enumerate(unique_tokens)}
    return [vocabulary, unique_tokens]

def build_tweet_indices(tweets, vocabularly):
    '''
    converts tweets from a word list to a numpy array of ints where each int is the index of the word in the vocab
    this can be used to access the word embedding from a word embedding matrix
    :param tweets: list of word lists
    :param vocabularly: map from words to indices
    :return: numpy array of ints of shape len(tweets) X tweet length (assumed equal for all tweets)
    '''
    return np.array([[vocabularly[word] for word in tweet] for tweet in tweets])

def tweetsToIndices(tweets, tweet_length, vocab):
    indexedTweets = np.ones((len(tweets),tweet_length), dtype=np.int)
    missing_word_index = len(vocab)
    for row, tweet in enumerate(tweets):
        for col, w in enumerate(tweet):
            indexedTweets[row,col] = vocab.get(w,missing_word_index)
    return indexedTweets

def extractWord2VecEmbeddings(embedding_file, vocab_inv):
    model = Word2Vec.load_word2vec_format(embedding_file, binary=True)
    vocab_length = len(vocab_inv)
    embedding_length = model.size
    embedding_matrix = np.zeros((vocab_length, embedding_length))
    for idx, word in enumerate(vocab_inv):
        if word in model:
            embedding_matrix[idx] = model[word]
    return embedding_matrix, [vocab_length, embedding_length]

def extractBy(condition, data, tol = 1e-6):
    not_condition = condition[:]==False
    return (data[condition], data[not_condition])

def partion(condition, ratios=[.6,.2,.2]):
    ''' returns two lists (l1,l2). l1 is a list of numpy arrays where each array contains indices
    into the data where the condition is True and l2 is a list of numpy arrays where each array contains
    indicies into the data where the condition is False. The len(l1)=len(l2)=len(ratios) and
    the lists in l1 and l2 have lengths determined by the ratio values.'''
    pos = np.where(condition)[0]
    neg = np.where(condition[:]==False)[0]

    #SHOULD ALSO USE np.where(condition) to split data
    #NEED TO MODIFY TO RETURN MASKS ONLY
    #MASK SHOULD BE AN 1D NUMPY ARRAY
    #if not (np.sum(ratios) == 1 or np.sum(ratios) == 1.0): raise Exception('Ratios must sum to 1, got {0}'.format(np.sum(ratios)))
    #(pos, neg) = extractBy(condition, data)
    pos_row_count = pos.shape[0]
    neg_row_count = neg.shape[0]
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    pdata = []
    ndata = []

    for i in range(len(ratios)):
        r = ratios[i]
        if i==len(ratios)-1:
            s2 = pos_row_count
            s4 = neg_row_count
        else:
            s2 = min(s1 + int(round(r*pos_row_count)), pos_row_count)
            s4 = min(s3 + int(round(r*neg_row_count)), neg_row_count)
        if s2<=s1: raise Exception('Insufficient positive data for partition, s1={0}, s2={1}'.format(s1,s2))
        if s4<=s3: raise Exception('Insufficient negative data for partition, s3={0}, s4={1}'.format(s3,s4))
        pdata.append(pos[s1:s2])
        ndata.append(neg[s3:s4])
        s1 = s2
        s3 = s4
    return(pdata,ndata)

