# Copyright (C) Mehrdad Zakershahrak, 2017

# Make sure that you have gensim, logging and numpy installed
import gensim, logging
import numpy as np
import time
import csv
# I print the time at the beginning and at the end so we can see how long the code takes
print time.strftime('%X %x %Z')


def read_all_lines(file_object):
    data = file_object.read()
    items = data.split("\n")
    return items[:-1]


def extract_abstract_and_title(piece):
    if piece[:2] == '#*' or piece[:2] == '#!':
        return [piece[2:]]
    return []


def filter_data(sentences):
    # The english common words that should not effect my citation count
    stoplist = set(
        ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as",
         "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot",
         "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each",
         "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd",
         "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
         "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
         "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
         "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
         "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
         "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
         "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
         "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
         "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
         "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"])

    documents = []
    for x in sentences:
        for s in x:
            documents.append(s)
    # All of the words should be lower case and if the words happen in the same sentence they
    # will get more chance to happen again together (This is a very naive way of explaining 
    # document embedding)
    texts = [[word for word in x.lower().split() if word not in stoplist] for x in documents]

    # Remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]

    tokens = []
    # Get rid of all of empty lists in tokens
    for text in texts:
        if text:
            tokens.append(text)
    return tokens

# The good old parsing
def extract_records(lines,author_citation, pub_citation):
    records = []
    title = ''
    authors = ''
    year = ''
    pubvenue = ''
    citations = ''
    index = ''
    arnetid = ''
    abstract = ''
    for l in lines:
        if l[:2] == "#*":
            title = l[2:]
        elif l[:2] == "#@":
            authors = l[2:]
        elif l[:5] == "#year":
            year = l[5:]
        elif l[:5] == "#conf":
            pubvenue = l[5:]
        elif l[:9] == "#citation":
            citations = l[9:]
        elif l[:6] == "#index":
            index = l[6:]
        elif l[:8] == "#arnetid":
            arnetid = l[8:]
        elif l[:2] == "#!":
            abstract = l[2:]
        elif l == '':
            # Creating a dictionary (hashmap) named record to keep on record
            if int(citations) < 0:
                citations = '0'
            record = {'title': title, 'auhtors': authors, 'year': year, 'pubvenue': pubvenue,
                      'citations': citations,'index': index, 'arnetid': arnetid, 'abstract': abstract}
            records.append(record)
            names = authors.split(',')
            for name in names:
                if name in author_citation:
                    author_citation[name] += int(citations)
                else:
                    author_citation[name] = int(citations)

            if pubvenue in pub_citation:
                pub_citation[pubvenue] += int(citations)
            else:
                pub_citation[pubvenue] = int(citations)
    return records

# This is the hard part! I'm joking
def aggregate_word_features(sentence, model_wv, vocabulary):
    i = 0
    result = 0
    for word in sentence.split():
        word = word.lower()
        if word in vocabulary:
            feature_value = model_wv[word]
            if i == 0:
                result = feature_value
            else:
                result = result + feature_value
            i = i + 1
    return result

# The main code starts from here

# make sure the output is placed in the same folder as this code
f = open('test.txt')
lines = read_all_lines(f)
f.close()

sentences = []
for line in lines:
    x = extract_abstract_and_title(line)
    if x != []:
        sentences.append(x)

tokens = filter_data(sentences)

# This is Mehrdad's Magic (Deep learning for document embedding)
model = gensim.models.Word2Vec(tokens, size=32, min_count=10)

# creates a hashmap of all the important vocabulary
vocabulary = set(model.wv.vocab)

print 'len(vocabulary) = ', len(vocabulary)
author_citation = {}
pub_citation = {}
records = extract_records(lines, author_citation, pub_citation)
print 'len(records) = ', len(records)

count = 0
for r in records:
    # this count is to show you how fast the records are getting updated
    if count % 50000 == 0:
        print count
    r['title_fv'] = aggregate_word_features(r['title'], model.wv, vocabulary)
    r['abstract_fv'] = aggregate_word_features(r['abstract'], model.wv, vocabulary)
    count = count + 1
# here you have the records updated (our feature matrix is ready)

print time.strftime('%X %x %Z')
print('updating author and pubvenue')

for rec in records:
    cite_auth = 0
    cite_venue = 0
    names = rec['auhtors'].split(',')
    for name in names:
        cite_auth += author_citation[name]
    rec['author_citation'] = cite_auth

    if rec['pubvenue']:
        cite_venue = pub_citation[rec['pubvenue']]
    rec['pub_citation'] = cite_venue



print time.strftime('%X %x %Z')
print('writing data')


with open('output.csv', 'wb') as f:
    f.write('index, author_citation, citations, year, pub_citation, title_fv,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, abstract_fv')
    j = 0
    for rec in records:
        if j % 50000 == 0:
            print j
        j = j + 1
        f.write('\n' + rec['index'] + ',' + str(rec['author_citation']) + ',' + rec['citations'] + ',' + rec['year'] + ',' + str(rec['pub_citation']))
        if type(rec['title_fv']) is not int:
            for x in rec['title_fv']:
                f.write(',' + str(x))
        else:
            f.write(',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
        if type(rec['abstract_fv']) is not int:
            for x in rec['abstract_fv']:
                f.write(',' + str(x))
        else:
            f.write(',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
    f.close()

print time.strftime('%X %x %Z')


