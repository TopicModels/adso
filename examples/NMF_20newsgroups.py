"""
Analyze the 20newsgroups dataset with NMF
=========================================
"""

# %%
# import
import adso

import matplotlib.pyplot as plt

import nltk

import numpy as np

# %%
# set seed
adso.set_seed(1234)

# %%
# Download the dataset
data = adso.data.load_20newsgroups(split="test")

print("Number of documents: ", len(data))

# %%
# Tokenize the dataset using a stemmer and a stopwords list, removing punctation

adso.transform.nltk_download("stopwords")

snowball = nltk.stem.snowball.SnowballStemmer("english")


def stemmer(word):
    ret = snowball.stem(word)
    if ret.isalpha():
        return ret
    else:
        return None


tokenizer = adso.transform.Tokenizer(
    stemmer=stemmer,
    stopwords=nltk.corpus.stopwords.words("english") + [None],
)

tokens = tokenizer.fit_transform(data)

print("First ten tokens of the first document:")
print(tokens[0][:10])

# %%
# Transform the list of tokens in a list of numbers.
# We will use the frequency and the TFIDF frequency (a correction
# for the distribution among the documents).

freq = adso.transform.FreqVectorizer(max_freq=0.75, max_size=10000)

tfidf = adso.transform.TFIDFVectorizer(max_freq=0.75, max_size=10000)

# %%
# Generate the vocabulary and share it between the vectorizer.

freq.fit(tokens)

# I will write an ad hoc function later
vocab = freq.vocab

print("Number of words in vocabulary: ", len(vocab))

tfidf.vocab = vocab

print("index of word 'god': ", vocab["god"])
print("word at index 32: ", vocab[32])

# %%
# Create the frequency matrices from tokens.
freq_matrix = freq.transform(tokens)
tfidf_matrix = tfidf.transform(tokens)

# %%
# NMF1 using frequency matrix and ACLS algorithm
NMF1 = adso.topicmodel.NMF(
    n_topic=20, max_iter=100, tolerance=1e-3, lambdaH=0.001, lambdaW=0.001
)
W1, H1, iter1 = NMF1.fit_transform(freq_matrix)
print("NMF1 ended after", iter1, "iterations")

# %%
# NMF2 using frequency matrix and AHCLS algorithm
NMF2 = adso.topicmodel.NMF(
    n_topic=20,
    max_iter=100,
    tolerance=1e-3,
    lambdaH=0.001,
    lambdaW=0.001,
    alphaH=0.01,
    alphaW=0.01,
    method="AHCLS",
)
W2, H2, iter2 = NMF2.fit_transform(freq_matrix)
print("NMF2 ended after", iter2, "iterations")

# %%
# NMF3 using tfidf matrix and ALS algorithm
NMF3 = adso.topicmodel.NMF(n_topic=20, max_iter=100, tolerance=1e-3, method="ALS")
W3, H3, iter3 = NMF3.fit_transform(tfidf_matrix)
print("NMF3 ended after", iter3, "iterations")

# %%
# Check the 10 most characteristic words for the first topic of each model

print("10 most characteristic words for the first topic of NMF1")
print(
    list(
        map(
            lambda i: vocab[i],
            np.argsort(np.squeeze(-H1[0, :].toarray()))[:10].tolist(),
        )
    )
)
print("10 most characteristic words for the first topic of NMF2")
print(
    list(
        map(
            lambda i: vocab[i],
            np.argsort(np.squeeze(-H2[0, :].toarray()))[:10].tolist(),
        )
    )
)
print("10 most characteristic words for the first topic of NMF3")
print(
    list(
        map(
            lambda i: vocab[i],
            np.argsort(np.squeeze(-H3[0, :].toarray()))[:10].tolist(),
        )
    )
)
# %%
# Print the confusion matrix (not diagonalized) for NMF1
predicted_topic = np.argmax(W1, axis=1)

listvectorizer = adso.transform.ListVectorizer()
labels = list(map(lambda l: [l], data.get_labels()))

label_topic = np.squeeze(listvectorizer.fit_transform(labels))

confusion = np.zeros((20, 20))
for i in zip(label_topic, predicted_topic):
    confusion[i] += 1

fig, ax = plt.subplots()
ax.imshow(confusion)
ax.set_xticks(np.arange(20))
ax.set_yticks(np.arange(20))
ax.set_yticklabels(list(listvectorizer.vocab.stoi.keys()))

# %%
# Print the confusion matrix (not diagonalized) for NMF2
predicted_topic = np.argmax(W2, axis=1)

listvectorizer = adso.transform.ListVectorizer()
labels = list(map(lambda l: [l], data.get_labels()))

label_topic = np.squeeze(listvectorizer.fit_transform(labels))

confusion = np.zeros((20, 20))
for i in zip(label_topic, predicted_topic):
    confusion[i] += 1

fig, ax = plt.subplots()
ax.imshow(confusion)
ax.set_xticks(np.arange(20))
ax.set_yticks(np.arange(20))
ax.set_yticklabels(list(listvectorizer.vocab.stoi.keys()))

# %%
# Print the confusion matrix (not diagonalized) for NMF3
predicted_topic = np.argmax(W3, axis=1)

listvectorizer = adso.transform.ListVectorizer()
labels = list(map(lambda l: [l], data.get_labels()))

label_topic = np.squeeze(listvectorizer.fit_transform(labels))

confusion = np.zeros((20, 20))
for i in zip(label_topic, predicted_topic):
    confusion[i] += 1

fig, ax = plt.subplots()
ax.imshow(confusion)
ax.set_xticks(np.arange(20))
ax.set_yticks(np.arange(20))
ax.set_yticklabels(list(listvectorizer.vocab.stoi.keys()))
