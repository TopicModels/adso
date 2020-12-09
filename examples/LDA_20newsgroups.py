"""
Analyze the 20newsgroups dataset with LDA
=========================================
"""

# %%
# import
import random

import adso
import matplotlib.pyplot as plt
import nltk
import numpy as np

# %%
# set seed
adso.set_seed(1234)

# %%
# Download the dataset and select 1000 random elements
data = adso.data.load_20newsgroups(split="test")

new_data = []
for i in random.sample(range(len(data)), k=1000):
    new_data.append(data[i])
data = adso.data.LabelledDataset(new_data)

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
# We will use the absolute frequency.

vectorizer = adso.transform.CountVectorizer(max_freq=0.7, min_freq=0.1, max_size=1000)


# %%
# Generate the vocabulary.

vectorizer.fit(tokens)
vocab = vectorizer.vocab

print("Number of words in vocabulary: ", len(vocab))

print("index of word 'god': ", vocab["god"])
print("word at index 52: ", vocab[52])

# %%
# Create the count matrices from tokens.
count_matrix = vectorizer.transform(tokens)

# %%
# LDA
LDA = adso.topicmodel.LDA(n_topic=20, tolerance=0.001, max_iter=100)
ret = LDA.fit_transform(count_matrix)
estimation = ret[0]
beta = ret[2]
print("LDA ended after", ret[6], "iterations, achiving a loglikelihood of", ret[5])

# %%
# Check the 10 most characteristic words for each topic

for i in range(20):
    print("10 most characteristic words of topic", i)
    print(
        list(
            map(
                lambda j: vocab[j],
                np.argsort(np.squeeze(-beta[i, :].toarray()))[:10].tolist(),
            )
        )
    )

# %%
# Print the confusion matrix (not diagonalized)
predicted_topic = np.argmax(estimation, axis=1)

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
# Print the confusion matrix skipping the first topic
confusion = confusion[:, 1:]
fig, ax = plt.subplots()
ax.imshow(confusion)
ax.set_xticks(np.arange(19))
ax.set_yticks(np.arange(20))
ax.set_yticklabels(list(listvectorizer.vocab.stoi.keys()))

# %%
