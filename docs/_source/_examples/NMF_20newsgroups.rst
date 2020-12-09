.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download__examples_NMF_20newsgroups.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr__examples_NMF_20newsgroups.py:


Analyze the 20newsgroups dataset with NMF
=========================================

import


.. code-block:: default

    import adso

    import matplotlib.pyplot as plt

    import nltk

    import numpy as np








set seed


.. code-block:: default

    adso.set_seed(1234)








Download the dataset


.. code-block:: default

    data = adso.data.load_20newsgroups(split="test")

    print("Number of documents: ", len(data))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Number of documents:  7532




Tokenize the dataset using a stemmer and a stopwords list, removing punctation


.. code-block:: default


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





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    First ten tokens of the first document:
    ['aidler', 'e', 'alan', 'idler', 'subject', 'doctrin', 'origin', 'sin', 'organ', 'univers']




Transform the list of tokens in a list of numbers.
We will use the frequency and the TFIDF frequency (a correction
for the distribution among the documents).


.. code-block:: default


    freq = adso.transform.FreqVectorizer(max_freq=0.75, max_size=10000)

    tfidf = adso.transform.TFIDFVectorizer(max_freq=0.75, max_size=10000)








Generate the vocabulary and share it between the vectorizer.


.. code-block:: default


    freq.fit(tokens)

    # I will write an ad hoc function later
    vocab = freq.vocab

    print("Number of words in vocabulary: ", len(vocab))

    tfidf.vocab = vocab

    print("index of word 'god': ", vocab["god"])
    print("word at index 32: ", vocab[32])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Number of words in vocabulary:  10000
    index of word 'god':  32
    word at index 32:  god




Create the frequency matrices from tokens.


.. code-block:: default

    freq_matrix = freq.transform(tokens)
    tfidf_matrix = tfidf.transform(tokens)








NMF1 using frequency matrix and ACLS algorithm


.. code-block:: default

    NMF1 = adso.topicmodel.NMF(
        n_topic=20, max_iter=100, tolerance=1e-3, lambdaH=0.001, lambdaW=0.001
    )
    W1, H1, iter1 = NMF1.fit_transform(freq_matrix)
    print("NMF1 ended after", iter1, "iterations")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Iteration 1 - Error 305.1758155027004
    Iteration 11 - Error 257.3421065099152
    Iteration 21 - Error 170.63392034256293
    Iteration 31 - Error 159.9954495702163
    Iteration 41 - Error 154.04000548583855
    Iteration 51 - Error 156.2181911808671
    Iteration 61 - Error 204.83118800820122
    Iteration 71 - Error 162.57089323899004
    Iteration 81 - Error 159.4030973037672
    Iteration 91 - Error 158.828101953583
    NMF1 ended after 100 iterations




NMF2 using frequency matrix and AHCLS algorithm


.. code-block:: default

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





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/tnto/Documenti/Universita/Tesi/src/adso/.nox/docs/lib/python3.8/site-packages/scipy/sparse/linalg/dsolve/linsolve.py:144: SparseEfficiencyWarning: spsolve requires A be CSC or CSR matrix format
      warn('spsolve requires A be CSC or CSR matrix format',
    Iteration 1 - Error 170.77542271706886
    Iteration 11 - Error 150.69746064173484
    Iteration 21 - Error 149.07915383746712
    Iteration 31 - Error 149.05744643993975
    NMF2 ended after 31 iterations




NMF3 using tfidf matrix and ALS algorithm


.. code-block:: default

    NMF3 = adso.topicmodel.NMF(n_topic=20, max_iter=100, tolerance=1e-3, method="ALS")
    W3, H3, iter3 = NMF3.fit_transform(tfidf_matrix)
    print("NMF3 ended after", iter3, "iterations")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Iteration 1 - Error 29161773.19350617
    Iteration 11 - Error 24063771.987029664
    Iteration 21 - Error 23896399.348707188
    Iteration 31 - Error 23888636.76562582
    NMF3 ended after 31 iterations




Check the 10 most characteristic words for the first topic of each model


.. code-block:: default


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




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    10 most characteristic words for the first topic of NMF1
    ['thank', 'ani', 'look', 'help', 'program', 'need', 'pleas', 'advanc', 'could', 'inform']
    10 most characteristic words for the first topic of NMF2
    ['one', 'two', 'want', 'time', 'year', 'onli', 'thing', 'card', 'way', 'line']
    10 most characteristic words for the first topic of NMF3
    ['ppd', 'merc', 'asthma', 'cds', 'nova', 'howland', 'teenag', 'mob', 'disc', 'rob']




Print the confusion matrix (not diagonalized) for NMF1


.. code-block:: default

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




.. image:: /_examples/images/sphx_glr_NMF_20newsgroups_001.png
    :alt: NMF 20newsgroups
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    [Text(0, 0, 'rec.sport.hockey'), Text(0, 1, 'soc.religion.christian'), Text(0, 2, 'rec.motorcycles'), Text(0, 3, 'rec.sport.baseball'), Text(0, 4, 'rec.autos'), Text(0, 5, 'sci.med'), Text(0, 6, 'sci.crypt'), Text(0, 7, 'comp.windows.x'), Text(0, 8, 'sci.space'), Text(0, 9, 'comp.os.ms-windows.misc'), Text(0, 10, 'sci.electronics'), Text(0, 11, 'comp.sys.ibm.pc.hardware'), Text(0, 12, 'misc.forsale'), Text(0, 13, 'comp.graphics'), Text(0, 14, 'comp.sys.mac.hardware'), Text(0, 15, 'talk.politics.mideast'), Text(0, 16, 'talk.politics.guns'), Text(0, 17, 'alt.atheism'), Text(0, 18, 'talk.politics.misc'), Text(0, 19, 'talk.religion.misc')]



Print the confusion matrix (not diagonalized) for NMF2


.. code-block:: default

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




.. image:: /_examples/images/sphx_glr_NMF_20newsgroups_002.png
    :alt: NMF 20newsgroups
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    [Text(0, 0, 'rec.sport.hockey'), Text(0, 1, 'soc.religion.christian'), Text(0, 2, 'rec.motorcycles'), Text(0, 3, 'rec.sport.baseball'), Text(0, 4, 'rec.autos'), Text(0, 5, 'sci.med'), Text(0, 6, 'sci.crypt'), Text(0, 7, 'comp.windows.x'), Text(0, 8, 'sci.space'), Text(0, 9, 'comp.os.ms-windows.misc'), Text(0, 10, 'sci.electronics'), Text(0, 11, 'comp.sys.ibm.pc.hardware'), Text(0, 12, 'misc.forsale'), Text(0, 13, 'comp.graphics'), Text(0, 14, 'comp.sys.mac.hardware'), Text(0, 15, 'talk.politics.mideast'), Text(0, 16, 'talk.politics.guns'), Text(0, 17, 'alt.atheism'), Text(0, 18, 'talk.politics.misc'), Text(0, 19, 'talk.religion.misc')]



Print the confusion matrix (not diagonalized) for NMF3


.. code-block:: default

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



.. image:: /_examples/images/sphx_glr_NMF_20newsgroups_003.png
    :alt: NMF 20newsgroups
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    [Text(0, 0, 'rec.sport.hockey'), Text(0, 1, 'soc.religion.christian'), Text(0, 2, 'rec.motorcycles'), Text(0, 3, 'rec.sport.baseball'), Text(0, 4, 'rec.autos'), Text(0, 5, 'sci.med'), Text(0, 6, 'sci.crypt'), Text(0, 7, 'comp.windows.x'), Text(0, 8, 'sci.space'), Text(0, 9, 'comp.os.ms-windows.misc'), Text(0, 10, 'sci.electronics'), Text(0, 11, 'comp.sys.ibm.pc.hardware'), Text(0, 12, 'misc.forsale'), Text(0, 13, 'comp.graphics'), Text(0, 14, 'comp.sys.mac.hardware'), Text(0, 15, 'talk.politics.mideast'), Text(0, 16, 'talk.politics.guns'), Text(0, 17, 'alt.atheism'), Text(0, 18, 'talk.politics.misc'), Text(0, 19, 'talk.religion.misc')]




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 15 minutes  38.056 seconds)


.. _sphx_glr_download__examples_NMF_20newsgroups.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: NMF_20newsgroups.py <NMF_20newsgroups.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: NMF_20newsgroups.ipynb <NMF_20newsgroups.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
