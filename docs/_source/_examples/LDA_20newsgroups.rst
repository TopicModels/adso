.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download__examples_LDA_20newsgroups.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr__examples_LDA_20newsgroups.py:


Analyze the 20newsgroups dataset with LDA
=========================================

import


.. code-block:: default

    import random

    import adso
    import matplotlib.pyplot as plt
    import nltk
    import numpy as np








set seed


.. code-block:: default

    adso.set_seed(1234)








Download the dataset and select 1000 random elements


.. code-block:: default

    data = adso.data.load_20newsgroups(split="test")

    new_data = []
    for i in random.sample(range(len(data)), k=1000):
        new_data.append(data[i])
    data = adso.data.LabelledDataset(new_data)

    print("Number of documents: ", len(data))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Number of documents:  1000




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
    ['rtaraz', 'ramin', 'taraz', 'subject', 'wing', 'ding', 'organ', 'worcest', 'polytechn', 'institut']




Transform the list of tokens in a list of numbers.
We will use the absolute frequency.


.. code-block:: default


    vectorizer = adso.transform.CountVectorizer(max_freq=0.7, min_freq=0.1, max_size=1000)









Generate the vocabulary.


.. code-block:: default


    vectorizer.fit(tokens)
    vocab = vectorizer.vocab

    print("Number of words in vocabulary: ", len(vocab))

    print("index of word 'god': ", vocab["god"])
    print("word at index 52: ", vocab[52])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Number of words in vocabulary:  1000
    index of word 'god':  52
    word at index 52:  god




Create the count matrices from tokens.


.. code-block:: default

    count_matrix = vectorizer.transform(tokens)








LDA


.. code-block:: default

    LDA = adso.topicmodel.LDA(n_topic=20, tolerance=0.001, max_iter=100)
    ret = LDA.fit_transform(count_matrix)
    estimation = ret[0]
    beta = ret[2]
    print("LDA ended after", ret[6], "iterations, achiving a loglikelihood of", ret[5])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Iteration 1 Log-Likelihood -2165701.2114364556
    Iteration 2 Log-Likelihood -2160482.3346176264
    Iteration 3 Log-Likelihood -2163212.9631883884
    Iteration 4 Log-Likelihood -2168112.996969577
    Iteration 5 Log-Likelihood -2173911.817309296
    Iteration 6 Log-Likelihood -2180224.827983113
    Iteration 7 Log-Likelihood -2186886.8326180084
    Iteration 8 Log-Likelihood -2193803.660600017
    Iteration 9 Log-Likelihood -2200911.3998690704
    Iteration 10 Log-Likelihood -2208162.4613385475
    Iteration 11 Log-Likelihood -2215519.673691593
    Iteration 12 Log-Likelihood -2222953.241613101
    Iteration 13 Log-Likelihood -2230438.912128744
    Iteration 14 Log-Likelihood -2237956.734111518
    Iteration 15 Log-Likelihood -2245490.1476548053
    Iteration 16 Log-Likelihood -2253025.2760382164
    Iteration 17 Log-Likelihood -2260550.353173516
    Iteration 18 Log-Likelihood -2268055.249179338
    Iteration 19 Log-Likelihood -2275531.072524249
    Iteration 20 Log-Likelihood -2282969.8357664314
    Iteration 21 Log-Likelihood -2290364.1765527423
    Iteration 22 Log-Likelihood -2297707.127931875
    Iteration 23 Log-Likelihood -2304991.933236058
    Iteration 24 Log-Likelihood -2312211.901354372
    Iteration 25 Log-Likelihood -2319360.2985252636
    Iteration 26 Log-Likelihood -2326430.272993087
    Iteration 27 Log-Likelihood -2333414.8090984584
    Iteration 28 Log-Likelihood -2340306.7076238645
    Iteration 29 Log-Likelihood -2347098.589511253
    Iteration 30 Log-Likelihood -2353782.9203738156
    Iteration 31 Log-Likelihood -2360352.0535009955
    Iteration 32 Log-Likelihood -2366798.2892966587
    Iteration 33 Log-Likelihood -2373113.9492324963
    Iteration 34 Log-Likelihood -2379291.4624240845
    Iteration 35 Log-Likelihood -2385323.4628397017
    Iteration 36 Log-Likelihood -2391202.8949087705
    Iteration 37 Log-Likelihood -2396923.124929807
    Iteration 38 Log-Likelihood -2402478.0552294045
    Iteration 39 Log-Likelihood -2407862.237542648
    Iteration 40 Log-Likelihood -2413070.981643885
    Iteration 41 Log-Likelihood -2418100.4549575425
    Iteration 42 Log-Likelihood -2422947.7687891275
    Iteration 43 Log-Likelihood -2427611.047023004
    Iteration 44 Log-Likelihood -2432089.4736624938
    Iteration 45 Log-Likelihood -2436383.316426467
    Iteration 46 Log-Likelihood -2440493.9247364057
    Iteration 47 Log-Likelihood -2444423.7016874366
    Iteration 48 Log-Likelihood -2448176.0509292097
    Iteration 49 Log-Likelihood -2451755.3006263403
    Iteration 50 Log-Likelihood -2455166.607739106
    Iteration 51 Log-Likelihood -2458415.8466764167
    Iteration 52 Log-Likelihood -2461509.486902886
    Iteration 53 Log-Likelihood -2464454.464287028
    Iteration 54 Log-Likelihood -2467258.050912735
    Iteration 55 Log-Likelihood -2469927.7277550315
    Iteration 56 Log-Likelihood -2472471.064093511
    Iteration 57 Log-Likelihood -2474895.606878144
    LDA ended after 57 iterations, achiving a loglikelihood of -2474895.606878144




Check the 10 most characteristic words for each topic


.. code-block:: default


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





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    10 most characteristic words of topic 0
    ['becaus', 'thing', 'problem', 'look', 'better', 'think', 'best', 'ever', 'name', 'real']
    10 most characteristic words of topic 1
    ['use', 'write', 'may', 'first', 'come', 'chang', 'least', 'w', 'wrote', 'mark']
    10 most characteristic words of topic 2
    ['one', 'game', 'even', 'post', 'said', 'usa', 'god', 'ask', 'might', 'opinion']
    10 most characteristic words of topic 3
    ['mean', 'b', 'long', 'nation', 'h', 'anyth', 'littl', 'stand', 'insid', 'goe']
    10 most characteristic words of topic 4
    ['articl', 'ani', 'line', 'could', 'subject', 'way', 'organ', 'state', 'back', 'help']
    10 most characteristic words of topic 5
    ['also', 'new', 'inform', 'thank', 'f', 'noth', 'idea', 'man', 'object', 'exist']
    10 most characteristic words of topic 6
    ['would', 'whi', 'imag', 'live', 'tell', 'day', 'distribut', 'think', 'still', 'version']
    10 most characteristic words of topic 7
    ['interest', 'etc', 'mean', 'littl', 'start', 'reason', 'human', 'term', 'comput', 'third']
    10 most characteristic words of topic 8
    ['q', 'must', 'becom', 'na', 'comput', 'across', 'yes', 'happen', 'anyon', 'ca']
    10 most characteristic words of topic 9
    ['well', 'reason', 'someth', 'follow', 'question', 'find', 'start', 'happen', 'stephanopoulo', 'true']
    10 most characteristic words of topic 10
    ['would', 'onli', 'doe', 'support', 'group', 'cours', 'like', 'kill', 'work', 'applic']
    10 most characteristic words of topic 11
    ['like', 'time', 'good', 'would', 'work', 'believ', 'mani', 'anoth', 'us', 'world']
    10 most characteristic words of topic 12
    ['say', 'set', 'window', 'c', 'person', 'fire', 'respons', 'public', 'law', 'abov']
    10 most characteristic words of topic 13
    ['anyon', 'comput', 'veri', 'someon', 'question', 'say', 'complet', 'yes', 'russian', 'chip']
    10 most characteristic words of topic 14
    ['subject', 'organ', 'line', 'know', 'go', 'right', 'peopl', 'see', 'take', 'year']
    10 most characteristic words of topic 15
    ['get', 'univers', 'write', 'system', 'two', 'much', 'x', 'call', 'program', 'govern']
    10 most characteristic words of topic 16
    ['need', 'make', 'want', 'sinc', 'one', 'talk', 'possibl', 'tri', 'keep', 'enough']
    10 most characteristic words of topic 17
    ['never', 'drive', 'bit', 'human', 'veri', 'involv', 'shall', 'someon', 'nt', 'person']
    10 most characteristic words of topic 18
    ['one', 'make', 'john', 'origin', 'look', 'u', 'befor', 'discuss', 'doe', 'law']
    10 most characteristic words of topic 19
    ['put', 'correct', 'price', 'littl', 'nation', 'detail', 'tank', 'mean', 'h', 'anyth']




Print the confusion matrix (not diagonalized)


.. code-block:: default

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




.. image:: /_examples/images/sphx_glr_LDA_20newsgroups_001.png
    :alt: LDA 20newsgroups
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    [Text(0, 0, 'comp.windows.x'), Text(0, 1, 'rec.motorcycles'), Text(0, 2, 'comp.sys.ibm.pc.hardware'), Text(0, 3, 'talk.politics.mideast'), Text(0, 4, 'comp.graphics'), Text(0, 5, 'comp.sys.mac.hardware'), Text(0, 6, 'sci.crypt'), Text(0, 7, 'sci.med'), Text(0, 8, 'sci.electronics'), Text(0, 9, 'rec.sport.baseball'), Text(0, 10, 'soc.religion.christian'), Text(0, 11, 'rec.autos'), Text(0, 12, 'rec.sport.hockey'), Text(0, 13, 'talk.politics.guns'), Text(0, 14, 'comp.os.ms-windows.misc'), Text(0, 15, 'alt.atheism'), Text(0, 16, 'misc.forsale'), Text(0, 17, 'talk.politics.misc'), Text(0, 18, 'sci.space'), Text(0, 19, 'talk.religion.misc')]



Print the confusion matrix skipping the first topic


.. code-block:: default

    confusion = confusion[:, 1:]
    fig, ax = plt.subplots()
    ax.imshow(confusion)
    ax.set_xticks(np.arange(19))
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(list(listvectorizer.vocab.stoi.keys()))




.. image:: /_examples/images/sphx_glr_LDA_20newsgroups_002.png
    :alt: LDA 20newsgroups
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    [Text(0, 0, 'comp.windows.x'), Text(0, 1, 'rec.motorcycles'), Text(0, 2, 'comp.sys.ibm.pc.hardware'), Text(0, 3, 'talk.politics.mideast'), Text(0, 4, 'comp.graphics'), Text(0, 5, 'comp.sys.mac.hardware'), Text(0, 6, 'sci.crypt'), Text(0, 7, 'sci.med'), Text(0, 8, 'sci.electronics'), Text(0, 9, 'rec.sport.baseball'), Text(0, 10, 'soc.religion.christian'), Text(0, 11, 'rec.autos'), Text(0, 12, 'rec.sport.hockey'), Text(0, 13, 'talk.politics.guns'), Text(0, 14, 'comp.os.ms-windows.misc'), Text(0, 15, 'alt.atheism'), Text(0, 16, 'misc.forsale'), Text(0, 17, 'talk.politics.misc'), Text(0, 18, 'sci.space'), Text(0, 19, 'talk.religion.misc')]




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 13 minutes  3.446 seconds)


.. _sphx_glr_download__examples_LDA_20newsgroups.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: LDA_20newsgroups.py <LDA_20newsgroups.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: LDA_20newsgroups.ipynb <LDA_20newsgroups.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
