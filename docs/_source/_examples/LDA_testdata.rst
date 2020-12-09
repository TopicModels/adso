.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download__examples_LDA_testdata.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr__examples_LDA_testdata.py:


Analyze a very simple dataset with LDA
======================================

import


.. code-block:: default

    import adso
    import matplotlib.pyplot as plt
    import nltk
    import numpy as np








set seed


.. code-block:: default

    adso.set_seed(1234)








Download the dataset and select 1000 random elements


.. code-block:: default

    data = adso.data.load_labelled_test_dataset(lines=True)

    print("Number of documents: ", len(data))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Number of documents:  12




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

    [nltk_data] Downloading package stopwords to /home/tnto/.adso/NLTK...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /home/tnto/.adso/NLTK...
    [nltk_data]   Package punkt is already up-to-date!
    First ten tokens of the first document:
    ['linear', 'algebra', 'studi', 'matric', 'vector', 'vectori', 'space']




Transform the list of tokens in a list of numbers.
We will use the absolute frequency.


.. code-block:: default


    vectorizer = adso.transform.CountVectorizer()









Generate the vocabulary.


.. code-block:: default


    vectorizer.fit(tokens)
    vocab = vectorizer.vocab

    print("Number of words in vocabulary: ", len(vocab))

    print("index of word 'bird': ", vocab["bird"])
    print("word at index 1: ", vocab[1])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Number of words in vocabulary:  50
    index of word 'bird':  1
    word at index 1:  bird




Create the count matrices from tokens.


.. code-block:: default

    count_matrix = vectorizer.transform(tokens)








LDA


.. code-block:: default

    LDA = adso.topicmodel.LDA(n_topic=4, tolerance=1e-3, max_iter=200)
    ret = LDA.fit_transform(count_matrix)
    estimation = ret[0]
    beta = ret[2]
    print("LDA ended after", ret[6], "iterations, achiving a loglikelihood of", ret[5])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Iteration 1 Log-Likelihood -497.4960522897091
    Iteration 2 Log-Likelihood -490.3637651912628
    Iteration 3 Log-Likelihood -488.2187675212483
    Iteration 4 Log-Likelihood -487.4344857012694
    Iteration 5 Log-Likelihood -487.2205134256087
    LDA ended after 5 iterations, achiving a loglikelihood of -487.2205134256087




Check the 10 most characteristic words for each topic


.. code-block:: default


    for i in range(4):
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
    ['dinosaur', 'prove', 'subfield', 'like', 'entiti', 'relat', 'ancient', 'concept', 'wide', 'probabl']
    10 most characteristic words of topic 1
    ['reptil', 'fli', 'space', 'matric', 'wing', 'among', 'wide', 'vector', 'mani', 'probabl']
    10 most characteristic words of topic 2
    ['bird', 'studi', 'vectori', 'sometim', 'egg', 'fli', 'matric', 'mani', 'descend', 'lay']
    10 most characteristic words of topic 3
    ['linear', 'theorem', 'one', 'entiti', 'like', 'use', 'calculus', 'lay', 'vector', 'descend']




Print the confusion matrix (not diagonalized)


.. code-block:: default

    print(beta.todense())
    print(estimation.todense())

    predicted_topic = np.argmax(estimation, axis=1)

    listvectorizer = adso.transform.ListVectorizer()
    labels = list(map(lambda l: [l], data.get_labels()))

    label_topic = np.squeeze(listvectorizer.fit_transform(labels))

    confusion = np.zeros((4, 4))
    for i in zip(label_topic, predicted_topic):
        confusion[i] += 1

    fig, ax = plt.subplots()
    ax.imshow(confusion)
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(list(listvectorizer.vocab.stoi.keys()))




.. image:: /_examples/images/sphx_glr_LDA_testdata_001.png
    :alt: LDA testdata
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[0.34291811 0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.07824034 0.         0.         0.         0.07824034 0.02564681
      0.         0.         0.         0.08572953 0.         0.
      0.         0.         0.         0.         0.         0.08572953
      0.0665698  0.         0.         0.0665698  0.03985091 0.
      0.         0.02494121 0.03899384 0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.0665698 ]
     [0.         0.         0.237522   0.17455686 0.         0.
      0.         0.10275189 0.158348   0.         0.         0.
      0.         0.         0.         0.         0.         0.02726827
      0.         0.         0.         0.         0.02502113 0.
      0.         0.         0.         0.         0.         0.
      0.01769463 0.079174   0.         0.01769463 0.04237039 0.
      0.         0.         0.02072957 0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.079174   0.01769463]
     [0.         0.32315953 0.         0.06425021 0.         0.
      0.16157976 0.05673078 0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.08078988 0.08078988 0.         0.         0.05525809 0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.01674266 0.02928199 0.02289018 0.         0.         0.
      0.02773715 0.08078988 0.         0.         0.         0.
      0.         0.        ]
     [0.         0.         0.         0.         0.17656598 0.
      0.         0.         0.         0.         0.17656598 0.17656598
      0.09599524 0.         0.         0.         0.09599524 0.03146679
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.08828299
      0.06998748 0.03060106 0.         0.         0.         0.
      0.05797327 0.         0.         0.         0.         0.
      0.         0.        ]]
    [[2.56468127e-302 4.43669983e-204 7.40564159e-204 5.55596415e-253]
     [8.57295264e-252 1.00000000e-300 8.07898813e-252 1.76565976e-251]
     [0.00000000e+000 0.00000000e+000 0.00000000e+000 5.50454036e-303]
     [7.82403359e-202 1.58347999e-201 1.61579763e-201 9.59952449e-202]
     [4.46517388e-254 1.40095463e-303 0.00000000e+000 1.69494941e-302]
     [2.65286704e-203 7.49728398e-204 1.67426610e-252 6.17870364e-203]
     [3.42918106e-051 2.37521998e-051 1.00000000e-100 1.00000000e-100]
     [8.55279182e-053 1.00000000e-150 9.46275499e-053 3.06010621e-102]
     [1.33716946e-252 3.61848840e-253 1.47069887e-253 0.00000000e+000]
     [7.82403359e-252 2.37521998e-251 7.24162174e-154 5.56515784e-203]
     [0.00000000e+000 3.04700961e-252 1.33403156e-203 0.00000000e+000]
     [1.78606955e-203 3.32757543e-204 3.23159525e-301 9.59952449e-302]]

    [Text(0, 0, 'linearalgebra'), Text(0, 1, 'geometry'), Text(0, 2, 'dinosaur'), Text(0, 3, 'bird')]




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  8.898 seconds)


.. _sphx_glr_download__examples_LDA_testdata.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: LDA_testdata.py <LDA_testdata.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: LDA_testdata.ipynb <LDA_testdata.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
