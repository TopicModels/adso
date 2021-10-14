.. image:: https://codecov.io/gh/TnTo/adso/branch/0.1.0/graph/badge.svg?token=I66EZEIVJB
:target: https://codecov.io/gh/TnTo/adso

adso
====

**A topic modelling library built on top of many great libraries.**

This is a work in progress, undocumented and really experimental

install
^^^^^^^

..
    To install::

        conda install -c conda-forge adso

    Or (with some dependencies missing)::

        pip install adso

To install clone the repo, install dependencies with conda (``environment.yml``) then::

    pip install .

config
^^^^^^

adso need to write some files to disk.
As default adso uses the ``~/.adso`` folder, but it can be change setting the enviromental variable ``ADSODIR`` or with the function ``set_adso_dir()`` from the code (or the REPL).

..
    documentation
    ^^^^^^^^^^^^^

    Documentation with examples is hosted on `GitHub Pages <https://tnto.github.io/adso/index.html>`_

    Some examples on how to use adso are also available in ``tests`` and ``examples`` folders.

documentation
^^^^^^^^^^^^^

Some examples on how to use adso are also available in ``tests`` folders.


