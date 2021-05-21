# -*- coding: utf-8 -*-

import os.path

from setuptools import find_packages, setup

readme = ""
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.rst")
if os.path.exists(readme_path):
    with open(readme_path, "rb") as stream:
        readme = stream.read().decode("utf8")


setup(
    long_description=readme,
    name="adso",
    version="0.1.0",
    description="A topic-modelling library",
    python_requires=">=3.7.0",
    url="https://github.com/TnTo/adso",
    project_urls={
        "documentation": "https://tnto.github.io/adso/",
        "repository": "https://github.com/TnTo/adso",
    },
    author="Michele 'TnTo' Ciruzzi",
    author_email="tnto@hotmail.it",
    license="GPL-3.0+",
    packages=find_packages(where="src", include=["adso*"], exclude=["vendor"]),
    package_dir={"": "src"},
    package_data={},
    install_requires=[
        "dask>=2021.5.0",
        "dask-ml>=1.8.0",
        "h5py>=3.0.0",
        "more-itertools>=8.7.0",
        "nltk>=3.6.1",
        "numpy>=1.19.0",
        "pathlib>=1.0.0",
        "scikit-learn>=0.23.2",
        "sparse>=0.12.0",
        "dill>=0.3.0",
        "pyldavis>=3.2.0",
        "pandas>=1.2.0",
        "gensim>=3.8.3",
        "tomotopy>=0.12.0",
    ],
)
