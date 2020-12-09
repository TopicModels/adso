"""Adso is a topic-modelling library based on numpy and nltk."""

from . import data, topicmodel, transform
from .common import ADSODIR, set_seed

# Create adso folder
ADSODIR.mkdir(exist_ok=True, parents=True)
