"""Adso is a topic-modelling library based on pytorch and nltk."""

from . import data, transform, topicmodel
from .common import ADSODIR, set_seed

# Create adso folder
ADSODIR.mkdir(exist_ok=True, parents=True)
