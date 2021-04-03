"""Adso is a topic-modelling library based on numpy and nltk."""

from . import data, topicmodel, transform
from .common import ADSODIR, PROJDIR, set_seed, set_adso_dir, set_project_name

# Create adso folder
ADSODIR.mkdir(exist_ok=True, parents=True)
PROJDIR.mkdir(exist_ok=True, parents=True)
