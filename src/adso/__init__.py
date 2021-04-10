"""Adso is a topic-modelling library based on numpy and nltk."""

from . import data
from .common import ADSODIR, PROJDIR, set_adso_dir, set_project_name, set_seed

# Create adso folder
ADSODIR.mkdir(exist_ok=True, parents=True)
PROJDIR.mkdir(exist_ok=True, parents=True)
