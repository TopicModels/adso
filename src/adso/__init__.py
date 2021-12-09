"""Adso is a topic-modelling library based on numpy and nltk."""

import dask
import ray
from ray.util.dask import ray_dask_get

from . import algorithms, corpora, data, metrics, visualization
from .common import set_adso_dir, set_project_name, set_seed

ray.init()
dask.config.set(scheduler=ray_dask_get)
