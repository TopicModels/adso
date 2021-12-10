import adso
from adso.algorithms import UMAP_HDBSCAN
from adso.corpora import get_20newsgroups
from adso.metrics.supervised import NMI


def test_UMAP_HDBSCAN():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dataset = get_20newsgroups("UH_20news", categories=["sci.space", "rec.autos"])

    u_args = {
        "n_components": 2,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "hellinger",
    }
    model = UMAP_HDBSCAN(u_args=u_args)

    topic_model = model.fit_transform(dataset, "test_simple_UMAP_HDBSCAN")

    assert round(NMI(dataset, topic_model), 5) == 0.24739

    return dataset, topic_model


if __name__ == "__main__":
    import shutil

    try:
        shutil.rmtree(".test/test")
    except FileNotFoundError:
        pass

    test_UMAP_HDBSCAN()
