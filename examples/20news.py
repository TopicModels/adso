import gc

import dask
from dask.distributed import Client

import adso
from adso.corpora import get_20newsgroups

if __name__ == "__main__":

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    dask.config.set({"temporary_directory": str(adso.common.ADSODIR / "dask")})
    client = Client()

    gc.set_threshold(50, 10, 10)

    adso.data.common.nltk_download("punkt")

    def my_tokenizer(doc):
        return list(
            filter(
                lambda s: s.isalpha() and len(s) >= 3,
                adso.data.common.tokenize_and_stem(doc),
            )
        )

    try:
        dataset = adso.data.LabeledDataset.load(".test/test/20news")
    except FileNotFoundError:
        dataset = get_20newsgroups(
            "20news", categories=["sci.space", "rec.autos"], overwrite=True
        )

        # tokenizer = None

        dataset.set_vectorizer_params(
            min_df=5,
            tokenizer=my_tokenizer,
            stopwords=adso.data.common.get_nltk_stopwords(),
            overwrite=True,
        )

    vocab = dataset.get_vocab()
    for w in vocab.compute():
        print(w)
