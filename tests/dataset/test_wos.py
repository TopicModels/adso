import shutil

import adso
from adso.corpora import get_wos

try:
    shutil.rmtree(".test/test/wos")
except FileNotFoundError:
    pass


def test_wos():

    adso.set_adso_dir(".test")
    adso.set_project_name("test")
    adso.set_seed(8686)

    get_wos("wos")


if __name__ == "__main__":

    test_wos()
