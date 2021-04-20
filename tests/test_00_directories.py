from pathlib import Path

import adso


def test_directories():

    assert adso.common.ADSODIR == Path(".adso_test")
    adso.set_adso_dir(".test")
    assert adso.common.ADSODIR == Path(".test")
    adso.set_project_name("test")
    assert adso.common.PROJDIR == Path(".test/test")


if __name__ == "__main__":
    test_directories()
