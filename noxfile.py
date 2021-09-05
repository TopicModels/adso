import nox

locations = "src/", "tests/", "noxfile.py", "docs/_source/conf.py"

nox.options.sessions = "black", "lint", "mypy"

ADSODIR = ".adso_test"

py_version = ["3.9", "3.8", "3.7"]


def install_this(session):
    session.conda_install("mamba")
    if session.python:
        with open(f"{session.virtualenv.location}/conda-meta/pinned", "w") as f:
            f.write(f"python={session.python}.*")
    session.run(
        "mamba",
        "env",
        "update",
        "--file",
        "environment.yml",
        "--prefix",
        f"{session.virtualenv.location}",
    )
    session.install(".")


@nox.session()
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", "--exclude", "src/vendor/*", *args)


@nox.session()
def lint(session):
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-import-order",
        "darglint",
    )
    session.run("flake8", *args)


@nox.session(venv_backend="conda")
def mypy(session):
    args = session.posargs or locations
    install_this(session)
    session.conda_install("mypy")
    session.run("python", "-m", "pip", "install", "types-requests")
    session.run("mypy", *args)


@nox.session(python=py_version, venv_backend="conda")
def xdoctest(session):
    args = session.posargs or ["all"]
    install_this(session)
    session.conda_install("xdoctest")
    session.run("python", "-m", "xdoctest", "adso", *args)


@nox.session(venv_backend="conda")
def cov(session):
    install_this(session)
    session.conda_install("coverage", "pytest", "pytest-cov")
    session.run("rm", "-rf", ADSODIR, external=True)
    session.run("rm", "-rf", ".test", external=True)
    session.run(
        "pytest",
        "--cov-config",
        ".coveragerc",
        "--cov-report",
        "html:coverage",
        "--cov-report",
        "term",
        "--cov",
        "adso",
        "-v",
        "--durations=0",
        env={"ADSODIR": ADSODIR},
    )


@nox.session(venv_backend="conda")
def coverage(session):
    install_this(session)
    session.conda_install("coverage", "codecov", "pytest-cov")
    session.run("rm", "-rf", ADSODIR, external=True)
    session.run("rm", "-rf", ".test", external=True)
    session.run(
        "pytest",
        "--cov-config",
        ".coveragerc",
        "--cov",
        "adso",
        env={"ADSODIR": ADSODIR},
    )
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)


@nox.session(python=py_version, venv_backend="conda")
def test(session):
    install_this(session)
    session.conda_install("pytest")
    session.run("rm", "-rf", ADSODIR, external=True)
    session.run("rm", "-rf", ".test", external=True)
    session.run("pytest", env={"ADSODIR": ADSODIR})


@nox.session(python=py_version)
def pip_test(session):
    session.install("pytest")
    session.install(".")
    session.run("rm", "-rf", ADSODIR, external=True)
    session.run("rm", "-rf", ".test", external=True)
    session.run(
        "pytest",
        "--ignore=tests/test_06_LDAGS.py",
        "--ignore=tests/test_09_TM.py",
        "--ignore=tests/test_10_hSBM.py",
        "--ignore=tests/test_12_UMAP_HDBSCAN.py",
        env={"ADSODIR": ADSODIR},
    )


@nox.session(venv_backend="conda")
def docs(session):
    install_this(session)
    session.conda_install("sphinx", "sphinx-autodoc-typehints", "sphinx-gallery")
    session.run("sphinx-build", "-b", "html", "docs/_source", "docs")
    session.run("sphinx-build", "-M", "latexpdf", "docs/_source", "docs/_latex")
    session.run("cp", "docs/_latex/latex/adso.pdf", "docs", external=True)
