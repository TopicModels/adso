import nox

locations = "src/", "tests/", "noxfile.py", "docs/_source/conf.py"

nox.options.sessions = "black", "lint", "mypy"

ADSODIR = ".adso_test"

py_version = ["3.8", "3.9", "3.7", "3.6", "pypy3.7", "pypy3.6"]


def install_this(session):
    session.run("conda", "env", "update", "--file", "environment.yml")
    session.install(".")


@nox.session()
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


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


@nox.session(python=py_version[0], venv_backend="conda")
def mypy(session):
    args = session.posargs or locations
    install_this(session)
    session.conda_install("mypy")
    session.run("mypy", *args)


@nox.session(python=py_version, venv_backend="conda")
def xdoctest(session):
    args = session.posargs or ["all"]
    install_this(session)
    session.conda_install("xdoctest")
    session.run("python", "-m", "xdoctest", "adso", *args)


@nox.session(python=py_version[0], venv_backend="conda")
def cov(session):
    install_this(session)
    session.conda_install("coverage[toml]", "pytest", "pytest-cov")
    session.run("rm", "-rf", ADSODIR, external=True)
    session.run(
        "pytest",
        "--cov-report",
        "html:coverage",
        "--cov-report",
        "term",
        "--cov",
        "adso",
        env={"ADSODIR": ADSODIR},
    )


@nox.session(python=py_version, venv_backend="conda")
def test(session):
    install_this(session)
    session.conda_install("pytest")
    session.run("rm", "-rf", ADSODIR, external=True)
    session.run("pytest", env={"ADSODIR": ADSODIR, "PYTHONWARNINGS": "default"})


@nox.session(python=py_version)
def poetry_test(session):
    session.install("poetry", "pytest")
    session.run("rm", "-rf", "poetry.lock", external=True)
    session.run("poetry", "install")
    session.install(".")
    session.run("rm", "-rf", ADSODIR, external=True)
    session.run("pytest", env={"ADSODIR": ADSODIR, "PYTHONWARNINGS": "default"})


@nox.session(python=py_version[0], venv_backend="conda")
def docs(session):
    install_this(session)
    session.conda_install("sphinx", "sphinx-autodoc-typehints", "sphinx-gallery")
    session.run("sphinx-build", "-b", "html", "docs/_source", "docs")
    session.run("sphinx-build", "-M", "latexpdf", "docs/_source", "docs/_latex")
    session.run("cp", "docs/_latex/latex/adso.pdf", "docs", external=True)
