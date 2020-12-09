import tempfile

import nox

locations = "src/", "tests/", "noxfile.py", "docs/_source/conf.py"

nox.options.sessions = "black", "lint", "mypy"

ADSODIR = ".adso_test"


def install_this(session):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install("-r", requirements.name)
        session.install(".")


def install_with_constraints(session, *args, **kwargs):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session()
def black(session):
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session()
def lint(session):
    args = session.posargs or locations
    install_with_constraints(
        session,
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


@nox.session()
def mypy(session):
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run("mypy", *args)


@nox.session(python=["3.9", "3.8", "3.7"])
def xdoctest(session):
    args = session.posargs or ["all"]
    install_this(session)
    install_with_constraints(session, "xdoctest")
    session.run("python", "-m", "xdoctest", "adso", *args)


@nox.session(python="3.8")
def cov(session):
    install_this(session)
    install_with_constraints(session, "coverage[toml]", "pytest", "pytest-cov")
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


@nox.session(python=["3.9", "3.8", "3.7", "3.6"])
def test(session):
    install_this(session)
    install_with_constraints(session, "pytest")
    session.run("rm", "-rf", ADSODIR, external=True)
    session.run("pytest", env={"ADSODIR": ADSODIR, "PYTHONWARNINGS": "default"})


@nox.session(python="3.8")
def docs(session):
    install_this(session)
    install_with_constraints(
        session, "sphinx", "sphinx-autodoc-typehints", "sphinx-gallery"
    )
    session.run("sphinx-build", "-b", "html", "docs/_source", "docs")
    session.run("sphinx-build", "-M", "latexpdf", "docs/_source", "docs/_latex")
    session.run("cp", "docs/_latex/latex/adso.pdf", "docs", external=True)
