from setuptools import find_packages, setup

setup(
    name="bbrl_examples",
    packages=[
        package for package in find_packages() if package.startswith("bbrl_examples")
    ],
    url="https://github.com/osigaud/bbrl_examples",
    version="0.0.1",
    install_requires=open("requirements.txt", "r").read().splitlines(),
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
    author="Olivier Sigaud",
    author_email="Olivier.Sigaud@isir.upmc.fr",
    license="MIT",
    description="Examples of RL code with bbrl",
    long_description=open("README.md").read(),
)
