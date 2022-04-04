from setuptools import find_packages, setup

setup(
    name="my_gym",
    packages=[package for package in find_packages() if package.startswith("my_gym")],
    version="0.0.1",
    install_requires=["gym==0.21.0", "numpy>=1.19.1"],
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
    description="Additional gym library",
    author="Olivier Sigaud",
    license="MIT",
)
