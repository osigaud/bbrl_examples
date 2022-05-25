from setuptools import find_packages, setup

setup(
    name="bbrl_examples",
    packages=[package for package in find_packages() if package.startswith("bbrl_examples")],
    version="0.0.1",
    install_requires=["numpy>=1.19.1", "torch", "hydra", "git+https://github.com/osigaud/bbrl.git@main", "gym==0.21.0", "git+https://github.com/osigaud/my_gym.git"],
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
    description="Examples of RL code with bbrl",
    author="Olivier Sigaud",
    license="MIT",
)
