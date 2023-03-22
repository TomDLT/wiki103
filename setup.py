from setuptools import find_packages, setup

requirements = [
    "numpy",
    "torch",
    "matplotlib",
    "datasets",
    "tqdm",
    "prettytable",
    "Unidecode",
]

if __name__ == "__main__":
    setup(
        name='wiki103',
        version='0.1',
        packages=find_packages(),
        maintainer="Tom Dupre la Tour",
        maintainer_email="tomdlt@berkeley.edu",
        install_requires=requirements,
    )
