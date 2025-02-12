from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Easy Language Model Training Library"

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="easylm",
    author="Sayed Shaun",
    version=VERSION,
    packages=find_packages(),
    description=DESCRIPTION,
    install_requires=[
        "sentencepiece",
        "torch"
        "numpy"
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.8",
)