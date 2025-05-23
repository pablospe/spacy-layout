#!/usr/bin/env python

if __name__ == "__main__":
    from setuptools import find_packages, setup

    setup(
        name="spacy-layout-azure",
        packages=find_packages(),
        install_requires=[
            "spacy>=3.7.5",
            "pandas",
            "srsly",  # version range set by spaCy
            "azure-ai-documentintelligence>=1.0.2",
            "azure-core>=1.34.0",
            "python-dotenv>=1.1.0",
        ],
    )
