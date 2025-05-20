#!/usr/bin/env python

if __name__ == "__main__":
    from setuptools import find_packages, setup

    setup(
        name="spacy_layout",
        packages=find_packages(),
        install_requires=[
            "spacy>=3.7.5",
            "docling>=2.5.2",
            "pandas",  # version range set by Docling
            "srsly",  # version range set by spaCy
        ],
        extras_require={
            "azure": [
                "azure-ai-documentintelligence>=1.0.2",
                "azure-core>=1.34.0",
                "python-dotenv>=1.1.0",
            ],
        },
    )
